# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE refinement fine-tuning.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models import DreamTokenizer, DreamModel
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from train.utils import get_config, flatten_omega_conf, AverageMeter
from pde.prompt_files.refine_prompt import refine_prompt

from torch.utils.data import Dataset, DataLoader


try:
    import apex  # noqa: F401

    IS_APEX_AVAILABLE = True
except ImportError:
    IS_APEX_AVAILABLE = False

logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    # 这里额外保存每个样本对应的 start_pos 与 source_type，便于构造 attention mask 及区分数据来源。 
    def __init__(self, inputs, labels, pmasks, start_pos, source_type):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.start_pos = start_pos  # shape: [N]
        self.source_type = source_type  # 0: refine span-masking, 1: random-masking

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.start_pos[idx],
            self.source_type[idx],
        )

# 抓取 responses 中的 代码块
def extract_long_code(full_output: str) -> str:
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return ""

def locate_block_in_code_lines(
    lines: List[str],
    error_line: int,
    k_up: int,
    k_down: int,
) -> Tuple[int, int]:
    if not lines:
        return 1, 1

    n_lines = len(lines)
    idx = max(0, min(error_line - 1, n_lines - 1))

    def indent_level(s: str) -> int:
        stripped = s.lstrip("\t ")
        return len(s) - len(stripped)

    while idx > 0 and not lines[idx].strip():
        idx -= 1

    indent_base = indent_level(lines[idx])

    start_idx = idx
    steps_up = 0
    while start_idx > 0 and steps_up < k_up:
        candidate = start_idx - 1
        stripped = lines[candidate].strip()
        if not stripped or stripped.startswith("#"):
            break
        if '"""' in lines[candidate] or "'''" in lines[candidate]:
            break
        if indent_level(lines[candidate]) < indent_base:
            break
        start_idx = candidate
        steps_up += 1

    end_idx = idx
    steps_down = 0
    while end_idx < n_lines - 1 and steps_down < k_down:
        candidate = end_idx + 1
        stripped = lines[candidate].strip()
        if not stripped or stripped.startswith("#"):
            break
        if '"""' in lines[candidate] or "'''" in lines[candidate]:
            break
        if indent_level(lines[candidate]) < indent_base:
            break
        end_idx = candidate
        steps_down += 1

    return start_idx + 1, end_idx + 1

def map_block_back_to_response(full_output: str, solver_block_lines: List[str]) -> Tuple[int, int]:
    if not solver_block_lines:
        return 0, 0

    matches = list(re.finditer(r"```python(.*?)```", full_output, re.DOTALL))
    if not matches:
        return 0, 0

    best_match = None
    best_len = -1
    for m in matches:
        seg_len = len(m.group(1))
        if seg_len > best_len:
            best_len = seg_len
            best_match = m

    if best_match is None:
        return 0, 0

    code_region_text = best_match.group(1)
    code_region_start = best_match.start(1)

    def norm_line(s: str) -> str:
        return s.rstrip("\n\r ")

    resp_lines = code_region_text.split("\n")
    resp_norm = [norm_line(l) for l in resp_lines]
    solver_norm = [norm_line(l) for l in solver_block_lines]

    block_len = len(solver_norm)
    if block_len == 0:
        return code_region_start, code_region_start

    start_idx_in_code: Optional[int] = None
    for i in range(0, len(resp_norm) - block_len + 1):
        if resp_norm[i : i + block_len] == solver_norm:
            start_idx_in_code = i
            break

    if start_idx_in_code is None:
        block_str = "\n".join(solver_block_lines).strip()
        idx = code_region_text.find(block_str)
        if idx == -1:
            return code_region_start, code_region_start
        start_char = code_region_start + idx
        end_char = start_char + len(block_str)
        return start_char, end_char

    char_offset_start = 0
    for j in range(start_idx_in_code):
        char_offset_start += len(resp_lines[j]) + 1

    char_offset_end = char_offset_start
    for j in range(start_idx_in_code, start_idx_in_code + block_len):
        char_offset_end += len(resp_lines[j])
        if j < start_idx_in_code + block_len - 1:
            char_offset_end += 1

    start_char = code_region_start + char_offset_start
    end_char = code_region_start + char_offset_end
    return start_char, end_char

def map_response_span_to_token_span(
    tokenizer: DreamTokenizer,
    response: str,
    span: Tuple[int, int],
) -> Tuple[int, int]:
    start_char, end_char = span
    start_char = max(0, start_char)
    end_char = max(start_char, min(len(response), end_char))

    try:
        enc = tokenizer(
            response,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc.get("offset_mapping")
        if offsets is not None:
            token_start = None
            token_end = None
            for i, (s, e) in enumerate(offsets):
                if e <= start_char:
                    continue
                if s >= end_char:
                    break
                if token_start is None:
                    token_start = i
                token_end = i + 1
            if token_start is None:
                token_start = 0
            if token_end is None:
                token_end = token_start
            return token_start, token_end
    except (TypeError, NotImplementedError, AttributeError):
        pass

    prefix_enc = tokenizer(response[:start_char], add_special_tokens=False)
    span_enc = tokenizer(response[start_char:end_char], add_special_tokens=False)
    token_start = len(prefix_enc["input_ids"])
    token_end = token_start + len(span_enc["input_ids"])
    return token_start, token_end



# response 字符文本 --> (不重复)采样报错行 --> 定位代码块 --> 映射至 token ids 并输出
def sample_mask_token_span(
    response: str,
    tokenizer: DreamTokenizer,
    rng: random.Random,
    k_up: int,
    k_down: int,
) -> Optional[Tuple[int, int]]:
    code_segment = extract_long_code(response)
    if not code_segment:
        return None

    code_lines = code_segment.split("\n")
    # 只在有效代码行中选择
    candidates = [i for i, line in enumerate(code_lines) if line.strip() and not line.strip().startswith("#")]
    if not candidates:
        return None

    error_line = rng.choice(candidates) + 1
    start_line, end_line = locate_block_in_code_lines(code_lines, error_line, k_up, k_down)
    block_lines = code_lines[start_line - 1 : end_line]
    span = map_block_back_to_response(response, block_lines)
    if span == (0, 0):
        return None
    token_span = map_response_span_to_token_span(tokenizer, response, span)
    if token_span[1] <= token_span[0]:
        return None
    return token_span


# 找到 response 非 padding 部分的有效截止位置
def find_response_end(sample_ids: torch.Tensor, start_pos: int, pad_id: Optional[int]) -> int:
    seq_len = sample_ids.shape[0]
    if pad_id is None:
        return seq_len
    for idx in range(seq_len - 1, start_pos - 1, -1):
        if sample_ids[idx].item() != pad_id:
            return idx + 1
    return start_pos



# 输入样本的 token ids / labels 序列 --> 在 token span 位置“插入”固定长度的 MASK 块，并右移后续 token
# - input 序列在该 span 位置替换为 target_len 个 MASK token
# - label 序列在该 span 位置替换为 [原 span token][pad ...]，总长度同样为 target_len
# - 为了保持总长度不变，若插入后变长，则从序列尾部截断；若变短，则在尾部补 pad
def apply_mask_span_to_tokens(
    sample_ids: torch.Tensor,
    labels: torch.Tensor,
    start_pos: int,
    token_span: Tuple[int, int],
    mask_id: int,
    pad_id: Optional[int],
    target_len: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    # 确保 input_ids 和 label 长度一致
    seq_len = sample_ids.shape[0]
    if labels.shape[0] != seq_len:
        return None

    # 查找 response 有效长度
    resp_end = find_response_end(sample_ids, start_pos, pad_id)
    available = resp_end - start_pos
    if available <= 0:
        return None

    # 裁剪 token span 使其长度不超过 response 有效长度
    rel_start, rel_end = token_span
    rel_start = max(0, min(rel_start, available - 1))
    rel_end = max(rel_start + 1, min(rel_end, available))

    # 需要替换的 block 的长度
    span_len = rel_end - rel_start
    if span_len <= 0:
        return None

    # 目标 MASK 块长度（指定的 refine_mask_block_tokens）
    block_len = target_len
    if block_len <= 0:
        return None

    # 绝对坐标
    abs_start = start_pos + rel_start
    abs_end = start_pos + rel_end
    if abs_start >= abs_end or abs_start >= seq_len:
        return None

    # 前缀 / 后缀切片（注意同时对 input_ids 和 labels 做一致的切分）
    prefix_input = sample_ids[:abs_start]
    suffix_input = sample_ids[abs_end:]

    prefix_label = labels[:abs_start]
    suffix_label = labels[abs_end:]

    # 构造 MASK 块（input）
    mask_block_input = torch.full((block_len,), mask_id, dtype=sample_ids.dtype, device=sample_ids.device)

    # 构造 label 块：span 内真实 token 若不足 block_len，用 pad_id 补足
    if pad_id is None:
        return None
    orig_span_labels = labels[abs_start:abs_end]
    block_label = torch.full((block_len,), pad_id, dtype=labels.dtype, device=labels.device)
    copy_len = min(span_len, block_len)
    block_label[:copy_len] = orig_span_labels[:copy_len]

    # 拼接插入后的完整序列
    new_input = torch.cat([prefix_input, mask_block_input, suffix_input], dim=0)
    new_label = torch.cat([prefix_label, block_label, suffix_label], dim=0)

    # 根据长度差异，在尾部进行截断或补 pad，保持与原 seq_len 相同
    new_len = new_input.shape[0]
    if new_len > seq_len:
        cut = new_len - seq_len
        new_input = new_input[:-cut]
        new_label = new_label[:-cut]
    elif new_len < seq_len:
        pad_size = seq_len - new_len
        pad_tokens = torch.full((pad_size,), pad_id, dtype=new_input.dtype, device=new_input.device)
        new_input = torch.cat([new_input, pad_tokens], dim=0)
        pad_labels = torch.full((pad_size,), pad_id, dtype=new_label.dtype, device=new_label.device)
        new_label = torch.cat([new_label, pad_labels], dim=0)

    # 重新构造 pmask：在整个 MASK 块上计算 loss
    pmask = torch.zeros_like(new_input, dtype=torch.bool)
    block_abs_start = prefix_input.shape[0]
    loss_len = min(block_len, seq_len - block_abs_start)
    if loss_len <= 0:
        return None
    pmask[block_abs_start : block_abs_start + loss_len] = True

    return new_input, new_label, pmask


def build_refine_training_batches(
    config,
    tokenizer,
    uni_prompting,
    prompts: List[str],
    responses: List[str],
    mask_id: int,
    pad_id: Optional[int],
):
    """将一批 (prompt, response) 展开成多个局部掩码训练样本。

    - 先通过 UniversalPrompting 得到拼接好的 [prompt | response] token；
    - 对每条样本调用 sample_mask_token_span + apply_mask_span_to_tokens，生成多份不同 mask 视角；
    """
    input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompts, responses))

    B, L = input_ids_lm.shape
    max_gen_len = config.training.max_gen_length
    if max_gen_len + start_pos < L:
        L_after = start_pos + max_gen_len
    else:
        L_after = L
    input_ids_lm = input_ids_lm[:, :L_after]
    labels_lm = labels_lm[:, :L_after]

    mask_per_sample = getattr(config.training, "refine_masks_per_sample", 1)
    mask_attempts = getattr(config.training, "refine_mask_attempts", mask_per_sample * 4)
    k_up = getattr(config.training, "refine_block_k_up", 4)
    k_down = getattr(config.training, "refine_block_k_down", 4)
    rng = random.Random(getattr(config.training, "refine_mask_seed", config.training.seed or 0))
    mask_block_tokens = getattr(config.training, "refine_mask_block_tokens", 64)

    masked_inputs = []
    masked_labels = []
    masked_pmasks = []

    for idx in range(B):
        response_text = responses[idx]
        sample_success = 0
        attempts = 0
        # 采样 训练用"报错行" 循环
        while sample_success < mask_per_sample and attempts < max(mask_attempts, mask_per_sample):
            attempts += 1
            token_span = sample_mask_token_span(response_text, tokenizer, rng, k_up, k_down)
            if token_span is None:
                continue
            result = apply_mask_span_to_tokens(
                sample_ids=input_ids_lm[idx],
                labels=labels_lm[idx],
                start_pos=start_pos,
                token_span=token_span,
                mask_id=mask_id,
                pad_id=pad_id,
                target_len=mask_block_tokens,
            )
            if result is None:
                continue
            masked_sample, new_label, pmask = result
            masked_inputs.append(masked_sample)
            masked_labels.append(new_label)
            masked_pmasks.append(pmask)
            sample_success += 1

        # 若所有尝试都未能在代码块中成功采样 span，则直接跳过该样本
        if sample_success == 0:
            continue

    if not masked_inputs:
        raise RuntimeError("No refine samples were generated. Check the dataset or mask parameters.")

    input_tensor = torch.stack(masked_inputs)
    label_tensor = torch.stack(masked_labels)
    pmask_tensor = torch.stack(masked_pmasks)
    return input_tensor, label_tensor, pmask_tensor, start_pos, drop_num


def build_random_mask_training_batches(
    config,
    tokenizer,
    uni_prompting,
    prompts: List[str],
    responses: List[str],
    mask_id: int,
    pad_id: Optional[int],
):
    """使用 random-masking 策略构造一批 (input_ids, labels, p_mask_lm)。

    - 先通过 UniversalPrompting 得到 [prompt | response] 的 token 序列；
    - 随机选择若干比例 t \in [lower_p, upper_p]，对 response 区域做多次随机掩码；
    - 每次掩码产生一条新的训练样本，p_mask_lm 仅在被 mask 的位置为 True。
    """
    input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompts, responses))

    B, L = input_ids_lm.shape
    max_gen_len = getattr(config.training, "max_gen_length", L)
    if max_gen_len + start_pos < L:
        L_after = start_pos + max_gen_len
    else:
        L_after = L
    input_ids_lm = input_ids_lm[:, :L_after]
    labels_lm = labels_lm[:, :L_after]

    lower = getattr(config.training, "lower_p", 0.1)
    upper = getattr(config.training, "upper_p", 0.5)
    m = getattr(config.training, "mask_times_per_sample", 1)
    post_num = getattr(config.training, "post_num", None)

    noisy_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    pmask_list: List[torch.Tensor] = []

    device = input_ids_lm.device
    _, L_after = input_ids_lm.shape

    for b in range(B):
        base_ids = input_ids_lm[b]
        label_ids = labels_lm[b]

        if post_num is not None and pad_id is not None:
            pad_mask_b = base_ids == pad_id
            pad_mask_b[:start_pos] = False
            keep_first_pad_b = pad_mask_b & (
                torch.cumsum(pad_mask_b.int(), dim=0) <= post_num
            )
            tail_pad_b = pad_mask_b & ~keep_first_pad_b
        else:
            tail_pad_b = torch.zeros(L_after, dtype=torch.bool, device=device)

        for _ in range(m):
            t = (upper - lower) * torch.rand(1, device=device) + lower
            rand_mask = torch.rand(L_after, device=device) < t
            rand_mask[:start_pos] = False
            rand_mask = rand_mask & ~tail_pad_b

            if not rand_mask.any():
                continue

            noisy_ids = base_ids.clone()
            noisy_ids[rand_mask] = mask_id
            noisy_ids[tail_pad_b] = mask_id

            noisy_list.append(noisy_ids)
            label_list.append(label_ids)
            pmask_list.append(rand_mask)

    if not noisy_list:
        raise RuntimeError("No random-masking samples were generated. Check mask parameters.")

    noisy_batch = torch.stack(noisy_list)
    labels_batch = torch.stack(label_list)
    pmask_batch = torch.stack(pmask_list)

    valid_rows = pmask_batch.any(dim=1)
    noisy_batch = noisy_batch[valid_rows]
    labels_batch = labels_batch[valid_rows]
    pmask_batch = pmask_batch[valid_rows]

    return noisy_batch, labels_batch, pmask_batch, start_pos, drop_num


def simple_collate(batch):
    inp, lbl, msk, spos, stype = zip(*batch)
    return {
        "input_ids": torch.stack(inp),
        "labels": torch.stack(lbl),
        "p_mask_lm": torch.stack(msk),
        "start_pos": torch.stack(spos),
        "source_type": torch.stack(stype),
    }


def make_causal_attention_mask(input_ids, pad_id, start_pos_vec):
    B, T = input_ids.shape
    device = input_ids.device
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1).view(1, 1, T, T)
    bias = torch.zeros((B, 1, T, T), device=device)
    bias.masked_fill_(causal_mask, float("-inf"))
    idx = torch.arange(T, device=device).view(1, T)
    start_pos_vec = start_pos_vec.view(B, 1)
    pad_prefix_mask = (input_ids == pad_id) & (idx < start_pos_vec)
    bias.masked_fill_(pad_prefix_mask.view(B, 1, 1, T), float("-inf"))
    return bias


def make_attention_mask(input_ids, pad_id, pos_vec):
    B, T = input_ids.shape
    device = input_ids.device
    idx = torch.arange(T, device=device).view(1, T)
    pos_vec = pos_vec.view(B, 1)
    keep = ~input_ids.ne(pad_id) & (idx <= pos_vec)
    bias = torch.zeros(B, 1, 1, T, device=device)
    bias.masked_fill_(keep[:, None, None, :], float("-inf"))
    return bias


def forward_process(model, config, input_ids, labels, p_mask_lm, start_pos_vec, pad_id):
    import torch.nn.functional as F

    if config.training.method == "ar":
        attn_mask = make_causal_attention_mask(input_ids, pad_id, start_pos_vec)
        logits = model(input_ids, attention_mask=attn_mask, is_causal=False).logits
    else:
        attention_mask = make_attention_mask(input_ids, pad_id, start_pos_vec)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, is_casual=False).logits

    B, T, _ = logits.shape
    shift_mask = p_mask_lm[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    logp_tok = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    loss_per_sample = - (logp_tok * shift_mask).sum(dim=1)
    mask_num = shift_mask.sum(dim=1).clamp(min=1)
    loss_per_sample = loss_per_sample / mask_num
    loss_mean = loss_per_sample.sum() / B
    return loss_mean, loss_per_sample


def main():
    config = get_config()
    pretrained_model = config.model.pretrained_model

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    logger.info("Loading models and optimizer")
    tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100,
    )
    model = DreamModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
                "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    ####### DATA PROCESS #######
    logger.info("Creating dataloaders and lr_scheduler")
    with open(f"./data/{config.dataset.optimization_data}.json", "r") as f:
        dataset_entries = json.load(f)

    total_raw_samples = len(dataset_entries)
    data_seed = config.training.seed if config.training.seed is not None else 0
    enable_validation = getattr(config.training, "enable_validation", False)
    validation_ratio = float(getattr(config.training, "validation_ratio", 0.0))
    dataset_load = dataset_entries  # 默认全部用于训练
    val_dataset_load = []

    if enable_validation and validation_ratio > 0 and total_raw_samples > 1:
        proposed_val_size = max(1, int(total_raw_samples * validation_ratio))
        val_size = min(total_raw_samples - 1, proposed_val_size)
        if val_size > 0:
            # 均匀选取验证样本（不依赖随机数）
            raw_positions = np.linspace(0, total_raw_samples - 1, num=val_size, dtype=int).tolist()
            seen = set()
            val_indices = []
            for idx in raw_positions:
                idx = int(min(max(idx, 0), total_raw_samples - 1))
                if idx in seen:
                    continue
                val_indices.append(idx)
                seen.add(idx)
            if len(val_indices) < val_size:
                for idx in range(total_raw_samples):
                    if idx in seen:
                        continue
                    val_indices.append(idx)
                    seen.add(idx)
                    if len(val_indices) == val_size:
                        break
            train_indices = [idx for idx in range(total_raw_samples) if idx not in seen]
            val_dataset_load = [dataset_entries[i] for i in val_indices]
            dataset_load = [dataset_entries[i] for i in train_indices]
        else:
            enable_validation = False
    else:
        enable_validation = False

    num_train_samples = len(dataset_load)
    num_val_samples = len(val_dataset_load)
    has_validation = enable_validation and num_val_samples > 0

    # 将 raw_data --> TrainDataset 的过程封装为一个可复用的函数
    def build_masked_dataset(sample_entries, shuffle_seed=None):
        if not sample_entries:
            return None, 0

        refine_prompts_local = [refine_prompt.strip() for _ in sample_entries]
        refine_responses_local = [x["response"] for x in sample_entries]
        input_ids_refine, labels_refine, p_mask_refine, start_pos_refine, drop_num_refine = build_refine_training_batches(
            config=config,
            tokenizer=tokenizer,
            uni_prompting=uni_prompting,
            prompts=refine_prompts_local,
            responses=refine_responses_local,
            mask_id=mask_id,
            pad_id=pad_id,
        )

        pde_prompts_local = [x["prompt"] for x in sample_entries]
        pde_responses_local = [x["response"] for x in sample_entries]
        input_ids_rm, labels_rm, p_mask_rm, start_pos_rm, drop_num_rm = build_random_mask_training_batches(
            config=config,
            tokenizer=tokenizer,
            uni_prompting=uni_prompting,
            prompts=pde_prompts_local,
            responses=pde_responses_local,
            mask_id=mask_id,
            pad_id=pad_id,
        )

        seq_len_refine = input_ids_refine.shape[1]
        seq_len_rm = input_ids_rm.shape[1]
        max_len = max(seq_len_refine, seq_len_rm)

        def pad_to_max_len(tensor, target_len, pad_value):
            if tensor.shape[1] == target_len:
                return tensor
            if tensor.shape[1] > target_len:
                return tensor[:, :target_len]
            pad_size = target_len - tensor.shape[1]
            pad_tensor = torch.full(
                (tensor.shape[0], pad_size),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat([tensor, pad_tensor], dim=1)

        input_ids_refine = pad_to_max_len(input_ids_refine, max_len, pad_id if pad_id is not None else 0)
        labels_refine = pad_to_max_len(labels_refine, max_len, pad_id if pad_id is not None else -100)
        p_mask_refine = pad_to_max_len(p_mask_refine, max_len, False)

        input_ids_rm = pad_to_max_len(input_ids_rm, max_len, pad_id if pad_id is not None else 0)
        labels_rm = pad_to_max_len(labels_rm, max_len, pad_id if pad_id is not None else -100)
        p_mask_rm = pad_to_max_len(p_mask_rm, max_len, False)

        input_ids = torch.cat([input_ids_refine, input_ids_rm], dim=0)
        labels = torch.cat([labels_refine, labels_rm], dim=0)
        p_mask_lm = torch.cat([p_mask_refine, p_mask_rm], dim=0)

        num_refine = input_ids_refine.shape[0]
        num_rm = input_ids_rm.shape[0]
        start_pos_refine_tensor = torch.full((num_refine,), start_pos_refine, dtype=torch.long)
        start_pos_rm_tensor = torch.full((num_rm,), start_pos_rm, dtype=torch.long)
        start_pos_all = torch.cat([start_pos_refine_tensor, start_pos_rm_tensor], dim=0)

        source_type_refine = torch.zeros((num_refine,), dtype=torch.long)
        source_type_rm = torch.ones((num_rm,), dtype=torch.long)
        source_type_all = torch.cat([source_type_refine, source_type_rm], dim=0)

        if shuffle_seed is not None:
            num_samples = input_ids.shape[0]
            g = torch.Generator()
            g.manual_seed(shuffle_seed)
            perm_tokens = torch.randperm(num_samples, generator=g)
            input_ids = input_ids[perm_tokens]
            labels = labels[perm_tokens]
            p_mask_lm = p_mask_lm[perm_tokens]
            start_pos_all = start_pos_all[perm_tokens]
            source_type_all = source_type_all[perm_tokens]

        dataset_split = TrainDataset(input_ids, labels, p_mask_lm, start_pos_all, source_type_all)
        drop_num_split = drop_num_refine + drop_num_rm
        return dataset_split, drop_num_split

    eval_steps = int(getattr(config.training, "eval_steps", 0) or 0)

    dataset_lm, drop_num = build_masked_dataset(dataset_load, shuffle_seed=data_seed)
    if dataset_lm is None:
        raise RuntimeError("No training data available after preprocessing.")

    val_dataset_lm = None
    val_drop_num = 0
    if has_validation:
        val_dataset_lm, val_drop_num = build_masked_dataset(val_dataset_load, shuffle_seed=None)
        if val_dataset_lm is None or len(val_dataset_lm) == 0:
            has_validation = False
            val_dataset_lm = None

    total_batch_size_lm = (
        config.training.batch_size_lm
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0,
    )

    val_dataloader_lm = None
    if has_validation and val_dataset_lm is not None:
        val_dataloader_lm = DataLoader(
            val_dataset_lm,
            batch_size=config.training.batch_size_lm,
            sampler=None,
            collate_fn=simple_collate,
            num_workers=0,
        )

    logger.info("Preparing model, optimizer and dataloaders")
    if val_dataloader_lm is not None:
        (
            model,
            optimizer,
            lr_scheduler,
            train_dataloader_lm,
            val_dataloader_lm,
        ) = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader_lm, val_dataloader_lm
        )
    else:
        model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader_lm
        )

    def run_validation(global_step):
        if not has_validation or val_dataloader_lm is None:
            return

        accelerator.wait_for_everyone()
        model.eval()

        total_sum = 0.0
        total_count = 0
        refine_sum = 0.0
        refine_count = 0
        random_sum = 0.0
        random_count = 0

        with torch.no_grad():
            for batch in val_dataloader_lm:
                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                p_mask_batch = batch["p_mask_lm"].to(accelerator.device)
                start_pos_batch = batch["start_pos"].to(accelerator.device)
                source_type_batch = batch["source_type"].to(accelerator.device)

                _, loss_per_sample = forward_process(
                    model=model,
                    config=config,
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_batch,
                    start_pos_vec=start_pos_batch,
                    pad_id=pad_id,
                )

                loss_detached = loss_per_sample.detach().float()
                gathered_loss = accelerator.gather_for_metrics(loss_detached)
                gathered_source = accelerator.gather_for_metrics(source_type_batch)

                gathered_loss = gathered_loss.cpu()
                gathered_source = gathered_source.cpu()

                total_sum += gathered_loss.sum().item()
                total_count += gathered_loss.numel()

                refine_mask = gathered_source == 0
                if refine_mask.any():
                    refine_sum += gathered_loss[refine_mask].sum().item()
                    refine_count += refine_mask.sum().item()

                random_mask = gathered_source == 1
                if random_mask.any():
                    random_sum += gathered_loss[random_mask].sum().item()
                    random_count += random_mask.sum().item()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and total_count > 0:
            log_items = {
                "val/loss": total_sum / total_count,
            }
            if refine_count > 0:
                log_items["val/loss_infilling"] = refine_sum / refine_count
            if random_count > 0:
                log_items["val/loss_generate-from-scratch"] = random_sum / random_count
            accelerator.log(log_items, step=global_step)

        model.train()

    logger.info("***** Running training *****")
    logger.info("------------------------------")
    logger.info(f"  Num of response (Raw Data) = {total_raw_samples}")
    logger.info(f"  Num of response (Train Split) = {num_train_samples}")
    if has_validation:
        logger.info(f"  Num of response (Validation Split) = {num_val_samples}")
    if has_validation and val_dataset_lm is not None:
        logger.info(f"  Num validation data (After Masked) = {len(val_dataset_lm)}")
        logger.info(f"  Num sample dropped (Validation) = {val_drop_num}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data (After Masked) = {len(dataset_lm)}")
    logger.info("------------------------------")
    logger.info(f"  Num of Process = {accelerator.num_processes}")
    logger.info(
        f"  Instantaneous batch size per device = {config.training.batch_size_lm}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{total_batch_size_lm}"
    )
    logger.info("------------------------------")
    logger.info(f"  Num of Gradient-Update Steps of all Epochs = {max_train_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    from tqdm.auto import tqdm

    global_step = 0
    grad_acc_steps = accelerator.gradient_accumulation_steps
    # loss_stats: dict, {sum_total, count_total, sum_refine, count_refine, sum_random, count_random}
    loss_stats = None
    last_eval_step = -1
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)
            input_ids = batch["input_ids"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            start_pos_batch = batch["start_pos"].to(accelerator.device)
            source_type_batch = batch["source_type"].to(accelerator.device)

            if (step - 1) % grad_acc_steps == 0:
                loss_stats = {
                    "sum_total": 0.0,
                    "count_total": 0,
                    "sum_refine": 0.0,
                    "count_refine": 0,
                    "sum_random": 0.0,
                    "count_random": 0,
                }

            loss_lm, loss_per_sample = forward_process(
                model=model,
                config=config,
                input_ids=input_ids,
                labels=labels,
                p_mask_lm=p_mask_lm,
                start_pos_vec=start_pos_batch,
                pad_id=pad_id,
            )
            loss_for_backward = loss_lm / grad_acc_steps
            accelerator.backward(loss_for_backward)

            with torch.no_grad():
                # 一次梯度更新的总 loss 统计
                loss_detached = loss_per_sample.detach().float()
                loss_stats["sum_total"] += loss_detached.sum().item()
                loss_stats["count_total"] += loss_detached.numel()
                # 分别统计两类样本的 loss
                refine_mask = source_type_batch == 0
                random_mask = source_type_batch == 1
                if refine_mask.any():
                    loss_stats["sum_refine"] += loss_detached[refine_mask].sum().item()
                    loss_stats["count_refine"] += refine_mask.sum().item()
                if random_mask.any():
                    loss_stats["sum_random"] += loss_detached[random_mask].sum().item()
                    loss_stats["count_random"] += random_mask.sum().item()

            if (step + 1) % grad_acc_steps == 0:
                if config.training.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                else:
                    grad_norm = None

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                del input_ids, labels, p_mask_lm, source_type_batch
                torch.cuda.empty_cache()

                global_step += 1
                should_eval = (
                    has_validation
                    and val_dataloader_lm is not None
                    and eval_steps > 0
                    and global_step % eval_steps == 0
                )
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]["lr"]
                    avg_total = loss_stats["sum_total"] / max(loss_stats["count_total"], 1)
                    log_items = {
                        "train/loss": avg_total,
                        "train/lr": current_lr,
                        "train/epoch": epoch + (step / len(train_dataloader_lm)),
                    }
                    if loss_stats["count_refine"] > 0:
                        log_items["train/loss_infilling"] = (
                            loss_stats["sum_refine"] / loss_stats["count_refine"]
                        )
                    if loss_stats["count_random"] > 0:
                        log_items["train/loss_generate-from-scratch"] = (
                            loss_stats["sum_random"] / loss_stats["count_random"]
                        )
                    if grad_norm is not None:
                        log_items["train/grad_norm"] = float(grad_norm)
                    accelerator.log(log_items, step=global_step)

                if should_eval:
                    run_validation(global_step)
                    last_eval_step = global_step

                loss_stats = None

    accelerator.wait_for_everyone()
    if (
        has_validation
        and val_dataloader_lm is not None
        and eval_steps > 0
        and last_eval_step != global_step
    ):
        run_validation(global_step)
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(
                f"removing checkpoints: {', '.join(p.name for p in to_remove)}"
            )
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


if __name__ == "__main__":
    main()
