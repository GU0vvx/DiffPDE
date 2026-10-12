# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE solver generation.
from __future__ import annotations
import math, json, os, sys, time, re, importlib.util
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from termcolor import cprint
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp
import re


if __package__:
    from sample.dream import DreamTokenizer
    from sample.dream.modeling_dream import DreamModel
    from sample.dream.generation_utils_block import DreamGenerationMixin
    from sample.dream.generation_utils_block import DreamGenerationConfig
else:
    from dream import DreamTokenizer
    from dream.modeling_dream import DreamModel
    from dream.generation_utils_block import DreamGenerationMixin
    from dream.generation_utils_block import DreamGenerationConfig
import types


from transformers.utils import ModelOutput
from typing import Any, Dict, Optional, Tuple, Union
import torch.distributions as dists
from dataclasses import dataclass
from torch.nn import functional as F
import torch


from omegaconf import DictConfig, ListConfig, OmegaConf


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pde.paths import get_solver_template_path
from pde.prompt_files.codepde_loader import get_generation_prompts, get_pde_description


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)

    tmp_conf = OmegaConf.merge(yaml_conf, cli_conf)
    data_type = OmegaConf.select(tmp_conf, "dataset.data_type", default=None)
    pde_type = OmegaConf.select(tmp_conf, "dataset.pde_type", default=None)

    conf = yaml_conf
    if data_type == "pde":
        if not pde_type:
            raise ValueError("dataset.pde_type is required when dataset.data_type == 'pde'")
        base_conf_dir = os.path.dirname(os.path.abspath(str(cli_conf.config)))
        pde_conf_path = os.path.join(base_conf_dir, "pde_type", f"{pde_type}.yaml")
        if not os.path.exists(pde_conf_path):
            raise FileNotFoundError(f"PDE config not found for type '{pde_type}': {pde_conf_path}")
        pde_conf = OmegaConf.load(pde_conf_path)
        conf = OmegaConf.merge(conf, {"pde": pde_conf})

    conf = OmegaConf.merge(conf, cli_conf)
    return conf

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):

    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs

    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)
    
    if tar == "confidence":
        return target, x0
    
    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        target = top1_probs - top2_probs 
    
    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)
    
    return target, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


@torch.no_grad()
def _sample(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: float = 0.9,
    frozen_tail_ids: Optional[torch.LongTensor] = None,
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length # prompt + response 总长度
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    # 生成部分使用 mask token 填充
    tail_len = 0
    if frozen_tail_ids is not None:
        if frozen_tail_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("frozen_tail_ids batch size must match input_ids")
        tail_len = frozen_tail_ids.shape[1]

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    if tail_len > 0:
        if tail_len > x.shape[1] - input_ids.shape[1]:
            raise ValueError("frozen tail is longer than the padded region")
        x[:, -tail_len:] = frozen_tail_ids.to(x.device)

    gen_length = max_length - input_ids.shape[1] - tail_len # response 长度, 已剔除 suffix, 只生成中间部分
    if gen_length < 0:
        raise ValueError("max_length is smaller than input length plus frozen tail")
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    

    # base = steps // num_blocks, rem = steps % num_blocks
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)] # 每个 block 的解码步数list
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]


    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        # attention_mask [B, prompt_len], 真实token = 1, padding = 0
        # 这里填充 response 的长度
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        
        # tok_idx : 只对非 padding token 进行位置编码, 从0开始
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)

        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        # attention_mask[:, :, i, j] == True 当且仅当 token i 和 j 均不是 padding
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        # true --> 0.0 , false --> -inf
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), 
                                     torch.tensor(float("-inf"), device=attention_mask.device))
    else: # 全注意力
        tok_idx = None
        attention_mask = "full"
    


    # Initialize cache for the prompt
    past_key_values = None

    # Process each block
    for num_block in range(num_blocks):
        
        # 当前解码块范围
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # define window slice
        # cgws: 在当前 block 之后还能看到多少个 token
        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end) # slice 索引

        # update cache
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Extract only previous block cache
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values

        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)
        
        # logits 生成, 右移一位并在左侧补一个 logits[0], 实现 ar 的预测模式
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        # logits --> token-ids
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        # 只更新当前 block 的起始 token(作为后续块内更新的起点)
        x[:, current_block_start] = x0[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())
        
        
        
        spb = steps_per_block[num_block] # 当前块更新步数, 只用于 unmask_threshold == none 的情况
        i = 1
        while True: # 每个 block 内的解码循环
            
            
            if cgws is not None:
                # 仍为 mask 的 token 位置, 用于后续更新
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)
            
            
            # Prepare attention mask for cached generation
            if attention_mask != "full":
                # Adjust attention mask for current position
                if cgws is not None:
                    # 从 att 矩阵中取一部分
                    # [B, 1, window_size, prompt_len + response_len(up to current block end + cgws)]
                    # 代表当前 window 内的 token 做 query, 能够看到直至 window_end 位置的 token 的 key
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask
            
            if use_cache:
                if cgws is not None:
                    model_output = model(x[:, window_slice], current_attention_mask, 
                                    tok_idx[:, window_slice] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                else:
                    model_output = model(x[:, current_block_start:], current_attention_mask,
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
            
            # 只考虑block内的mask位置 
            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            target, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar)

            # —— pad token penalty ——
            _pad_target_divisor = pad_target_penalty
            _pad_mask_flat = (x0 == pad_token_id)  
            if _pad_mask_flat.any():
                target = target.clone()
                target[_pad_mask_flat] = target[_pad_mask_flat] / _pad_target_divisor


            # target: 根据 logits 计算出 token ids 的置信度
            # 对于 unmask_treshold 的情况, 根据置信度是否超过阈值, 决定是否解码
            if cgws is not None:
                full_target = torch.full_like(x[:, window_slice], -torch.inf, device=model.device, dtype=logits.dtype)
            else:
                full_target = torch.full_like(x[:, current_block_start:], -torch.inf, device=model.device, dtype=logits.dtype)
            
            full_target = full_target.float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf

            if unmask_threshold is None:

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                t = timesteps[num_block][i]
                s = timesteps[num_block][i + 1]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < spb - 1 else int(num_mask_token)
                
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        full_target = full_target / alg_temp
                        full_target = F.softmax(full_target, dim=-1)
                        transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)
                    
                    if cgws is not None:
                        x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)

                    
                    
                    if cgws is not None:
                        x[:, window_slice][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    else:
                        x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    
                
                    
            else:
                # target >= unmask_threshold 作为条件, 置信度足够高的 mask 位置进行解码
                # 若没有位置满足条件, 选择置信度最高位置, 至少解码 1 个 token
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]
                
                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)  # [B]
                no_sel = no_sel & mask_index.any(dim=-1)

                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                xwin[selected_map] = x_candidates[selected_map]

                
            
            if histories is not None:
                histories.append(x.clone().cpu())

            i += 1

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
        
        block_all_pad = torch.all(
            x[:, current_block_start:current_block_end] == pad_token_id
        )
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break

    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x




import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list



def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def extract_long_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = max(matches, key=len).strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output



def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]       
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0):  
        cur = snap[sample_idx]       
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:    
            break
    return step_map



from tqdm import tqdm




def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, batch_size, config, gpu_time_dict=None):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load model once
    model_gpu = (DreamModel.from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())
    model_gpu.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model_gpu)
    model_gpu._sample = types.MethodType(DreamGenerationMixin._sample, model_gpu)   
    tokenizer_gpu = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    pad_id = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_id = tokenizer_gpu.convert_tokens_to_ids("<|im_end|>")

    # process in chunks of `batch_size`
    t0 = time.time()
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # tokenize & move to GPU
        enc = tokenizer_gpu(batch_prompts,
                            padding=True,
                            return_tensors="pt", padding_side="left")
        prompt_ids = enc["input_ids"].to(device)

        attn_mask = prompt_ids.ne(pad_id)
        attn_mask = attn_mask.to(device=model_gpu.device)

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None

        output_history = bool(getattr(config.rollout, "output_unmasking_history", False))

        generation_config = DreamGenerationConfig(
            output_history=output_history,
            return_dict_in_generate=True,   
            max_length=config.rollout.max_gen_length + prompt_ids.shape[1],     
            steps=config.rollout.steps,                  
            temperature=config.rollout.temperature,  
            top_p=config.rollout.top_p,               
            top_k=config.rollout.top_k,            
            tar=config.rollout.target,               
            alg_temp=config.rollout.alg_temp,        
        )
        
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        generation_ids = _sample(
            model_gpu,
            prompt_ids,
            attention_mask=attn_mask,
            generation_config=generation_config,
            block_length=config.rollout.block_size,
            use_cache=config.rollout.use_cache,
            further_horizon=config.rollout.further_horizon,
            mask_token_id = mask_id,
            eos_token_id = eos_id,
            pad_token_id = pad_id,
            pad_target_penalty = config.rollout.pad_target_penalty,
            unmask_threshold = unmask_threshold
        )
        generation_ids.sequences = generation_ids.sequences.cpu()
        torch.cuda.empty_cache()

        # decode
        seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
        texts   = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        eos_text = tokenizer_gpu.eos_token
        if eos_text is not None:
            texts = [t.split(eos_text)[0] for t in texts]

        # Computing denoising step maps requires storing the full generation
        # history, which is unnecessary for PDE evaluation and slows inference.
        for i, idx in enumerate(batch_idxs):
            if output_history and generation_ids.history is not None:
                m = denoise_step_map(generation_ids.history, mask_id=mask_id, sample_idx=i)
                step_map = m[prompt_ids.shape[1]:].tolist()
            else:
                step_map = []
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map

        # free unused GPU cache
        torch.cuda.empty_cache()

    # 记录当前 GPU 完成其负责所有 prompts 的总耗时
    if gpu_time_dict is not None:
        gpu_time_dict[int(rank)] = float(time.time() - t0)

def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


if __name__ == "__main__":

    config = get_config()

    mp.set_start_method("spawn", force=True)

    
    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size
    system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
    
    project_name = config.experiment.project

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    
    if config.dataset.data_type == "pde":
        base_system_prompt, cg_prompt = get_generation_prompts(config)
        pde_type = config.dataset.pde_type
        pde_desc_tmpl = get_pde_description(pde_type, config)

        pde_params = {}
        if hasattr(config, "pde"):
            for key, value in OmegaConf.to_container(config.pde, resolve=True).items():
                pde_params[key] = value
                pde_params[f"{pde_type}_{key}"] = value

        pde_desc_filled = str(pde_desc_tmpl)
        for key, value in pde_params.items():
            placeholder = "{" + str(key) + "}"
            pde_desc_filled = pde_desc_filled.replace(placeholder, str(value))

        solver_path = get_solver_template_path(str(pde_type), config)
        with open(solver_path, "r", encoding="utf-8") as sf:
            solver_template_text = sf.read()

        cg_filled = cg_prompt.replace("{pde_description}", pde_desc_filled)
        cg_filled = cg_filled.replace("{solver_template}", solver_template_text)
        # 为了与 PDE SFT 训练时的 prompt 形式对齐，这里不再加入对话式特殊 token，直接使用纯文本 system prompt
        system_prompts = str(base_system_prompt) + "\n" + str(cg_filled)
    
    
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)
    
    num = len(data)

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)




    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        # preprocess
        # PDE 场景下 prompt 不依赖单个样本内容，直接使用统一的 system_prompts
        generation_prompts = generation_prompts + [system_prompts] * k_sample

        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = system_prompts

    

    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    # 固定随机种子
    random.seed(42)
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    # --------------------- 2. split to each GPU ----------------------
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 1, "need >=2 GPUs for parallel inference"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx,     n_gpu)

    

    # ------------------- 4. launch all workers -----------------------
    manager      = mp.Manager()
    seq_dict     = manager.dict()   # {shuffled_pos: text}
    step_dict    = manager.dict()   # {shuffled_pos: step_map}
    gpu_time_dict = manager.dict()  # {gpu_rank: total_time_seconds}
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                args=(pretrained_model, rk,
                    prompt_chunks[rk],
                    idx_chunks[rk],
                    seq_dict,
                    step_dict,
                    batch_size,
                    config,
                    gpu_time_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]

    # 汇总每个 GPU 的总耗时，并计算“总工作量时间”（各 GPU 时间之和）
    per_gpu_times = {int(k): float(v) for k, v in gpu_time_dict.items()}
    total_generation_time = sum(per_gpu_times.values())

    cprint("generation job done!", "green")
    for gid in sorted(per_gpu_times.keys()):
        cprint(f"[timing] GPU {gid}: {per_gpu_times[gid]:.2f} s", "cyan")
    cprint(f"[timing] Total generation time (sum over GPUs): {total_generation_time:.2f} s", "cyan")






    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)

    # 计算平均每条 response 的生成用时, 使用“各 GPU 总时间之和 / response 总数”
    num_responses = len(restored_outputs)
    total_tokens = sum(response_length) if response_length else 0
    avg_time_per_response = (
        total_generation_time / num_responses if num_responses > 0 else 0.0
    )
    tokens_per_second = (
        total_tokens / total_generation_time if total_generation_time > 0 else 0.0
    )

    # 将生成阶段的速度指标单独落盘到 project/results 目录
    results_dir = os.path.join("..", project_name, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "generation_metrics.json")

    metrics = {
        "model": pretrained_model,
        "dataset": dataset,
        "data_type": config.dataset.data_type,
        "pde_type": getattr(config.dataset, "pde_type", None),
        "num_node": config.experiment.num_node,
        "node_index": config.experiment.node_index,
        "num_samples": num,
        "num_responses": num_responses,
        "per_gpu_times_sec": per_gpu_times,
        "total_generation_time_sec": total_generation_time,
        "avg_time_per_response_sec": avg_time_per_response,
        "avg_tokens_per_response": mean_response_length,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
    }

    with open(metrics_path, "w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, indent=2, ensure_ascii=False)




    # process generated codes
    i = 0
    for full_output in restored_outputs:
        if config.dataset.data_type == "pde":
            extracted_output = extract_long_code(full_output)
        
        
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
