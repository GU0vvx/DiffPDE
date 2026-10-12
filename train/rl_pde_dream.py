# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE reinforcement-learning training.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import wandb
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models import DreamModel, DreamTokenizer
from models.logging import set_verbosity_error, set_verbosity_info
from models.lr_schedulers import get_scheduler
from train.prompting_utils import UniversalPrompting
from train.utils import AverageMeter, flatten_omega_conf, get_config


logger = get_logger(__name__, log_level="INFO")


def _is_wandb_disabled() -> bool:
	return str(os.environ.get("WANDB_MODE", "")).strip().lower() == "disabled"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8", errors="surrogatepass") as f:
		f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _metrics_jsonl_path(project_name: str, current_epoch: int) -> Path:
	return Path(project_name) / "temp_data" / f"train_metrics_step{int(current_epoch)}.jsonl"


class TrainDataset(torch.utils.data.Dataset):
	"""一个最小可用的 PPO-style RL dataset。"""

	def __init__(
		self,
		inputs: torch.Tensor,
		labels: torch.Tensor,
		pmasks: torch.Tensor,
		reward: List[float],
	):
		self.inputs = inputs
		self.labels = labels
		self.pmasks = pmasks
		self.reward = reward

		l_minus1 = inputs.shape[1] - 1
		self.logp_old_tok = torch.full((len(inputs), l_minus1), float("-inf"))
		self.logp_ref_tok = torch.full((len(inputs), l_minus1), float("-inf"))

	def __len__(self) -> int:
		return len(self.inputs)

	def __getitem__(self, idx: int):
		return idx, self.inputs[idx], self.labels[idx], self.pmasks[idx], self.reward[idx]

################################
################################
# Rollout 数据处理相关组件

def _safe_float(x: Any, default: float = 0.0) -> float:
	try:
		return float(x)
	except Exception:
		return float(default)


def _read_json(path: str) -> List[Dict[str, Any]]:
	with open(path, "r") as f:
		return json.load(f)


def resolve_pretrained_model(config, project_name: str, current_epoch: int) -> str:
	if current_epoch == 1:
		return str(config.model.pretrained_model)
	return "./" + project_name + "/ckpt/" + str(config.model.optimized_name)


def maybe_enable_tf32(config) -> None:
	if getattr(config.training, "enable_tf32", False):
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.deterministic = False


def setup_accelerator(config, project_name: str) -> Accelerator:
	config.experiment.logging_dir = str(Path(project_name) / "logs")
	log_with = None if _is_wandb_disabled() else "wandb"
	return Accelerator(
		gradient_accumulation_steps=config.training.gradient_accumulation_steps,
		mixed_precision=config.training.mixed_precision,
		log_with=log_with,
		project_dir=config.experiment.logging_dir,
		split_batches=True,
	)


def setup_logging_and_save_config(config, accelerator: Accelerator, project_name: str) -> None:
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
		# wandb init（支持离线模式：WANDB_MODE=offline）
		# 注意：wandb 在 offline 模式下会显式忽略 resume；若希望跨进程合并，
		# 推荐在父进程统一 init 一个 run，并让子进程写 jsonl 指标由父进程回灌。
		if not _is_wandb_disabled():
			resume_wandb_run = config.wandb.resume
			if isinstance(resume_wandb_run, str) and resume_wandb_run.lower() == "auto":
				# wandb 官方 resume 推荐值为 allow/must；auto 在部分环境下可能不被识别
				resume_wandb_run = "allow"
			run_id = config.wandb.get("run_id", None)
			if run_id is None:
				resume_wandb_run = False
				run_id = wandb.util.generate_id()
				config.wandb.run_id = run_id

			wandb_init_kwargs = dict(
				name=str(project_name),
				id=run_id,
				resume=resume_wandb_run,
				entity=config.wandb.get("entity", None),
				config_exclude_keys=[],
			)
			wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
			wandb_config.pop("experiment.resume_from_checkpoint", None)
			accelerator.init_trackers(
				str(project_name),
				config=wandb_config,
				init_kwargs={"wandb": wandb_init_kwargs},
			)

		os.makedirs(project_name, exist_ok=True)
		config_path = Path(project_name) / "config.yaml"
		logging.info(f"Saving config to {config_path}")
		OmegaConf.save(config, config_path)


################################
################################
# 模型训练初始化组件


def build_model_and_tokenizer(
	config,
	accelerator: Accelerator,
	pretrained_model: str,
) -> Tuple[DreamTokenizer, UniversalPrompting, DreamModel, int, Optional[int]]:
	tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
	uni_prompting = UniversalPrompting(
		tokenizer,
		max_prompt_len=config.training.max_prompt_len,
		max_gen_length=config.training.max_gen_length,
		ignore_id=-100,
	)
	model = DreamModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
	if getattr(config.training, "gradient_checkpointing_enable", False):
		model.gradient_checkpointing_enable()
		if hasattr(model, "config"):
			model.config.use_cache = False
	else:
		model = model.to(accelerator.device)
	mask_id = model.config.mask_token_id
	pad_id = model.config.pad_token_id
	return tokenizer, uni_prompting, model, mask_id, pad_id


def build_ref_model(
	config,
	accelerator: Accelerator,
) -> DreamModel:
	"""加载 reference policy（固定为初始 pretrained_model）。"""
	ref_path = str(config.model.pretrained_model)
	ref_model = DreamModel.from_pretrained(ref_path, torch_dtype=torch.bfloat16)
	ref_model = ref_model.to(accelerator.device)
	ref_model.eval()
	for p in ref_model.parameters():
		p.requires_grad = False
	return ref_model


def build_optimizer(config, model: torch.nn.Module) -> AdamW:
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

	if config.optimizer.name != "adamw":
		raise ValueError(f"Optimizer {config.optimizer.name} not supported")

	return AdamW(
		optimizer_grouped_parameters,
		lr=optimizer_config.learning_rate,
		betas=(optimizer_config.beta1, optimizer_config.beta2),
		weight_decay=optimizer_config.weight_decay,
		eps=optimizer_config.epsilon,
	)


def parse_offline_rollout_entries(
	dataset_entries: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[List[int]], List[float], List[str]]:
	prompt_list: List[str] = []
	response_list: List[str] = []
	step_map_list: List[List[int]] = []
	reward_list: List[float] = []
	sample_type_list: List[str] = []

	for x in dataset_entries:
		prompt_list.append(str(x.get("prompt", "")))
		response_list.append(str(x.get("response", "")))
		reward_list.append(_safe_float(x.get("reward", 0.0)))
		sample_type_list.append(str(x.get("sample_type", "scratch")))

		sm = x.get("step_map", None)
		if isinstance(sm, list) and sm:
			step_map_list.append([int(v) for v in sm])
		else:
			step_map_list.append([])

	return prompt_list, response_list, step_map_list, reward_list, sample_type_list


def compute_tail_pad_mask(
	base_ids: torch.Tensor,
	start_pos: int,
	pad_id: Optional[int],
	post_num: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
	if post_num is not None and pad_id is not None:
		pad_mask_b = base_ids == pad_id
		pad_mask_b[:start_pos] = False
		keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= int(post_num))
		tail_pad_b = pad_mask_b & ~keep_first_pad_b
		return keep_first_pad_b, tail_pad_b
	zeros = torch.zeros_like(base_ids, dtype=torch.bool, device=base_ids.device)
	return zeros, zeros


@torch.no_grad()
def prepare_inputs_and_labels_for_pde(
	config,
	uni_prompting: UniversalPrompting,
	pad_id: Optional[int],
	prompts: List[str],
	responses: List[str],
	step_maps: List[List[int]],
	rewards: List[float],
	mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float], int, int]:
	"""将 PDE rollout (prompt,response,reward,step_map) 转成训练 batch。"""

	input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompts, responses))
	bsz, seq_len = input_ids_lm.shape

	max_gen_len = config.training.max_gen_length
	l_after = min(seq_len, start_pos + max_gen_len)
	input_ids_lm = input_ids_lm[:, :l_after]
	labels_lm = labels_lm[:, :l_after]

	method = str(getattr(config.training, "method", "random_masking"))
	lower = float(getattr(config.training, "lower_p", 0.1))
	upper = float(getattr(config.training, "upper_p", 0.9))
	post_num = getattr(config.training, "post_num", None)

	noisy_list: List[torch.Tensor] = []
	label_list: List[torch.Tensor] = []
	pmask_list: List[torch.Tensor] = []
	reward_out: List[float] = []

	device = input_ids_lm.device

	if method == "TraceRL":
		pass

	elif method in ("random_masking", "coupled"):
		m = int(getattr(config.training, "mask_times_per_sample", 1) or 1)
		for b in range(bsz):
			base_ids = input_ids_lm[b][:l_after]
			label_ids = labels_lm[b]
			rwd = rewards[b]
			_, tail_pad_b = compute_tail_pad_mask(base_ids, start_pos, pad_id, post_num)

			for _ in range(m):
				t = (upper - lower) * torch.rand(1, device=device) + lower
				rand_mask = torch.rand(l_after, device=device) < t
				rand_mask[:start_pos] = False
				rand_mask = rand_mask & ~tail_pad_b[:l_after]

				if rand_mask.any():
					noisy_ids = base_ids.clone()
					noisy_ids[rand_mask] = mask_token_id
					noisy_ids[tail_pad_b[:l_after]] = mask_token_id
					noisy_list.append(noisy_ids)
					label_list.append(label_ids)
					pmask_list.append(rand_mask)
					reward_out.append(rwd)

				if method == "coupled":
					comp_mask = torch.zeros(l_after, device=device, dtype=torch.bool)
					comp_mask[start_pos:] = ~rand_mask[start_pos:]
					comp_mask = comp_mask & ~tail_pad_b[:l_after]
					if comp_mask.any():
						noisy_ids = base_ids.clone()
						noisy_ids[comp_mask] = mask_token_id
						noisy_ids[tail_pad_b[:l_after]] = mask_token_id
						noisy_list.append(noisy_ids)
						label_list.append(label_ids)
						pmask_list.append(comp_mask)
						reward_out.append(rwd)

	else:
		raise ValueError(f"Unsupported training.method={method!r} for PDE RL")

	if not noisy_list:
		raise RuntimeError("No training samples were generated. Check mask parameters / dataset format.")

	noisy_batch = torch.stack(noisy_list)
	labels_batch = torch.stack(label_list)
	pmask_batch = torch.stack(pmask_list)

	valid_rows = pmask_batch.any(dim=1)
	noisy_batch = noisy_batch[valid_rows]
	labels_batch = labels_batch[valid_rows]
	pmask_batch = pmask_batch[valid_rows]
	keep_idx = torch.where(valid_rows)[0].tolist()
	reward_out = [reward_out[i] for i in keep_idx]

	return noisy_batch, labels_batch, pmask_batch, reward_out, start_pos, drop_num


def simple_collate(batch):
	idx, inp, lbl, msk, rwd = zip(*batch)
	return {
		"ids": torch.tensor(idx),
		"input_ids": torch.stack(inp),
		"labels": torch.stack(lbl),
		"p_mask_lm": torch.stack(msk),
		"reward": list(rwd),
	}

def make_attention_mask(input_ids: torch.Tensor, pad_id: Optional[int], start_pos: int) -> torch.Tensor:
	bsz, t = input_ids.shape
	device = input_ids.device
	idx = torch.arange(t, device=device)
	if pad_id is None:
		keep = idx[None, :] <= start_pos
	else:
		keep = (~input_ids.ne(pad_id)) & (idx[None, :] <= start_pos)
	bias = torch.zeros(bsz, 1, 1, t, device=device)
	bias.masked_fill_(keep[:, None, None, :], float("-inf"))
	return bias

# 计算旧 policy 的 token-level logp. 用于后续计算 RL 的 loss (Importance Sampling).
@torch.no_grad()
def compute_logp_old_tok_parallel(
	model: torch.nn.Module,
	accelerator: Accelerator,
	dataset: TrainDataset,
	dataloader,
	start_pos: int,
	pad_id: Optional[int],
) -> None:
	model.eval()
	for batch in dataloader:
		ids = batch["ids"]
		input_ids_b = batch["input_ids"].to(accelerator.device)
		labels_b = batch["labels"].to(accelerator.device)
		attn = make_attention_mask(input_ids_b, pad_id, start_pos)
		logits = model(input_ids_b, attention_mask=attn, is_causal=False).logits
		logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
		tok_lp = logp.gather(-1, labels_b[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]
		dataset.logp_old_tok[ids] = tok_lp.float().cpu()
	accelerator.wait_for_everyone()
	model.train()


@torch.no_grad()
def compute_logp_ref_tok_parallel(
	ref_model: torch.nn.Module,
	accelerator: Accelerator,
	dataset: TrainDataset,
	dataloader,
	start_pos: int,
	pad_id: Optional[int],
) -> None:
	ref_model.eval()
	for batch in dataloader:
		ids = batch["ids"]
		input_ids_b = batch["input_ids"].to(accelerator.device)
		labels_b = batch["labels"].to(accelerator.device)
		attn = make_attention_mask(input_ids_b, pad_id, start_pos)
		logits = ref_model(input_ids_b, attention_mask=attn, is_causal=False).logits
		logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
		tok_lp = logp.gather(-1, labels_b[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]
		dataset.logp_ref_tok[ids] = tok_lp.float().cpu()
	accelerator.wait_for_everyone()


def ppo_forward_loss(
	model: torch.nn.Module,
	config,
	input_ids_b: torch.Tensor,
	labels_b: torch.Tensor,
	p_mask_b: torch.Tensor,
	start_pos: int,
	pad_id: Optional[int],
	adv: List[float],
	logp_old_tok: torch.Tensor,
	logp_ref_tok: Optional[torch.Tensor] = None,
	return_kl: bool = False,
) -> torch.Tensor:
	adv_t = torch.as_tensor(adv, device=input_ids_b.device).detach()

	attn_mask = make_attention_mask(input_ids_b, pad_id, start_pos)
	logits = model(input_ids_b, attention_mask=attn_mask, is_causal=False).logits

    # logits 预测的是下一 token 的分布, 因此 GT labels 右移对齐
	shift_logits = logits[:, :-1, :]
	shift_labels = labels_b[:, 1:]
	shift_mask = p_mask_b[:, 1:]

    # GT labels 的 logp 概率
	logp_new_tok = (
		F.log_softmax(shift_logits, dim=-1)
		.gather(-1, shift_labels.unsqueeze(-1))
		.squeeze(-1)
	)  # [B, T-1]

	ratio = torch.exp(logp_new_tok - logp_old_tok)  # [B, T-1], Importance Sampling ratio
	eps = float(getattr(config.training, "eps", 0.2))
	clipped = torch.clamp(ratio, 1 - eps, 1 + eps)  # [B, T-1]

	adv_tok = adv_t.unsqueeze(1)  # (B,) --> [B,1]
	surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)   # [B, T-1]
	surrogate_tok = surrogate_tok * shift_mask  # token-level: 只计算目标 token 的 logp

	num_mask = torch.clamp(shift_mask.sum(dim=1), min=1)
	surrogate_tok = surrogate_tok.sum(dim=1) / num_mask  # 对 token 数量归一化
	policy_loss = -(surrogate_tok.sum() / input_ids_b.shape[0])    # 对 batch-size 归一化, 并加上负号以最小化

    # 可选 KL 惩罚项
	beta = float(getattr(config.training, "beta", 0.0) or 0.0)
	if beta > 0:
		if logp_ref_tok is None:
			raise ValueError("logp_ref_tok is required when beta > 0")
		kl_tok = logp_new_tok - logp_ref_tok  # [B, T-1]
		if bool(getattr(config.training, "use_kl_estimator_k3", False)):
			kl_tok = (-kl_tok).exp() - 1.0 + kl_tok
		# 只在 mask token 上计算 KL，并按每条样本的 mask token 数归一化
		kl_per_sample = (kl_tok * shift_mask).sum(dim=1) / num_mask  # [B]
		kl_mean = kl_per_sample.mean()  # scalar
		kl_loss = beta * kl_mean
		loss = policy_loss + kl_loss
		if return_kl:
			return loss, kl_mean.detach(), kl_loss.detach()
		return loss
    
    # policy_loss: (B,)
	if return_kl:
		return policy_loss, None, None
	return policy_loss

################################
################################


def save_checkpoint(model, tokenizer, config, accelerator, name: str) -> None:
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
			logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
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

		metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
		with (save_base / "metadata.json").open("w") as f:
			json.dump(metadata, f, indent=2)

		logger.info(f"Saved model + tokenizer to {save_base / name}")

################################
################################
# 主训练流程

def main() -> None:
	"""PDE 任务的 Dream RL 训练脚本骨架（风格：尽量使用文件级函数）。"""
    # -------------------------
	# Read config and setup
	# -------------------------
	config = get_config()
	project_name = config.experiment.project
	current_epoch = int(getattr(config.experiment, "current_epoch", 1) or 1)

	maybe_enable_tf32(config)
	accelerator = setup_accelerator(config, project_name)
	setup_logging_and_save_config(config, accelerator, project_name)

	if getattr(config.training, "seed", None) is not None:
		set_seed(config.training.seed)

	pretrained_model = resolve_pretrained_model(config, project_name, current_epoch)
	logger.info("Loading models and optimizer")
	tokenizer, uni_prompting, model, mask_id, pad_id = build_model_and_tokenizer(
		config,
		accelerator,
		pretrained_model,
	)
	ref_model = build_ref_model(config, accelerator)
	optimizer = build_optimizer(config, model)

	# -------------------------
	# Rollout-Data preprocessing
	# -------------------------
	logger.info("Preprocessing PDE RL data")
	data_file = "./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json"
	dataset_entries = _read_json(data_file)
	prompt_list, response_list, step_map_list, reward_list, _sample_type_list = parse_offline_rollout_entries(
		dataset_entries
	)

	input_ids, labels, p_mask_lm, rewards, start_pos, drop_num = prepare_inputs_and_labels_for_pde(
		config=config,
		uni_prompting=uni_prompting,
		pad_id=pad_id,
		prompts=prompt_list,
		responses=response_list,
		step_maps=step_map_list,
		rewards=reward_list,
		mask_token_id=mask_id,
	)
	dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, rewards)

	# -------------------------
	# Prepare accelerator
	# -------------------------
	logger.info("Preparing model, optimizer and dataloaders")
	total_batch_size_lm = (
		config.training.batch_size_lm
		* accelerator.num_processes
		* config.training.gradient_accumulation_steps
	)
	num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
	num_train_epochs = int(getattr(config.training, "num_train_epochs", 1) or 1)
	num_update_steps_total = num_update_steps_per_epoch * num_train_epochs
	max_train_steps = num_update_steps_total

	lr_scheduler = get_scheduler(
		config.lr_scheduler.scheduler,
		optimizer=optimizer,
		num_training_steps=max_train_steps,
		num_warmup_steps=config.lr_scheduler.params.warmup_steps,
		min_lr_scale=config.lr_scheduler.params.min_lr_scale,
	)

	train_dataloader_lm = torch.utils.data.DataLoader(
		dataset_lm,
		batch_size=config.training.batch_size_lm,
		sampler=None,
		collate_fn=simple_collate,
		num_workers=0,
	)

	model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
		model, optimizer, lr_scheduler, train_dataloader_lm
	)

	# 让 ref model 也经过 accelerate 的 prepare 流程（DDP / device / mixed precision 对齐）
	ref_model = accelerator.prepare(ref_model)
	ref_model.eval()
	# 关键数值打到日志里，方便定位 padding_idx/embedding 尺寸不一致问题
	ref_unwrapped = accelerator.unwrap_model(ref_model)
	try:
		ref_cfg = getattr(ref_unwrapped, "config", None)
		ref_pad_id = getattr(ref_cfg, "pad_token_id", None)
		ref_vocab = getattr(ref_cfg, "vocab_size", None)
		emb = ref_unwrapped.get_input_embeddings() if hasattr(ref_unwrapped, "get_input_embeddings") else None
		emb_num = getattr(emb, "num_embeddings", None) if emb is not None else None
		emb_pad = getattr(emb, "padding_idx", None) if emb is not None else None
		logger.info(
			"[ref_model] tok_len=%s cfg_vocab=%s pad_id=%s emb.num_embeddings=%s emb.padding_idx=%s",
			int(len(tokenizer)),
			int(ref_vocab) if ref_vocab is not None else None,
			int(ref_pad_id) if ref_pad_id is not None else None,
			int(emb_num) if emb_num is not None else None,
			int(emb_pad) if emb_pad is not None else None,
		)
		if ref_pad_id is not None and emb_num is not None and int(ref_pad_id) >= int(emb_num):
			raise RuntimeError(
				"ref_model embedding is too small for pad_token_id: "
				f"pad_id={int(ref_pad_id)} >= num_embeddings={int(emb_num)}. "
				"Fix by aligning tokenizer/model vocab or resizing token embeddings in the pretrained checkpoint."
			)
	except Exception:
		# 不吞错误：让异常直接向上抛出以便观察与修复
		raise

	# -------------------------
	# Training
	# -------------------------
	logger.info("***** Running PDE-RL inference (compute old logp) *****")
	compute_logp_old_tok_parallel(
		model=model,
		accelerator=accelerator,
		dataset=dataset_lm,
		dataloader=train_dataloader_lm,
		start_pos=start_pos,
		pad_id=pad_id,
	)
	logger.info("***** Running PDE-RL inference (compute ref logp) *****")
	compute_logp_ref_tok_parallel(
		ref_model=ref_model,
		accelerator=accelerator,
		dataset=dataset_lm,
		dataloader=train_dataloader_lm,
		start_pos=start_pos,
		pad_id=pad_id,
	)
	# ref logits 已缓存到 CPU，释放 ref_model 显存
	try:
		ref_model.to("cpu")
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	except Exception:
		pass
	del ref_model

	logger.info("***** Running PDE-RL training *****")
	logger.info("***********************************")
	logger.info(f"  Num rollout samples = {len(dataset_entries)}")
	logger.info(f"  Num sample dropped = {drop_num}")
	logger.info(f"  Num training data (after masking) = {input_ids.shape[0]}")
	logger.info("***********************************")	
	logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
	logger.info(f"  Num train devices = {accelerator.num_processes}")
	logger.info(f"  Gradient accumulation steps = {config.training.gradient_accumulation_steps}")
	logger.info(f"  Total train batch size = {total_batch_size_lm}")
	logger.info("***********************************")
	logger.info(f"  Num training update-steps = {num_update_steps_per_epoch}")



	data_time_m = AverageMeter()
	end = time.time()

	from tqdm.auto import tqdm

	global_step = 0
	# 避免 wandb x 轴出现 0..1e6 的大跨度：
	# 让每个 RL step 的 wandb step 基于本轮预计训练步数做一个小偏移，保持单调递增即可。
	step_base = (int(current_epoch) - 1) * int(num_update_steps_total)
	accum_counter = 0

	for epoch in range(num_train_epochs):
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

			with accelerator.accumulate(model):
				accum_counter += 1
				input_ids_b = batch["input_ids"].to(accelerator.device)
				labels_b = batch["labels"].to(accelerator.device)
				p_mask_b = batch["p_mask_lm"].to(accelerator.device)
				old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)
				ref_lp = dataset_lm.logp_ref_tok[batch["ids"].cpu()].to(accelerator.device)
				reward_b = batch["reward"]

				loss_out = ppo_forward_loss(
					model=model,
					config=config,
					input_ids_b=input_ids_b,
					labels_b=labels_b,
					p_mask_b=p_mask_b,
					start_pos=start_pos,
					pad_id=pad_id,
					adv=reward_b,
					logp_old_tok=old_lp,
					logp_ref_tok=ref_lp,
					return_kl=True,
				)
				loss, kl_mean_t, _kl_loss_t = loss_out

				# 正常情况下按 gradient_accumulation_steps 缩放；
				# 但若最后一个 accum 不满，也要按实际累积步数缩放，避免最后一步无更新/无日志。
				denom = int(accelerator.gradient_accumulation_steps)
				if accelerator.sync_gradients:
					denom = max(1, int(accum_counter))
				loss = loss / float(denom)
				accelerator.backward(loss)

				if accelerator.sync_gradients:
					max_grad_norm = getattr(config.training, "max_grad_norm", None)
					if max_grad_norm is not None:
						accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
					optimizer.step()
					lr_scheduler.step()
					optimizer.zero_grad(set_to_none=True)
					global_step += 1
					wandb_step = step_base + global_step
					if accelerator.is_main_process:
						try:
							lr = float(lr_scheduler.get_last_lr()[0])
						except Exception:
							lr = float(getattr(config.optimizer.params, "learning_rate", 0.0))
						try:
							rew_mean = float(torch.as_tensor(reward_b).mean().item())
							rew_std = float(torch.as_tensor(reward_b, dtype=torch.float32).std(unbiased=False).item())
						except Exception:
							rew_mean = 0.0
							rew_std = 0.0
						metrics_payload = {
							"train/loss": float(loss.detach().float().item()),
							"train/lr": lr,
							"train/reward_mean": rew_mean,
							"train/reward_std": rew_std,
							"train/current_epoch": int(current_epoch),
							"rl/step": int(current_epoch),
						}
						# KL 只保留一个指标：batch mean（且已按 mask token 数归一化）的 KL
						if kl_mean_t is not None:
							metrics_payload["train/kl_loss"] = float(kl_mean_t.float().item())
						try:
							accelerator.log(metrics_payload, step=wandb_step)
						except Exception:
							pass
						_append_jsonl(
							_metrics_jsonl_path(project_name, int(current_epoch)),
							{"step": int(wandb_step), "metrics": metrics_payload},
						)
					accum_counter = 0

			end = time.time()

	accelerator.wait_for_everyone()
	save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
	save_every = int(getattr(config.experiment, "save_every", 1) or 1)
	if save_every > 0 and (current_epoch % save_every == 0):
		save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{current_epoch}")
	accelerator.end_training()


if __name__ == "__main__":
    main()

