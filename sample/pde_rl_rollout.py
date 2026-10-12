# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE rollout generation.

from __future__ import annotations

import json
import os
import random
import re
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

from termcolor import cprint
from omegaconf import OmegaConf

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from pde.paths import get_solver_template_path
from pde.prompt_files.codepde_loader import get_generation_prompts, get_pde_description


def get_config():
	"""与 PDE 系列脚本保持一致的配置合并逻辑：base -> pde_type -> cli。"""
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


def build_pde_system_prompt(config) -> str:
	"""构建与 pde_sample_dream.py 对齐的 PDE code-generation prompt（纯文本）。"""
	base_system_prompt, cg_prompt = get_generation_prompts(config)

	pde_type = str(config.dataset.pde_type)
	pde_desc_tmpl = get_pde_description(pde_type, config)

	pde_params: Dict[str, Any] = {}
	if hasattr(config, "pde"):
		try:
			for key, value in (OmegaConf.to_container(config.pde, resolve=True) or {}).items():  # type: ignore[arg-type]
				pde_params[str(key)] = value
				# 允许 {burgers_nu} 这种占位符
				pde_params[f"{pde_type}_{key}"] = value
		except Exception:
			pde_params = {}

	pde_desc_filled = str(pde_desc_tmpl)
	for key, value in pde_params.items():
		pde_desc_filled = pde_desc_filled.replace("{" + str(key) + "}", str(value))

	solver_path = get_solver_template_path(pde_type, config)
	with open(solver_path, "r", encoding="utf-8") as sf:
		solver_template_text = sf.read()

	cg_filled = str(cg_prompt).replace("{pde_description}", pde_desc_filled)
	cg_filled = cg_filled.replace("{solver_template}", solver_template_text)
	return str(base_system_prompt) + "\n" + str(cg_filled)


def get_token_lengths(strings: List[str], tokenizer) -> List[int]:
	"""近似 token 长度：直接 tokenizer.encode。"""
	lengths: List[int] = []
	for s in strings:
		try:
			lengths.append(len(tokenizer.encode(s, add_special_tokens=False)))
		except Exception:
			lengths.append(0)
	return lengths


def _split_even(lst: List[Any], n: int) -> List[List[Any]]:
	k, m = divmod(len(lst), n)
	return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
	config = get_config()
	mp.set_start_method("spawn", force=True)

	os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(getattr(config, "assigned_gpu", "")))

	project_name = str(config.experiment.project)
	current_epoch = int(getattr(config.experiment, "current_epoch", 1) or 1)

	if current_epoch == 1:
		pretrained_model = str(config.model.pretrained_model)
	else:
		pretrained_model = "../" + project_name + "/ckpt/" + str(config.model.optimized_name)

	pde_type = str(config.dataset.pde_type)
	k_sample = int(getattr(config.rollout, "num_response_per_task", 1) or 1)
	batch_size = int(getattr(config.rollout, "batch_size", 1) or 1)

	num_node = int(getattr(config.experiment, "num_node", 1) or 1)
	node_index = int(getattr(config.experiment, "node_index", 0) or 0)
	total_tasks = int(getattr(config.rollout, "num_task_per_step", 1) or 1)
	per_node_tasks = max(1, total_tasks // max(1, num_node))

	system_prompt = build_pde_system_prompt(config)
	
	#########################
	# 生成一个“伪数据集”：PDE prompt 与样本无关，这里只需要控制任务数量。
	data: List[Dict[str, Any]] = []
	for _ in range(per_node_tasks):
		data.append(
			{
				"prompt": system_prompt,
				"full_output": [],
				"extracted_output": [],
				"step_map": [],
				"response_length": [],
			}
		)
	##########################


	# 采样 prompts（每个 task 采样 k 个 response）
	generation_prompts: List[str] = [system_prompt] * (len(data) * k_sample)
	index_list: List[int] = []
	for i in range(len(data)):
		index_list.extend([i] * k_sample)

	cprint(
		f"[pde_rl_rollout] start scratch rollout: tasks={len(data)} k={k_sample} epoch={current_epoch}",
		"green",
	)

	# 复用 pde_sample_dream.py 中的 worker 与 extract_long_code
	import pde_sample_dream as psd
	from dream import DreamTokenizer

	tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

	# shuffle
	N = len(generation_prompts)
	shuffled_idx = list(range(N))
	random.seed(42 + current_epoch + node_index)
	random.shuffle(shuffled_idx)
	shuffled_prompts = [generation_prompts[i] for i in shuffled_idx]

	# gpu split
	n_gpu = torch.cuda.device_count()
	if n_gpu <= 0:
		raise RuntimeError("No CUDA device detected for pde_rl_rollout")

	prompt_chunks = _split_even(shuffled_prompts, n_gpu)
	idx_chunks = _split_even(shuffled_idx, n_gpu)

	manager = mp.Manager()
	seq_dict = manager.dict()
	step_dict = manager.dict()
	procs = []
	for rk in range(n_gpu):
		p = mp.Process(
			target=psd.worker,
			args=(
				pretrained_model,
				rk,
				prompt_chunks[rk],
				idx_chunks[rk],
				seq_dict,
				step_dict,
				batch_size,
				config,
			),
		)
		p.start()
		procs.append(p)

	for p in procs:
		p.join()

	restored_outputs = [seq_dict[i] for i in range(N)]
	restored_step_maps = [step_dict[i] for i in range(N)]

	# token lengths
	response_length = get_token_lengths(restored_outputs, tokenizer)

	# post-process: extract code + write back per task
	for i, full_output in enumerate(restored_outputs):
		extracted_output = psd.extract_long_code(full_output)
		task_i = index_list[i]
		data[task_i]["full_output"].append(full_output)
		data[task_i]["extracted_output"].append(extracted_output)
		data[task_i]["step_map"].append(restored_step_maps[i])
		data[task_i]["response_length"].append(response_length[i])

	if not bool(getattr(config.rollout, "output_unmasking_history", False)):
		for item in data:
			item["step_map"] = []

	# 针对当前 epoch step 保存
	outputs_name = f"rl-{pde_type}-step{current_epoch}"
	if num_node > 1:
		output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
	else:
		output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

	Path(os.path.dirname(output_file_name)).mkdir(parents=True, exist_ok=True)
	with open(output_file_name, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)

	cprint(f"[pde_rl_rollout] saved: {output_file_name}", "green")
