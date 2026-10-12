from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from termcolor import cprint


def _override_rollout_for_refine(cfg) -> None:
	"""在 refine rollout 进程内覆盖 cfg.rollout.* 的采样参数。

	原因：pde_refine_dream.refine_with_dream_model 当前只读取 config.rollout.*。
	为了做到“refine 更探索但不扰动 scratch”，我们只在 refine 脚本进程内用
	refine_rollout.generation.* 覆盖 rollout.*，且缺省不覆盖（保持兼容）。
	"""
	rollout_cfg = getattr(cfg, "rollout", None)
	ref_cfg = getattr(cfg, "refine_rollout", None)
	gen_cfg = getattr(ref_cfg, "generation", None) if ref_cfg is not None else None
	if rollout_cfg is None or gen_cfg is None:
		return

	keys = [
		"steps",
		"temperature",
		"top_p",
		"top_k",
		"target",
		"alg_temp",
		"pad_target_penalty",
		"use_cache",
		"further_horizon",
		"remasking_strategy",
		"dynamic_threshold",
	]

	overrode = False
	for k in keys:
		v = getattr(gen_cfg, k, None)
		if v is None:
			continue
		OmegaConf.update(cfg, f"rollout.{k}", v, merge=False)
		overrode = True

	if overrode:
		try:
			cprint(
				"[pde_rl_rollout_refine] override rollout.* from refine_rollout.generation: "
				+ OmegaConf.to_yaml(gen_cfg, resolve=True).strip().replace("\n", "; "),
				"green",
			)
		except Exception:
			cprint("[pde_rl_rollout_refine] override rollout.* from refine_rollout.generation", "green")


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


def _split_even(lst: List[Any], n: int) -> List[List[Any]]:
	k, m = divmod(len(lst), n)
	return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _safe_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		if isinstance(x, int):
			return x
		if isinstance(x, str) and not x.strip():
			return None
		return int(x)
	except Exception:
		return None


def _read_text_file(path: str) -> str:
	try:
		with open(path, "r", encoding="utf-8", errors="surrogatepass") as f:
			return f.read()
	except Exception:
		return ""


def _load_response_from_execute(exec_dir: str) -> str:
	"""Load the original model response text from an execute directory."""
	resp_path = os.path.join(exec_dir, "responses.txt")
	return _read_text_file(resp_path)


def _parse_error_line_from_execute(exec_dir: str) -> Optional[int]:
	"""Parse 1-based solver.py error line from execute/errors.txt if present."""
	errors_path = os.path.join(exec_dir, "errors.txt")
	text = _read_text_file(errors_path)
	if not text:
		return None

	# Typical pattern: File ".../solver.py", line 18, in solver
	m = re.search(r"solver\.py\", line (\d+)", text)
	if not m:
		return None
	return _safe_int(m.group(1))


def _locate_block_in_code_lines(
	code_lines: List[str],
	error_line_1based: int,
	k_up: int,
	k_down: int,
) -> Tuple[int, int]:
	"""给定代码行与一个 1-based 行号，返回一个扩展窗口 (start_line, end_line)。

	这里不依赖 pde_refine_dream 的同名函数（该模块未暴露该函数），
	采用一个稳定的最小实现：以 error_line 为中心上下扩展 k_up/k_down 行。
	"""
	if not code_lines:
		return 1, 1
	L = len(code_lines)
	line = max(1, min(int(error_line_1based), L))
	start_line = max(1, line - max(0, int(k_up)))
	end_line = min(L, line + max(0, int(k_down)))
	if end_line < start_line:
		end_line = start_line
	return start_line, end_line


def _sample_random_token_span_from_code(
	full_output: str,
	tokenizer,
	rng: random.Random,
	k_up: int,
	k_down: int,
) -> Optional[Tuple[int, int]]:
	"""在 full_output 的 code block 内随机选一块，返回 token span。"""
	import pde_refine_dream as prd

	code_segment = prd.extract_long_code(full_output)
	if not code_segment:
		return None
	code_lines = code_segment.split("\n")

	candidates = [i for i, line in enumerate(code_lines) if line.strip() and not line.strip().startswith("#")]
	if not candidates:
		return None

	error_line = rng.choice(candidates) + 1
	start_line, end_line = _locate_block_in_code_lines(code_lines, error_line, k_up, k_down)
	block_lines = code_lines[start_line - 1 : end_line]
	start_char, end_char = prd.map_block_back_to_response(full_output, block_lines)
	if end_char <= start_char:
		return None

	tok_start, tok_end = prd.map_response_span_to_token_span(tokenizer, full_output, (start_char, end_char))
	if tok_end <= tok_start:
		return None
	return tok_start, tok_end


def _sample_error_token_span_from_execute(
	full_output: str,
	tokenizer,
	exec_dir: str,
	error_line: Optional[int],
	k_up: int,
	k_down: int,
) -> Optional[Tuple[int, int]]:
	"""根据 execute 子目录中的 solver.py + error_line 定位 token span。

	如果定位失败，返回 None（上层再 fallback 到随机 span）。
	"""
	import pde_refine_dream as prd

	solver_path = os.path.join(exec_dir, "solver.py")
	if error_line is None or error_line <= 0:
		return None
	if not os.path.exists(solver_path):
		return None

	start_line, end_line = prd.locate_error_block_in_solver(solver_path, error_line)
	try:
		with open(solver_path, "r", encoding="utf-8", errors="surrogatepass") as f:
			solver_lines = f.readlines()
	except Exception:
		return None

	if start_line < 1 or end_line < start_line or end_line > len(solver_lines):
		return None
	block_lines = solver_lines[start_line - 1 : end_line]

	start_char, end_char = prd.map_block_back_to_response(full_output, block_lines)
	if end_char <= start_char:
		return None

	tok_start, tok_end = prd.map_response_span_to_token_span(tokenizer, full_output, (start_char, end_char))
	if tok_end <= tok_start:
		return None
	return tok_start, tok_end


if __name__ == "__main__":
	# 允许从 sample/ 目录直接运行时 import 到 repo 内其它模块（reward/pde 等）
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	# 注意：不要把 reward/ 目录本身塞进 sys.path。
	# 否则 `import reward` 可能解析成 reward/reward.py（模块）而不是 reward/（包/namespace package），
	# 导致 `from reward.xxx import yyy` 失败（'reward' is not a package）。

	cfg = get_config()
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(getattr(cfg, "assigned_gpu", "")))

	project_name = str(cfg.experiment.project)
	current_epoch = int(getattr(cfg.experiment, "current_epoch", 1) or 1)
	pde_type = str(cfg.dataset.pde_type)

	# ===== 多轮 refine attempt（由 rl_pde.py 驱动） =====
	attempt_raw = OmegaConf.select(cfg, "experiment.refine_attempt", default=None)
	attempt_i = _safe_int(attempt_raw)
	if attempt_raw is None:
		attempt_tag = ""
	else:
		attempt_i = int(attempt_i or 1)
		attempt_tag = f"-a{attempt_i}" if attempt_i > 0 else ""

	multi_attempt_enable = bool(
		OmegaConf.select(cfg, "refine_rollout.multi_attempt.enable", default=False)
	)
	only_bad_samples = bool(
		OmegaConf.select(cfg, "refine_rollout.multi_attempt.only_bad_samples", default=True)
	)

	# refine rollout 只用于 train_refine，相应输出文件使用 -refine 后缀；多 attempt 用额外 tag 隔离
	outputs_name = f"rl-{pde_type}-step{current_epoch}-refine{attempt_tag}"

	# 选择模型 ckpt
	if current_epoch == 1:
		pretrained_model = str(cfg.model.pretrained_model)
	else:
		pretrained_model = "../" + project_name + "/ckpt/" + str(cfg.model.optimized_name)

	# 采样参数
	ref_cfg = getattr(cfg, "refine_rollout", None)
	_override_rollout_for_refine(cfg)
	num_samples = int(getattr(ref_cfg, "num_samples_per_step", 0) or 0)
	good_ratio = float(getattr(ref_cfg, "good_ratio", 0.5) or 0.5)
	k_sample = int(getattr(ref_cfg, "num_response_per_sample", 0) or 0)
	seed = int(getattr(ref_cfg, "seed", 42) or 42)
	k_up = int(getattr(ref_cfg, "k_up", 3) or 3)
	k_down = int(getattr(ref_cfg, "k_down", 3) or 3)
	mask_block_tokens = int(
		getattr(ref_cfg, "mask_block_tokens", None)
		or getattr(getattr(cfg, "rollout", None), "refine_mask_block_tokens", 256)
		or 256
	)

	if num_samples <= 0:
		cprint("[pde_rl_rollout_refine] refine_rollout.num_samples_per_step <= 0, skip", "yellow")
		sampled: List[Dict[str, Any]] = []
	else:
		# 从 refine buffer 中按样本采样
		try:
			from reward.refine_buffer_utils import sample_records
		except Exception:
			# fallback：在极端环境下 reward 被错误解析时，直接按文件路径加载模块
			import importlib.util

			mod_path = os.path.join(repo_root, "reward", "refine_buffer_utils.py")
			spec = importlib.util.spec_from_file_location("refine_buffer_utils", mod_path)
			if spec is None or spec.loader is None:
				raise ImportError(f"Cannot load refine_buffer_utils from {mod_path}")
			_mod = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(_mod)
			sample_records = getattr(_mod, "sample_records")

		# 让每一轮的采样可复现但不同
		seed_eff = seed + 1000 * current_epoch
		sampled = sample_records(
			project_name=project_name,
			num_samples=num_samples,
			good_ratio=good_ratio,
			seed=seed_eff,
			require_round_for_bad=True,
			pde_type=pde_type,
		)

	cprint(
		f"[pde_rl_rollout_refine] sampled {len(sampled)}/{num_samples} from buffer; "
		f"k={k_sample} mask_block_tokens={mask_block_tokens} epoch={current_epoch}",
		"green",
	)

	# 构造 outputs JSON
	data: List[Dict[str, Any]] = []

	import torch
	from dream import DreamTokenizer
	from pde.prompt_files.refine_prompt import refine_prompt
	import pde_refine_dream as prd

	tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
	mask_token_id = None
	try:
		# 与 refine_with_dream_model 保持一致：mask_token_id 来自 DreamModel.config
		model = prd.get_dream_model(pretrained_model)
		mask_token_id = getattr(model.config, "mask_token_id", None)
	except Exception:
		mask_token_id = None
	if mask_token_id is None:
		mask_token_id = 151666

	# attempt 之间要有不同采样随机性；同时保持 buffer 抽样可复现（sample_records 的 seed_eff 不依赖 attempt）
	seed_gen = int(seed) + 1000 * int(current_epoch) + 100000 * int(attempt_i or 1)
	rng = random.Random(seed_gen)
	try:
		random.seed(seed_gen)
		torch.manual_seed(seed_gen)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed_gen)
	except Exception:
		pass

	# 对从 buffer 中采样的问题集合(sampled)逐一处理
	for idx, rec in enumerate(sampled):
		if not isinstance(rec, dict):
			continue

		origin_step = _safe_int(rec.get("origin_step"))
		round_idx = _safe_int(rec.get("origin_round"))
		is_bad = bool(rec.get("is_bad", True))
		origin_pde_type = rec.get("pde_type")
		if origin_step is None or round_idx is None:
			continue

		# attempt>1：可选只对坏样本继续 refine（节省算力）
		if multi_attempt_enable and only_bad_samples and (attempt_i or 1) > 1 and (not is_bad):
			continue

		# 新目录结构：execute/step-{origin_step}/{pde_type}/{round}
		# 兼容旧目录结构：execute/step-{origin_step}/{round}
		exec_dir_new = os.path.abspath(
			os.path.join(
				"..",
				project_name,
				"execute",
				f"step-{origin_step}",
				str(origin_pde_type or pde_type),
				str(round_idx),
			)
		)
		exec_dir_old = os.path.abspath(
			os.path.join("..", project_name, "execute", f"step-{origin_step}", str(round_idx))
		)
		exec_dir = exec_dir_new if os.path.exists(exec_dir_new) else exec_dir_old
		full_output = _load_response_from_execute(exec_dir)
		if not full_output.strip():
			# execute 目录不全/已清理时直接跳过
			continue

		error_line = _parse_error_line_from_execute(exec_dir)

		# bad 样本：优先使用 execute 子目录 + error_line 定位 span
		token_span: Optional[Tuple[int, int]] = None
		if is_bad:
			token_span = _sample_error_token_span_from_execute(
				full_output=full_output,
				tokenizer=tokenizer,
				exec_dir=exec_dir,
				error_line=error_line,
				k_up=k_up,
				k_down=k_down,
			)

		# good 或 bad 定位失败：随机 span
		if token_span is None:
			token_span = _sample_random_token_span_from_code(
				full_output=full_output,
				tokenizer=tokenizer,
				rng=rng,
				k_up=k_up,
				k_down=k_down,
			)
		if token_span is None:
			continue

		masked_ids = prd.build_masked_response(
			tokenizer=tokenizer,
			response=full_output,
			token_span=token_span,
			num_mask_tokens=mask_block_tokens,
			mask_token_id=int(mask_token_id),
		)

		# 将 token-level mask 序列 decode 成文本（refine 前输入），保持与 pde_refine_dream.py 一致
		masked_response_text = tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

		prompt_for_refine = refine_prompt.strip()
		item: Dict[str, Any] = {
			"prompt": prompt_for_refine,
			# refine 前的 masked response（token-level mask decode 后文本），同一条样本的 K 个候选共享
			"masked_response": masked_response_text,
			"full_output": [],
			"extracted_output": [],
			"step_map": [],
			"response_length": [],
			"refine_meta": {
				"buffer_id": rec.get("buffer_id"),
				"is_bad": is_bad,
				"origin_step": origin_step,
				"origin_round": round_idx,
				"attempt": int(attempt_i or 1),
				"sample_key": f"{int(origin_step)}:{str(origin_pde_type or pde_type)}:{int(round_idx)}",
				"error_line": error_line,
				"pde_type": rec.get("pde_type"),
			},
		}

		# 对同一个 masked 输入采样 K 个候选
		for _k in range(max(0, k_sample)):
			try:
				refined_resp = prd.refine_with_dream_model(
					config=cfg,
					tokenizer=tokenizer,
					prompt=prompt_for_refine,
					masked_response_ids=masked_ids,
					pretrained_model=pretrained_model,
				)
			except Exception as e:
				cprint(f"[pde_rl_rollout_refine] refine failed idx={idx}: {e}", "red")
				continue

			extracted = prd.extract_long_code(refined_resp)
			item["full_output"].append(refined_resp)
			item["extracted_output"].append(extracted)
			item["step_map"].append([])
			try:
				item["response_length"].append(len(tokenizer.encode(refined_resp, add_special_tokens=False)))
			except Exception:
				item["response_length"].append(0)

		if item["full_output"]:
			data.append(item)

	# 保存 outputs JSON
	num_node = int(getattr(cfg.experiment, "num_node", 1) or 1)
	node_index = int(getattr(cfg.experiment, "node_index", 0) or 0)
	if num_node > 1:
		output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
	else:
		output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

	Path(os.path.dirname(output_file_name)).mkdir(parents=True, exist_ok=True)
	with open(output_file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)

	cprint(f"[pde_rl_rollout_refine] saved: {output_file_name} (tasks={len(data)})", "green")
