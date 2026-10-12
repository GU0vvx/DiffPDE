from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf
from termcolor import cprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from pde.paths import get_evaluator_path, get_evaluator_python


def _safe_int(x: Any) -> int | None:
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


def _extract_best_python_code_region(full_output: str) -> tuple[int, int, str] | None:
	"""返回 full_output 中最长的 ```python ... ``` 代码块的 (start, end, code_text)。"""
	import re

	matches = list(re.finditer(r"```python(.*?)```", full_output, re.DOTALL))
	if not matches:
		return None
	best = max(matches, key=lambda m: len(m.group(1) or ""))
	return best.start(1), best.end(1), best.group(1)


def _mask_response_with_solver_block(
	full_output: str,
	solver_block_lines: list[str],
	placeholder_line: str,
) -> tuple[str | None, dict[str, Any] | None]:
	"""在 response 的 python 代码块中，用 solver_block_lines 对应窗口做行级替换。

	这是监控用途的近似 masked response（不保证与 token-level mask 完全一致）。
	"""
	region = _extract_best_python_code_region(full_output)
	if region is None:
		return None, None
	start, end, code_text = region

	def norm_line(s: str) -> str:
		return " ".join((s or "").strip().split())

	code_lines = code_text.split("\n")
	solver_lines = [l.rstrip("\n") for l in solver_block_lines]
	code_norm = [norm_line(l) for l in code_lines]
	solver_norm = [norm_line(l) for l in solver_lines if norm_line(l)]
	if not solver_norm:
		return None, None

	block_len = len(solver_norm)
	pos = None
	for i in range(0, len(code_norm) - block_len + 1):
		if code_norm[i : i + block_len] == solver_norm:
			pos = i
			break
	if pos is None:
		# 找不到就不强行 mask，避免误替换
		return None, None

	new_lines = list(code_lines)
	new_lines[pos : pos + block_len] = [placeholder_line]
	new_code_text = "\n".join(new_lines)
	new_full = full_output[:start] + new_code_text + full_output[end:]
	meta = {
		"approx_method": "line_mask",
		"code_region_char_span": [int(start), int(end)],
		"masked_block_code_lines": [int(pos + 1), int(pos + block_len)],
		"masked_block_len_lines": int(block_len),
	}
	return new_full, meta


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


def _extract_error_info(errors_path: str) -> Tuple[str, str]:
	import re

	error_type = "none"
	error_line = ""
	if not os.path.exists(errors_path):
		return error_type, error_line
	try:
		with open(errors_path, "r", encoding="utf-8", errors="surrogatepass") as f:
			content = f.read()
	except Exception:
		return error_type, error_line

	type_match = re.search(r"([A-Za-z]+Error):", content)
	if type_match:
		error_type = type_match.group(1)

	line_matches = re.findall(r'File ".*?/solver\.py", line (\d+)', content)
	if line_matches:
		error_line = line_matches[-1]

	return error_type, error_line


def _get_last_nonempty_line(path: str) -> str | None:
	if not os.path.exists(path):
		return None
	with open(path, "r", encoding="utf-8", errors="surrogatepass") as f:
		lines = f.readlines()
	for line in reversed(lines):
		if line.strip():
			return line
	return None


def _parse_results_from_line(line: str) -> Tuple[float, float, float]:
	import re

	rel_err = float(re.search(r"nRMSE: (.*?)\t", line).group(1))
	elapsed = float(re.search(r"Time: (.*?)s", line).group(1))
	avg_rate = float(re.search(r"Average convergence rate: (.*?)\t", line).group(1))
	return rel_err, elapsed, avg_rate


def main() -> None:
	##############################
	##############################
	# 定位配置文件 + 读取 outputs JSON文件
	cfg = get_config()
	evaluator_python = get_evaluator_python(cfg)
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(getattr(cfg, "assigned_gpu", "")))

	project_name = str(cfg.experiment.project)
	current_epoch = int(getattr(cfg.experiment, "current_epoch", 1) or 1)
	pde_type = str(cfg.dataset.pde_type)
	phase = str(getattr(cfg.experiment, "function", ""))

	# refine attempt（多轮 refine 时用于隔离 outputs/execute 目录）
	attempt_raw = OmegaConf.select(cfg, "experiment.refine_attempt", default=None)
	attempt_i = _safe_int(attempt_raw) if attempt_raw is not None else None

	num_node = int(getattr(cfg.experiment, "num_node", 1) or 1)
	node_index = int(getattr(cfg.experiment, "node_index", 0) or 0)

	# 针对当前 epoch step 保存（refine 相位使用独立文件，避免覆盖 scratch）
	suffix = ""
	if phase in {"train_refine", "refine"}:
		suffix = "-refine"

	attempt_tag = ""
	if suffix == "-refine" and attempt_i is not None and int(attempt_i) > 0:
		attempt_tag = f"-a{int(attempt_i)}"

	# ===== refine 监控参数（行级近似 masked response） =====
	masking_k_up = 3
	masking_k_down = 3
	mask_placeholder_line = "# === MASKED_BLOCK ==="
	if suffix == "-refine":
		try:
			ref_cfg = getattr(cfg, "refine_rollout", None)
			masking_k_up = int(getattr(ref_cfg, "k_up", 3) or 3)
			masking_k_down = int(getattr(ref_cfg, "k_down", 3) or 3)
		except Exception:
			masking_k_up, masking_k_down = 3, 3
	outputs_name = f"rl-{pde_type}-step{current_epoch}{suffix}{attempt_tag}"
	if num_node > 1:
		file_name = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
	else:
		file_name = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

	cprint(f"[pde_rl_execute] loading {file_name}", "green")
	with open(file_name, "r", encoding="utf-8") as f:
		data: List[Dict[str, Any]] = json.load(f)
	##############################
	##############################

	##############################
	##############################
	# 构建执行目录 + 评测环境
	
	# 执行目录隔离到 step/phase/pde_type 级别：
	# - 同一个 RL step 的所有候选都落到同一个 step 目录下
	# - 不同 PDE 类型用子目录隔离，避免 multi_pde 时互相覆盖
	exec_dir_name = f"step-{current_epoch}{suffix}{attempt_tag}"
	exec_step_root = os.path.abspath(os.path.join("..", project_name, "execute", exec_dir_name))
	exec_root = os.path.join(exec_step_root, pde_type)
	Path(exec_root).mkdir(parents=True, exist_ok=True)

	# 每个 PDE 子目录各自一份（便于单独查看）
	result_sheet_path = os.path.join(exec_root, "test_results.csv")
	with open(result_sheet_path, "w", encoding="utf-8", errors="surrogatepass") as f:
		f.write(
			"pde_type,round,task_idx,cand_idx,nRMSE,elapsed_time,convergence_rate,status,error_type,error_line\n"
		)

	# 同一个 step 下的总表：所有 PDE 的结果汇总到这里
	combined_sheet_path = os.path.join(exec_step_root, "test_results_all.csv")
	if not os.path.exists(combined_sheet_path):
		with open(combined_sheet_path, "w", encoding="utf-8", errors="surrogatepass") as f_all:
			f_all.write(
				"pde_type,round,task_idx,cand_idx,nRMSE,elapsed_time,convergence_rate,status,error_type,error_line\n"
			)

	# 写入子目录与索引映射
	counter = 0
	counter_map: Dict[int, Tuple[int, int]] = {}
	for task_idx, item in enumerate(data):
		full_outputs = item.get("full_output", []) or []
		extracted_outputs = item.get("extracted_output", []) or []
		m = min(len(full_outputs), len(extracted_outputs))
		item.setdefault("pde_metrics", [None] * m)
		item.setdefault("correctness", [False] * m)

		# ===== refine 监控信息：优先从 outputs JSON 的 refine_meta 回溯 scratch execute =====
		refine_meta = item.get("refine_meta") if isinstance(item, dict) else None
		origin_exec_dir = None
		origin_step = None
		origin_round = None
		origin_error_line = None
		origin_pde_type = None
		if suffix == "-refine" and isinstance(refine_meta, dict):
			origin_step = _safe_int(refine_meta.get("origin_step"))
			origin_round = _safe_int(refine_meta.get("origin_round"))
			origin_error_line = _safe_int(refine_meta.get("error_line"))
			origin_pde_type = refine_meta.get("pde_type")
			if origin_step is not None and origin_round is not None:
				# 新目录结构：execute/step-{origin_step}/{origin_pde_type}/{origin_round}
				# 兼容旧结构：execute/step-{origin_step}/{origin_round}
				origin_exec_dir_new = os.path.abspath(
					os.path.join(
						"..",
						project_name,
						"execute",
						f"step-{origin_step}",
						str(origin_pde_type or pde_type),
						str(origin_round),
					)
				)
				origin_exec_dir_old = os.path.abspath(
					os.path.join("..", project_name, "execute", f"step-{origin_step}", str(origin_round))
				)
				origin_exec_dir = origin_exec_dir_new if os.path.exists(origin_exec_dir_new) else origin_exec_dir_old

		# 预先计算 masked_response（同一个 group 内各候选共享）
		# token-level masked_response 优先从 rollout 输出读取；若缺失再尝试从 origin_exec_dir 近似重建
		masked_response_text = None
		masking_meta = None
		if suffix == "-refine" and isinstance(item, dict):
			try:
				mr = item.get("masked_response")
				if isinstance(mr, str) and mr.strip():
					masked_response_text = mr
			except Exception:
				pass
		if suffix == "-refine" and origin_exec_dir:
			try:
				origin_resp_path = os.path.join(origin_exec_dir, "responses.txt")
				origin_solver_path = os.path.join(origin_exec_dir, "solver.py")
				origin_err_path = os.path.join(origin_exec_dir, "errors.txt")
				if masked_response_text is None and os.path.exists(origin_resp_path) and os.path.exists(origin_solver_path):
					with open(origin_resp_path, "r", encoding="utf-8", errors="surrogatepass") as f:
						origin_full_output = f.read()

					error_line = origin_error_line
					if error_line is None and os.path.exists(origin_err_path):
						import re
						with open(origin_err_path, "r", encoding="utf-8", errors="surrogatepass") as f:
							err_text = f.read()
						m_line = re.search(r"solver\.py\", line (\d+)", err_text)
						if m_line:
							error_line = _safe_int(m_line.group(1))

					if error_line is not None and error_line > 0:
						with open(origin_solver_path, "r", encoding="utf-8", errors="surrogatepass") as f:
							solver_lines = f.readlines()
						L = len(solver_lines)
						line = max(1, min(int(error_line), L))
						start_line = max(1, line - max(0, int(masking_k_up)))
						end_line = min(L, line + max(0, int(masking_k_down)))
						block_lines = solver_lines[start_line - 1 : end_line]
						masked_response_text, approx_meta = _mask_response_with_solver_block(
							origin_full_output,
							block_lines,
							mask_placeholder_line,
						)
						if approx_meta:
							masking_meta = {
								"origin_step": origin_step,
								"origin_round": origin_round,
								"error_line": error_line,
								"solver_block_lines": [int(start_line), int(end_line)],
								**approx_meta,
							}
			except Exception:
				masked_response_text = None
				masking_meta = None

		for cand_idx in range(m):
			sub_dir = os.path.join(exec_root, str(counter))
			os.makedirs(sub_dir, exist_ok=True)
			counter_map[counter] = (task_idx, cand_idx)

			# responses/prompt
			with open(os.path.join(sub_dir, "responses.txt"), "w", encoding="utf-8", errors="surrogatepass") as f:
				f.write(str(full_outputs[cand_idx]))
			with open(os.path.join(sub_dir, "prompt.txt"), "w", encoding="utf-8", errors="surrogatepass") as f:
				f.write(str(item.get("prompt", "")))

			# ===== refine 监控文件 =====
			if suffix == "-refine":
				# 1) 原始(scratch) execute 文件：response/solver/errors
				if origin_exec_dir and os.path.isdir(origin_exec_dir):
					for src_name, dst_name in [
						("responses.txt", "origin_responses.txt"),
						("solver.py", "origin_solver.py"),
						("errors.txt", "origin_errors.txt"),
						("prompt.txt", "origin_prompt.txt"),
					]:
						src = os.path.join(origin_exec_dir, src_name)
						dst = os.path.join(sub_dir, dst_name)
						if os.path.exists(src):
							try:
								shutil.copyfile(src, dst)
							except Exception:
								pass

				# 2) 输出 JSON 自带的 refine_meta（溯源信息）
				if isinstance(refine_meta, dict):
					try:
						with open(
							os.path.join(sub_dir, "refine_meta.json"),
							"w",
							encoding="utf-8",
							errors="surrogatepass",
						) as f_meta:
							json.dump(refine_meta, f_meta, indent=2, ensure_ascii=False)
					except Exception:
						pass

				# 3) masked response（尽量重建；失败则不写）
				if isinstance(masked_response_text, str) and masked_response_text.strip():
					try:
						with open(
							os.path.join(sub_dir, "masked_response.txt"),
							"w",
							encoding="utf-8",
							errors="surrogatepass",
						) as f_mask:
							f_mask.write(masked_response_text)
					except Exception:
						pass
				if isinstance(masking_meta, dict):
					try:
						with open(
							os.path.join(sub_dir, "masking_meta.json"),
							"w",
							encoding="utf-8",
							errors="surrogatepass",
						) as f_m:
							json.dump(masking_meta, f_m, indent=2, ensure_ascii=False)
					except Exception:
						pass

			code_text = str(extracted_outputs[cand_idx])
			placeholder_msg = "We can not extract the code in the output."
			if code_text.strip() != placeholder_msg:
				with open(os.path.join(sub_dir, "solver.py"), "w", encoding="utf-8", errors="surrogatepass") as f:
					f.write(code_text)

			evaluator_src = get_evaluator_path(pde_type, cfg)
			evaluator_dst = os.path.join(sub_dir, "evaluator.py")
			shutil.copyfile(evaluator_src, evaluator_dst)

			open(os.path.join(sub_dir, "output.txt"), "w", encoding="utf-8", errors="surrogatepass").close()
			open(os.path.join(sub_dir, "errors.txt"), "w", encoding="utf-8", errors="surrogatepass").close()

			counter += 1

	# 评测数据集路径
	eval_dataset = os.path.join(str(cfg.dataset.root_dataset_folder), str(cfg.pde.dataset_for_eval))
	##############################
	##############################


	##############################
	##############################
	# 并行评测
	gpu_list = [g.strip() for g in str(getattr(cfg, "assigned_gpu", "0")).split(",") if g.strip()]
	if not gpu_list:
		gpu_list = ["0"]
	num_gpus = len(gpu_list)
	total_workers = max(1, num_gpus * int(getattr(cfg.execute, "max_workers_per_gpu", 1) or 1))

	def run_evaluator_in_subdir(sub_dir: str, gpu_id: str) -> Dict[str, Any]:
		job_out = open(os.path.join(sub_dir, "output.txt"), "w")
		job_err = open(os.path.join(sub_dir, "errors.txt"), "w")

		cmd = [
			evaluator_python,
			"evaluator.py",
			"--save-pth",
			".",
			"--run-id",
			"0",
			"--dataset-path-for-eval",
			str(eval_dataset),
		]

		pde_name = pde_type
		if pde_name == "advection":
			cmd.extend(["--beta", str(cfg.pde.beta)])
		elif pde_name == "burgers":
			cmd.extend(["--nu", str(cfg.pde.nu)])
		elif pde_name == "reacdiff1d":
			cmd.extend(["--nu", str(cfg.pde.nu), "--rho", str(cfg.pde.rho)])
		elif pde_name == "cns1d":
			cmd.extend(["--eta", str(cfg.pde.eta)])
		elif pde_name in ["darcy", "ins2d"]:
			pass
		else:
			raise ValueError(f"PDE {pde_name} not recognized")

		env = os.environ.copy()
		env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

		process = None
		try:
			process = subprocess.Popen(
				cmd,
				stdout=job_out,
				stderr=job_err,
				text=True,
				cwd=sub_dir,
				preexec_fn=os.setsid,
				env=env,
			)
			exit_code = process.wait(timeout=float(getattr(cfg.pde, "timeout", 300)))
			status = "completed"
			stderr = None
		except subprocess.TimeoutExpired:
			status = "timeout"
			stderr = "TimeoutExpired"
			exit_code = -1
			if process is not None:
				try:
					os.killpg(os.getpgid(process.pid), signal.SIGTERM)
				except Exception:
					pass
				time.sleep(1)
				try:
					os.killpg(os.getpgid(process.pid), signal.SIGKILL)
				except Exception:
					pass
		finally:
			job_out.close()
			job_err.close()

		return {"exit_code": exit_code, "stderr": stderr, "status": status}

	def eval_one(counter_idx: int, gpu_id: str):
		sub_dir = os.path.join(exec_root, str(counter_idx))
		solver_path = os.path.join(sub_dir, "solver.py")
		task_idx, cand_idx = counter_map[counter_idx]

		if not os.path.exists(solver_path):
			return counter_idx, task_idx, cand_idx, None, None, None, "no_solver", "no_solver", ""

		result = run_evaluator_in_subdir(sub_dir, gpu_id)
		errors_path = os.path.join(sub_dir, "errors.txt")

		if result["exit_code"] != 0:
			err_type, err_line = _extract_error_info(errors_path)
			if result.get("status") == "timeout" and err_type == "none":
				err_type = "timeout"
			return counter_idx, task_idx, cand_idx, None, None, None, "failed", err_type, err_line

		last_line = _get_last_nonempty_line(os.path.join(sub_dir, "output.txt"))
		if not last_line:
			err_type, err_line = _extract_error_info(errors_path)
			if err_type == "none":
				err_type = "no_output"
			return counter_idx, task_idx, cand_idx, None, None, None, "failed", err_type, err_line

		try:
			nrmse, elapsed, rate = _parse_results_from_line(last_line)
			return counter_idx, task_idx, cand_idx, nrmse, elapsed, rate, "ok", "none", ""
		except Exception:
			err_type, err_line = _extract_error_info(errors_path)
			if err_type == "none":
				err_type = "parse_failed"
			return counter_idx, task_idx, cand_idx, None, None, None, "failed", err_type, err_line

	tasks = []
	for i in range(counter):
		gpu_id = gpu_list[i % num_gpus]
		tasks.append((i, gpu_id))

	results = []
	# 并行核心逻辑
	with ThreadPoolExecutor(max_workers=total_workers) as executor:
		future_map = {executor.submit(eval_one, idx, gpu_id): idx for (idx, gpu_id) in tasks}
		for fut in as_completed(future_map):
			try:
				results.append(fut.result())
			except Exception:
				idx = future_map[fut]
				task_idx, cand_idx = counter_map[idx]
				results.append((idx, task_idx, cand_idx, None, None, None, "failed", "exception", ""))

	results.sort(key=lambda x: x[0])

	##############################
	##############################


	##############################
	##############################
	# 评测结果记录 --> csv + 回填JSON

	# 回填 data + 写 CSV
	for round_idx, task_idx, cand_idx, nrmse, elapsed, rate, status, err_type, err_line in results:
		item = data[task_idx]
		metrics = {
			"round": round_idx,
			"status": status,
			"nRMSE": nrmse,
			"elapsed_time": elapsed,
			"convergence_rate": rate,
			"error_type": err_type,
			"error_line": err_line,
		}
		# pde_metrics 可能已初始化成 list[None]*m
		if isinstance(item.get("pde_metrics"), list) and cand_idx < len(item["pde_metrics"]):
			item["pde_metrics"][cand_idx] = metrics
		else:
			item.setdefault("pde_metrics", []).append(metrics)

		ok = bool(status == "ok" and nrmse is not None)
		if isinstance(item.get("correctness"), list) and cand_idx < len(item["correctness"]):
			item["correctness"][cand_idx] = ok
		else:
			item.setdefault("correctness", []).append(ok)

		row = (
			f"{pde_type},{round_idx},{task_idx},{cand_idx},{nrmse},{elapsed},{rate},{status},{err_type},{err_line}\n"
		)
		with open(result_sheet_path, "a", encoding="utf-8", errors="surrogatepass") as csv_f:
			csv_f.write(row)
		with open(combined_sheet_path, "a", encoding="utf-8", errors="surrogatepass") as csv_all:
			csv_all.write(row)

	# 保存更新后的 outputs JSON
	Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
	with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)

	cprint(f"[pde_rl_execute] updated outputs: {file_name}", "green")
	##############################
	##############################

if __name__ == "__main__":
	main()
