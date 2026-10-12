from __future__ import annotations

"""Skeleton script for PDE code refinement with Dream model.

当前文件是基于 pde_sample_dream.py 抽取的公共逻辑 + 二次修复流程的大纲，
用于后续实现：
1) 根据第一次执行生成的错误信息（CSV + errors.txt）定位出错代码块；
2) 在原始 LLM response 中用 [MASK] 覆盖该代码块；
3) 调用 Dream 模型对 [MASK] 段进行局部再采样，生成新的 solver 代码。

该脚本需要在 sample 目录下运行

example:
	python pde_refine_dream.py config=../configs/eval.yaml

"""

import os
import sys
import json
import re
import shutil
import subprocess
import signal
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

# 确保可以导入与 sample 同级的模块（例如 pde.*）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import torch
from termcolor import cprint
from omegaconf import OmegaConf

if __package__:
	from sample.dream import DreamTokenizer
	from sample.dream.modeling_dream import DreamModel
	from sample.dream.generation_utils_block import DreamGenerationMixin, DreamGenerationConfig
	from sample.pde_sample_dream import _sample as dream_block_sample
else:
	from dream import DreamTokenizer
	from dream.modeling_dream import DreamModel
	from dream.generation_utils_block import DreamGenerationMixin, DreamGenerationConfig
	from pde_sample_dream import _sample as dream_block_sample
from pde.prompt_files.refine_prompt import refine_prompt
from pde.paths import get_evaluator_python


USE_JINJA_RENDER = True

# 简单的全局缓存，避免在单次 refine 流程中重复加载 Dream 模型
_CACHED_DREAM_MODEL: Optional[DreamModel] = None
_CACHED_MODEL_PATH: Optional[str] = None


def _split_even(lst: list, n: int) -> list:
	if n <= 0:
		return [lst]
	k, m = divmod(len(lst), n)
	return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _refine_worker_process(
	config_container: Dict[str, Any],
	local_rank: int,
	failed_items: List[Dict[str, Any]],
	refined_rounds_out,
	gpu_time_dict,
) -> None:
	"""Refine a subset of failed rounds on a specific GPU (local rank within visible GPUs)."""
	try:
		config = OmegaConf.create(config_container)
		torch.cuda.set_device(local_rank)
	except Exception as e:
		cprint(f"[refine_worker] init failed on rank={local_rank}: {e}", "red")
		gpu_time_dict[int(local_rank)] = 0.0
		return

	project_name = config.experiment.project
	pretrained_model = config.model

	try:
		tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
		model = get_dream_model(pretrained_model)
		mask_token_id = getattr(model.config, "mask_token_id", None)
		if mask_token_id is None:
			raise ValueError("model.config.mask_token_id is required")
	except Exception as e:
		cprint(f"[refine_worker] load model/tokenizer failed on rank={local_rank}: {e}", "red")
		gpu_time_dict[int(local_rank)] = 0.0
		return

	mask_block_tokens = config.rollout.refine_mask_block_tokens
	exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))

	t0 = time.time()
	refined_local: List[int] = []
	for item in failed_items:
		round_idx = int(item["round_idx"])
		error_line_str = item.get("error_line", "")
		try:
			error_line = int(error_line_str)
		except (TypeError, ValueError):
			continue

		sub_dir = os.path.join(exec_root, str(round_idx))
		solver_path = os.path.join(sub_dir, "solver.py")
		responses_path = os.path.join(sub_dir, "responses.txt")
		if not os.path.exists(solver_path) or not os.path.exists(responses_path):
			continue

		# 1) locate error block in solver
		start_line, end_line = locate_error_block_in_solver(solver_path, error_line)
		with open(solver_path, "r", encoding="utf-8", errors="surrogatepass") as f_sol:
			solver_lines = f_sol.readlines()
		if start_line < 1 or end_line < start_line or end_line > len(solver_lines):
			continue
		block_lines = solver_lines[start_line - 1 : end_line]

		# 2) read full_output
		with open(responses_path, "r", encoding="utf-8", errors="surrogatepass") as f_resp:
			full_output = f_resp.read()

		prompt_for_refine = refine_prompt.strip()

		# 3) map solver block back to response span
		start_char, end_char = map_block_back_to_response(full_output, block_lines)
		if end_char <= start_char:
			continue

		# 4) map char span to token span and build masked ids
		tok_start, tok_end = map_response_span_to_token_span(
			tokenizer,
			full_output,
			(start_char, end_char),
		)
		if tok_end <= tok_start:
			continue

		masked_ids = build_masked_response(
			tokenizer,
			full_output,
			(tok_start, tok_end),
			mask_block_tokens,
			mask_token_id,
		)
		masked_text = tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

		# 5) local resample with Dream
		try:
			refined_resp = refine_with_dream_model(
				config=config,
				tokenizer=tokenizer,
				prompt=prompt_for_refine,
				masked_response_ids=masked_ids,
				pretrained_model=pretrained_model,
			)
		except Exception:
			continue

		# 6) write back
		try:
			backup_path = os.path.join(sub_dir, "ori_responses.txt")
			try:
				shutil.copy2(responses_path, backup_path)
			except Exception:
				pass

			masked_resp_path = os.path.join(sub_dir, "masked_responses.txt")
			try:
				with open(masked_resp_path, "w", encoding="utf-8", errors="surrogatepass") as f_masked:
					f_masked.write(masked_text)
			except Exception:
				pass

			with open(responses_path, "w", encoding="utf-8", errors="surrogatepass") as f_resp:
				f_resp.write(refined_resp)

			code_output = extract_long_code(refined_resp)
			with open(solver_path, "w", encoding="utf-8", errors="surrogatepass") as f_sol:
				f_sol.write(code_output)

			prompt_path = os.path.join(sub_dir, "prompt.txt")
			try:
				with open(prompt_path, "w", encoding="utf-8", errors="surrogatepass") as f_prompt:
					f_prompt.write(prompt_for_refine)
			except Exception:
				pass

			refined_local.append(round_idx)
		except Exception:
			continue

	gpu_time_dict[int(local_rank)] = float(time.time() - t0)
	for r in refined_local:
		refined_rounds_out.append(int(r))


#####################################################################
#####################################################################
# Refine 流程相关的函数

def archive_current_execute(project_name: str, step_idx: int) -> None:
	"""将当前项目下的 execute 目录归档到 history/execute_{step_idx}。

	具体行为:
	- 在 .. / {project_name} 目录下创建 history 子目录 (若不存在);
	- 使用 shutil.copytree 将当前的 execute 整个复制为
	  history/execute_{step_idx}, 其中 step_idx=0 通常表示原始 (未 refine) 结果;
	- 不修改当前 execute 目录本身, 仅做快照备份。

	安全策略:
	- 若源 execute 目录不存在, 记录 warning 并直接返回;
	- 若目标 history/execute_{step_idx} 已存在, 为避免误覆盖, 记录 warning 并跳过复制。
	"""
	exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))
	if not os.path.exists(exec_root):
		cprint(
			f"[archive_current_execute] execute dir not found at {exec_root}, skip archiving.",
			"red",
		)
		return

	history_root = os.path.abspath(os.path.join("..", project_name, "history"))
	os.makedirs(history_root, exist_ok=True)

	dst_dir = os.path.join(history_root, f"execute_{step_idx}")
	if os.path.exists(dst_dir):
		cprint(
			f"[archive_current_execute] target history dir already exists: {dst_dir}, skip to avoid overwrite.",
			"yellow",
		)
		return

	try:
		shutil.copytree(exec_root, dst_dir)
		cprint(
			f" archived execute directory to {dst_dir}",
			"cyan",
		)
	except Exception as e:
		cprint(
			f" failed to archive execute directory: {e}",
			"red",
		)


def refine_once_inplace(config, tokenizer: DreamTokenizer, step_idx: int) -> List[int]:
	"""对当前 execute 目录执行单轮掩码+重解码, 并就地覆盖相关文件。

	具体流程:
	1) 读取 .. / {project_name} / execute/test_results.csv, 找到失败的 round;
	2) 对每个失败 round:
	   - 在 solver.py 中基于 error_line 定位错误代码块行号范围;
	   - 将该代码块映射回 responses.txt 中的 full_output 文本;
	   - 将对应 token 区间替换为固定长度的 MASK 段;
	   - 使用 refine_with_dream_model 进行局部再采样;
	   - 用生成结果覆盖当前 round 的 responses.txt, 并用提取出的代码块覆盖 solver.py;
	3) 仅修改能够成功完成上述流程的 round, 失败样本会跳过并打印 warning。
	4) 返回报错 round_idx, 用于定位后续需要重新评测的样本, 以及需要更新的 csv 行数.

	step_idx 目前仅用于日志标记, 便于调试不同 refine 轮次的行为。
	"""
	project_name = config.experiment.project
	pretrained_model = config.model

	refined_rounds: List[int] = []
	exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))
	if not os.path.exists(exec_root):
		cprint(
			f"[refine] execute dir not found at {exec_root}, abort refine.",
			"red",
		)
		return refined_rounds

	# 加载 execute 结果, 筛选失败样本
	all_exec_results = load_execute_results(project_name)
	failed_items = [
		r
		for r in all_exec_results
		if r.get("error_type") not in ("", "none", "no_solver")
	]
	if not failed_items:
		cprint(
			f" no failed rounds found in execute results, nothing to refine.",
			"cyan",
		)
		return refined_rounds

	# 从第一次采样的 outputs JSON 中加载 round_idx -> prompt/full_output 的映射
	first_round_info = load_first_round_outputs(config)
	round_map: Dict[int, Dict[str, Any]] = first_round_info["round_map"]

	# 从配置中读取掩码块长度, 并取得模型的 mask_token_id
	mask_block_tokens = config.rollout.refine_mask_block_tokens
	model = get_dream_model(pretrained_model)
	mask_token_id = getattr(model.config, "mask_token_id", None)
	if mask_token_id is None:
		raise ValueError("model.config.mask_token_id is required for refine_once_inplace")


	for item in failed_items:
		round_idx = item["round_idx"]
		error_line_str = item.get("error_line", "")
		try:
			error_line = int(error_line_str)
		except (TypeError, ValueError):
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, invalid error_line='{error_line_str}'",
				"red",
			)
			continue

		sub_dir = os.path.join(exec_root, str(round_idx))
		solver_path = os.path.join(sub_dir, "solver.py")
		responses_path = os.path.join(sub_dir, "responses.txt")

		if not os.path.exists(solver_path) or not os.path.exists(responses_path):
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, missing solver.py or responses.txt",
				"red",
			)
			continue

		# 1) 在 solver.py 中定位错误代码块
		start_line, end_line = locate_error_block_in_solver(solver_path, error_line)
		cprint(
			f"  [refine] round_idx={round_idx}, error_type={item['error_type']}, "
			f"error_line={error_line}, block_lines=[{start_line}, {end_line}]",
			"yellow",
		)
		with open(solver_path, "r", encoding="utf-8", errors="surrogatepass") as f_sol:
			solver_lines = f_sol.readlines()
		if start_line < 1 or end_line < start_line or end_line > len(solver_lines):
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, invalid block range [{start_line}, {end_line}]",
				"red",
			)
			continue
		block_lines = solver_lines[start_line - 1 : end_line]

		# 2) 从 responses.txt 中读取 full_output
		with open(responses_path, "r", encoding="utf-8", errors="surrogatepass") as f_resp:
			full_output = f_resp.read()

		meta = round_map.get(round_idx)
		if meta is None:
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, not found in first-round outputs mapping",
				"red",
			)
			continue
		prompt_for_refine = refine_prompt.strip()

		# 3) 将 solver 代码块映射回 full_output 的字符区间
		start_char, end_char = map_block_back_to_response(full_output, block_lines)
		if end_char <= start_char:
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, failed to map block back to response.",
				"red",
			)
			continue

		# 4) 将字符区间映射到 token 区间, 构造 masked ids
		tok_start, tok_end = map_response_span_to_token_span(
			tokenizer,
			full_output,
			(start_char, end_char),
		)
		if tok_end <= tok_start:
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, empty token span [{tok_start}, {tok_end}).",
				"red",
			)
			continue

		masked_ids = build_masked_response(
			tokenizer,
			full_output,
			(tok_start, tok_end),
			mask_block_tokens,
			mask_token_id,
		)
		masked_text = tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

		# 5) 调用 Dream 模型对 MASK 区域进行局部再采样
		try:
			refined_resp = refine_with_dream_model(
				config=config,
				tokenizer=tokenizer,
				prompt=prompt_for_refine,
				masked_response_ids=masked_ids,
				pretrained_model=pretrained_model,
			)
		except Exception as e:
			cprint(
				f"  [refine] [skip] round_idx={round_idx}, model refine failed: {e}",
				"red",
			)
			continue

		# 6) 将 refined_resp 写回 responses.txt, 并从中提取代码块覆盖 solver.py
		try:
			# 在覆盖写入前, 先备份当前 responses.txt 为 ori_responses.txt
			backup_path = os.path.join(sub_dir, "ori_responses.txt")
			try:
				shutil.copy2(responses_path, backup_path)
			except Exception as e:
				cprint(
					f"  [refine] [warn] round_idx={round_idx}, failed to backup responses.txt: {e}",
					"red",
				)

			masked_resp_path = os.path.join(sub_dir, "masked_responses.txt")
			try:
				with open(
					masked_resp_path,
					"w",
					encoding="utf-8",
					errors="surrogatepass",
				) as f_masked:
					f_masked.write(masked_text)
			except Exception as e:
				cprint(
					f"  [refine] [warn] round_idx={round_idx}, failed to write masked_responses.txt: {e}",
					"red",
				)

			with open(responses_path, "w", encoding="utf-8", errors="surrogatepass") as f_resp:
				f_resp.write(refined_resp)

			code_output = extract_long_code(refined_resp)
			with open(solver_path, "w", encoding="utf-8", errors="surrogatepass") as f_sol:
				f_sol.write(code_output)

			# 更新本轮使用的 prompt, 以记录 refine 阶段的指导提示词
			prompt_path = os.path.join(sub_dir, "prompt.txt")
			try:
				with open(prompt_path, "w", encoding="utf-8", errors="surrogatepass") as f_prompt:
					f_prompt.write(prompt_for_refine)
			except Exception as e:
				cprint(
					f"  [refine] [warn] round_idx={round_idx}, failed to update prompt.txt: {e}",
					"red",
				)

			refined_rounds.append(round_idx)
			cprint(
				f"  [refine] round_idx={round_idx} refined and files updated in-place.",
				"green",
			)
		except Exception as e:
			cprint(
				f"  [refine] [warn] round_idx={round_idx}, failed to write updated files: {e}",
				"red",
			)

	return refined_rounds


def re_evaluate_execute_inplace(
	config,
	target_rounds: Optional[List[int]] = None,
) -> None:
	"""根据 target_rounds(List) 选择性地重新评测 execute 子目录, 并仅更新对应 CSV 行。"""
	project_name = config.experiment.project
	dataset_cfg = getattr(config, "dataset", None)
	if dataset_cfg is None or getattr(dataset_cfg, "data_type", "") != "pde":
		cprint(
			"[re_evaluate_execute_inplace] dataset type is not PDE, skip re-evaluation.",
			"yellow",
		)
		return

	pde_cfg = getattr(config, "pde", None)
	if pde_cfg is None:
		raise ValueError(
			"[re_evaluate_execute_inplace] config.pde is missing; cannot prepare PDE evaluator args."
		)

	exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))
	if not os.path.exists(exec_root):
		cprint(
			f"[re_evaluate_execute_inplace] execute directory not found: {exec_root}; skip.",
			"red",
		)
		return

	result_sheet_path = os.path.join(exec_root, "test_results.csv")
	header_line = "round,nRMSE,elapsed_time,convergence_rate,num_trial,error_type,error_line\n"
	row_map: Dict[int, str] = {}
	if os.path.exists(result_sheet_path):
		with open(result_sheet_path, "r", encoding="utf-8", errors="surrogatepass") as f:
			lines = f.readlines()
		if lines:
			header_candidate = lines[0].rstrip("\n")
			if header_candidate:
				header_line = header_candidate + "\n"
			for line in lines[1:]:
				stripped = line.strip()
				if not stripped:
					continue
				parts = stripped.split(",")
				try:
					round_idx = int(parts[0])
				except ValueError:
					continue
				row_map[round_idx] = stripped
	else:
		os.makedirs(os.path.dirname(result_sheet_path), exist_ok=True)

	# 评测数据集路径
	root_dataset = getattr(dataset_cfg, "root_dataset_folder", "")
	dataset_for_eval = getattr(pde_cfg, "dataset_for_eval", None)
	if not dataset_for_eval:
		raise ValueError(
			"[re_evaluate_execute_inplace] pde.dataset_for_eval is missing; cannot locate eval dataset."
		)
	eval_dataset = os.path.join(str(root_dataset), str(dataset_for_eval))
	evaluator_python = get_evaluator_python(config)

	sub_dirs = [
		os.path.join(exec_root, d)
		for d in os.listdir(exec_root)
		if os.path.isdir(os.path.join(exec_root, d)) and d.isdigit()
	]
	sub_dirs.sort(key=lambda p: int(os.path.basename(p)))

	target_set: Optional[set[int]] = None
	if target_rounds is not None:
		valid_rounds: List[int] = []
		for r in target_rounds:
			try:
				valid_rounds.append(int(r))
			except (TypeError, ValueError):
				continue
		if not valid_rounds:
			cprint(
				"[re_evaluate_execute_inplace] no valid target rounds provided; skip re-evaluation.",
				"yellow",
			)
			return
		target_set = set(valid_rounds)
		sub_dirs = [
			sub_dir
			for sub_dir in sub_dirs
			if int(os.path.basename(sub_dir)) in target_set
		]
		if not sub_dirs:
			cprint(
				"[re_evaluate_execute_inplace] none of the target rounds have execute directories; skip.",
				"yellow",
			)
			return
		missing_rounds = sorted(target_set - {int(os.path.basename(p)) for p in sub_dirs})
		if missing_rounds:
			cprint(
				f"[re_evaluate_execute_inplace] missing execute dirs for rounds: {missing_rounds}",
				"yellow",
			)
	elif not sub_dirs:
		cprint(
			"[re_evaluate_execute_inplace] no round sub-directories found; nothing to re-evaluate.",
			"yellow",
		)
		return

	def cleanup_gpu_processes(gpu_id: str) -> None:
		"""在超时/异常后尝试回收仍占用 GPU 的进程。"""
		try:
			result = subprocess.run(
				[
					"nvidia-smi",
					"--query-compute-apps=pid",
					"--format=csv,noheader,nounits",
					"-i",
					str(gpu_id),
				],
				capture_output=True,
				text=True,
			)
			pids = [pid.strip() for pid in result.stdout.strip().split("\n") if pid.strip()]
			for pid in pids:
				if pid.isdigit():
					try:
						os.kill(int(pid), signal.SIGKILL)
					except ProcessLookupError:
						continue
		except Exception as e:
			cprint(
				f"[re_evaluate_execute_inplace] failed to cleanup GPU processes: {e}",
				"yellow",
			)

	def get_last_line(output_file: str) -> Optional[str]:
		if not os.path.exists(output_file):
			return None
		with open(output_file, "r", encoding="utf-8", errors="surrogatepass") as f:
			lines = f.readlines()
		for line in reversed(lines):
			if line.strip():
				return line
		return None

	def parse_results_from_line(line: str) -> Tuple[float, float, float]:
		import re
		rel_err = float(re.search(r"nRMSE: (.*?)\t", line).group(1))
		elapsed = float(re.search(r"Time: (.*?)s", line).group(1))
		avg_rate = float(re.search(r"Average convergence rate: (.*?)\t", line).group(1))
		return rel_err, elapsed, avg_rate

	def extract_error_info(errors_path: str) -> Tuple[str, str]:
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

	def run_evaluator_in_subdir(sub_dir: str, gpu_id: str) -> Dict[str, Any]:
		"""复用 pde_execute.py 的 evaluator 调用方式。"""
		job_out = open(os.path.join(sub_dir, "output.txt"), "w", encoding="utf-8", errors="surrogatepass")
		job_err = open(os.path.join(sub_dir, "errors.txt"), "w", encoding="utf-8", errors="surrogatepass")

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

		pde_name = getattr(dataset_cfg, "pde_type", None)
		if not pde_name:
			raise ValueError("dataset.pde_type is missing; cannot determine evaluator hyperparams.")

		if pde_name == "advection":
			cmd.extend(["--beta", str(getattr(pde_cfg, "beta", 0.0))])
		elif pde_name == "burgers":
			cmd.extend(["--nu", str(getattr(pde_cfg, "nu", 0.0))])
		elif pde_name == "reacdiff1d":
			cmd.extend(
				[
					"--nu",
					str(getattr(pde_cfg, "nu", 0.0)),
					"--rho",
					str(getattr(pde_cfg, "rho", 0.0)),
				]
			)
		elif pde_name == "cns1d":
			cmd.extend(["--eta", str(getattr(pde_cfg, "eta", 0.0))])
		elif pde_name in ["darcy", "ins2d"]:
			pass
		else:
			raise ValueError(f"Unknown PDE type: {pde_name}")

		env = os.environ.copy()
		env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
		timeout = getattr(pde_cfg, "timeout", 3600)

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
			exit_code = process.wait(timeout=timeout)
			stderr = None
			status = "completed"
		except subprocess.TimeoutExpired:
			if process is not None:
				os.killpg(os.getpgid(process.pid), signal.SIGTERM)
				time.sleep(2)
				if process.poll() is None:
					os.killpg(os.getpgid(process.pid), signal.SIGKILL)
			cleanup_gpu_processes(gpu_id)
			job_out.write(f"Process exceeded the {timeout}-second timeout limit.\n")
			job_err.write(f"Process exceeded the {timeout}-second timeout limit.\n")
			exit_code = -1
			stderr = "timeout"
			status = "timeout"
		except Exception as e:
			stderr = str(e)
			exit_code = -1
			status = "exception"
		finally:
			job_out.close()
			job_err.close()

		return {"exit_code": exit_code, "stderr": stderr, "status": status}

	def eval_one_subdir(sub_dir: str, gpu_id: str) -> Tuple[int, Any, Any, Any, str, str]:
		round_idx = int(os.path.basename(sub_dir))
		solver_path = os.path.join(sub_dir, "solver.py")
		if not os.path.exists(solver_path):
			return round_idx, "none", "none", "none", "no_solver", ""

		result = run_evaluator_in_subdir(sub_dir, gpu_id)
		errors_path = os.path.join(sub_dir, "errors.txt")
		if result["exit_code"] != 0:
			error_type, error_line = extract_error_info(errors_path)
			if result["status"] == "timeout" and error_type == "none":
				error_type = "timeout"
			return round_idx, "failed", "failed", "failed", error_type, error_line

		output_file = os.path.join(sub_dir, "output.txt")
		last_line = get_last_line(output_file)
		if not last_line:
			error_type, error_line = extract_error_info(errors_path)
			return round_idx, "failed", "failed", "failed", error_type, error_line

		try:
			rel_err, elapsed, avg_rate = parse_results_from_line(last_line)
		except Exception:
			error_type, error_line = extract_error_info(errors_path)
			if error_type == "none":
				error_type = "parse_failed"
			return round_idx, "failed", "failed", "failed", error_type, error_line

		return round_idx, rel_err, elapsed, avg_rate, "none", ""

	gpu_list = [g.strip() for g in str(config.assigned_gpu).split(",") if g.strip()]
	if not gpu_list:
		gpu_list = ["0"]
	max_workers_per_gpu = int(getattr(config, "max_workers_per_gpu", 1))
	total_workers = max(1, len(gpu_list) * max_workers_per_gpu)

	tasks = []
	for i, sub_dir in enumerate(sub_dirs):
		gpu_id = gpu_list[i % len(gpu_list)]
		tasks.append((sub_dir, gpu_id))

	results = []
	from concurrent.futures import ThreadPoolExecutor, as_completed
	with ThreadPoolExecutor(max_workers=total_workers) as executor:
		future_to_round = {
			executor.submit(eval_one_subdir, sub_dir, gpu_id): int(os.path.basename(sub_dir))
			for sub_dir, gpu_id in tasks
		}
		for fut in as_completed(future_to_round):
			try:
				results.append(fut.result())
			except Exception:
				round_idx = future_to_round[fut]
				results.append((round_idx, "failed", "failed", "failed", "exception", ""))

	results.sort(key=lambda x: x[0])
	for round_idx, rel_err, elapsed, avg_rate, err_type, err_line in results:
		row_map[round_idx] = (
			f"{round_idx},{rel_err},{elapsed},{avg_rate},1,{err_type},{err_line}"
		)

	ordered_rounds = sorted(row_map.keys())
	with open(result_sheet_path, "w", encoding="utf-8", errors="surrogatepass") as csv_f:
		if not header_line.endswith("\n"):
			header_line_to_write = header_line + "\n"
		else:
			header_line_to_write = header_line
		csv_f.write(header_line_to_write)
		for ridx in ordered_rounds:
			line = row_map[ridx]
			if not line.endswith("\n"):
				line = line + "\n"
			csv_f.write(line)

	cprint(
		f" re-evaluated {len(results)} rounds; partial results synced to {result_sheet_path}",
		"cyan",
	)


def run_multi_step_refine_loop(config) -> Dict[str, Any]:
	"""按 rollout 配置执行多轮 "归档 -> refine -> 重新评测" 流程，并返回简单统计信息。"""
	rollout_cfg = getattr(config, "rollout", None)
	project_name = getattr(getattr(config, "experiment", None), "project", None)

	def _coerce_float(val):
		if val is None:
			return None
		try:
			return float(val)
		except Exception:
			return None

	good_nrmse_threshold = _coerce_float(
		OmegaConf.select(config, "execute.good_nrmse_threshold", default=None)
	)
	

	def compute_execute_pass_metrics(proj_name: str) -> Dict[str, Any]:
		"""读取 execute/test_results.csv，计算跑通率与 good@nRMSE。

		- pass: error_type 为空或 none
		- good: pass 且 nRMSE < threshold（threshold 未配置则不统计 good）
		"""
		import csv
		exec_root = os.path.abspath(os.path.join("..", proj_name, "execute"))
		csv_path = os.path.join(exec_root, "test_results.csv")
		metrics: Dict[str, Any] = {
			"total": 0,
			"passed": 0,
			"failed": 0,
			"pass_rate": 0.0,
			"good": 0,
			"good_rate": 0.0,
			"good_passed_rate": 0.0,
			"nrmse_threshold": good_nrmse_threshold,
			"nrmse_col": "nRMSE",
			"error_type_counts": {},
		}
		if not os.path.exists(csv_path):
			return metrics
		try:
			with open(csv_path, "r", encoding="utf-8", errors="surrogatepass") as f:
				reader = csv.DictReader(f)
				for row in reader:
					metrics["total"] += 1
					err_type = (row.get("error_type") or "").strip()
					key = err_type if err_type else "none"
					metrics["error_type_counts"][key] = metrics["error_type_counts"].get(key, 0) + 1
					if err_type in ("", "none"):
						metrics["passed"] += 1
						if good_nrmse_threshold is not None:
							raw = row.get("nRMSE")
							nrmse_val = _coerce_float(raw)
							if nrmse_val is not None and nrmse_val < good_nrmse_threshold:
								metrics["good"] += 1
					else:
						metrics["failed"] += 1
		except Exception:
			return metrics
		if metrics["total"] > 0:
			metrics["pass_rate"] = float(metrics["passed"]) / float(metrics["total"])
			metrics["good_rate"] = float(metrics["good"]) / float(metrics["total"])
		if metrics["passed"] > 0:
			metrics["good_passed_rate"] = float(metrics["good"]) / float(metrics["passed"])
		return metrics

	summary: Dict[str, Any] = {
		"refine_enabled": False,
		"max_steps": 0,
		"actual_iterations": 0,
		"refined_rounds_per_step": [],
		"total_refined_rounds": 0,
		"iterations": [],
		"execute_metrics_before_refine": {},
	}
	if rollout_cfg is None:
		cprint(
			"[run_multi_step_refine_loop] rollout config is missing; skip refine loop.",
			"yellow",
		)
		return summary

	enable_refine = bool(getattr(rollout_cfg, "enable_refine", False))
	summary["refine_enabled"] = enable_refine
	if not enable_refine:
		cprint(
			"[run_multi_step_refine_loop] rollout.enable_refine is False; skip refine loop.",
			"yellow",
		)
		return summary

	max_steps = int(getattr(rollout_cfg, "refine_max_steps", 0) or 0)
	summary["max_steps"] = max_steps
	if max_steps <= 0:
		cprint(
			"[run_multi_step_refine_loop] rollout.refine_max_steps <= 0; nothing to do.",
			"yellow",
		)
		return summary

	if not project_name:
		cprint(
			"[run_multi_step_refine_loop] experiment.project is missing; skip refine loop.",
			"red",
		)
		return summary

	pretrained_model = config.model
	# tokenizer/model 会在每个 GPU worker 内按需加载

	# 可见 GPU 数（在 __main__ 中通常已设置 CUDA_VISIBLE_DEVICES=assigned_gpu）
	n_visible_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
	if n_visible_gpu <= 0:
		n_visible_gpu = 1

	def has_failed_rounds() -> bool:
		try:
			results = load_execute_results(project_name)
		except FileNotFoundError:
			cprint(
				f"[run_multi_step_refine_loop] test_results.csv not found under project {project_name}; stop refine loop.",
				"red",
			)
			return False
		except Exception as err:
			cprint(
				f"[run_multi_step_refine_loop] failed to read test_results.csv: {err}",
				"red",
			)
			return False
		return any(r.get("error_type") not in ("", "none", "no_solver") for r in results)

	# 记录第一次 execute（sample 后）的跑通率，作为 refine 的 baseline
	summary["execute_metrics_before_refine"] = compute_execute_pass_metrics(project_name)

	for step_idx in range(max_steps):
		cprint(
			f" ==== refine iteration {step_idx + 1} / {max_steps} ====",
			"cyan",
		)
		archive_current_execute(project_name, step_idx)

		if not has_failed_rounds():
			cprint(
				"[run_multi_step_refine_loop] no failed rounds remaining; terminate refine loop.",
				"green",
			)
			break

		iter_metrics: Dict[str, Any] = {
			"step_idx": step_idx,
			"num_refined_rounds": 0,
			"refined_round_ids": [],
			"refine_per_gpu_times_sec": {},
			"refine_total_generation_time_sec": 0.0,
			"avg_time_per_refined_round_sec": 0.0,
			"reexecute_time_sec": 0.0,
			"execute_pass_metrics_after": {},
		}

		# --- 1) refine generation: per-iteration, parallel across GPUs ---
		all_exec_results = load_execute_results(project_name)
		failed_items = [
			r
			for r in all_exec_results
			if r.get("error_type") not in ("", "none", "no_solver")
		]

		refined_rounds: List[int] = []
		if failed_items:
			config_container = OmegaConf.to_container(config, resolve=True)
			chunks = _split_even(failed_items, n_visible_gpu)
			manager = mp.Manager()
			refined_rounds_out = manager.list()
			gpu_time_dict = manager.dict()
			procs: List[mp.Process] = []
			for rk in range(n_visible_gpu):
				p = mp.Process(
					target=_refine_worker_process,
					args=(config_container, rk, chunks[rk], refined_rounds_out, gpu_time_dict),
				)
				p.start()
				procs.append(p)
			for p in procs:
				p.join()

			refined_rounds = sorted({int(r) for r in list(refined_rounds_out)})
			per_gpu_times = {int(k): float(v) for k, v in gpu_time_dict.items()}
			# ensure every visible GPU rank has an entry
			for rk in range(n_visible_gpu):
				per_gpu_times.setdefault(int(rk), 0.0)
			iter_metrics["refined_round_ids"] = refined_rounds
			iter_metrics["num_refined_rounds"] = int(len(refined_rounds))
			iter_metrics["refine_per_gpu_times_sec"] = per_gpu_times
			iter_metrics["refine_total_generation_time_sec"] = float(sum(per_gpu_times.values()))
			if iter_metrics["num_refined_rounds"] > 0:
				iter_metrics["avg_time_per_refined_round_sec"] = (
					iter_metrics["refine_total_generation_time_sec"] / iter_metrics["num_refined_rounds"]
				)

		summary["refined_rounds_per_step"].append(int(iter_metrics["num_refined_rounds"]))
		summary["total_refined_rounds"] += int(iter_metrics["num_refined_rounds"])
		summary["actual_iterations"] = step_idx + 1

		_t1 = time.time()
		re_evaluate_execute_inplace(config, refined_rounds)
		iter_metrics["reexecute_time_sec"] = float(time.time() - _t1)
		iter_metrics["execute_pass_metrics_after"] = compute_execute_pass_metrics(project_name)
		summary["iterations"].append(iter_metrics)

		if not has_failed_rounds():
			cprint(
				"[run_multi_step_refine_loop] all rounds passed after re-evaluation; stopping loop.",
				"green",
			)
			break
	else:
		cprint(
			" reached refine_max_steps limit.",
			"cyan",
		)

	return summary


def get_dream_model(pretrained_model: str) -> DreamModel:
	"""按需加载并缓存 Dream 模型，用于局部再采样阶段。

	当前实现仅用于推理：
	- 始终 eval()；
	- 若有 GPU 则放在第一块 GPU 上，否则使用 CPU。
	"""
	global _CACHED_DREAM_MODEL, _CACHED_MODEL_PATH
	if _CACHED_DREAM_MODEL is not None and _CACHED_MODEL_PATH == pretrained_model:
		return _CACHED_DREAM_MODEL

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 与 pde_sample_dream / sft_pde_dream 一致，优先使用 bfloat16
	dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
	model = DreamModel.from_pretrained(
		pretrained_model,
		trust_remote_code=True,
		torch_dtype=dtype,
	)
	model = model.to(device).eval()

	# 仅在需要时可以挂载 diffusion_generate / _sample；当前局部 refine 不直接调用
	model.diffusion_generate = DreamGenerationMixin.diffusion_generate.__get__(model, DreamModel)
	model._sample = DreamGenerationMixin._sample.__get__(model, DreamModel)

	_CACHED_DREAM_MODEL = model
	_CACHED_MODEL_PATH = pretrained_model
	return model

#####################################################################
#####################################################################

def get_config():
	"""复用 pde_sample_dream.py 中的配置读取与 PDE 子配置挂载逻辑。"""
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


def extract_long_code(full_output: str) -> str:
	"""复制自 pde_sample_dream.py：从 LLM 完整输出中提取最长的 ```python 代码块。"""
	matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
	if matches:
		code_output = max(matches, key=len).strip()
	else:
		code_output = "We can not extract the code in the output. "
	return code_output


def get_data_chunk(data: list, num_node: int, node_idx: int) -> list:
	"""与 pde_sample_dream.py 保持一致的多机切分逻辑。"""
	total = len(data)
	chunk_size = (total + num_node - 1) // num_node
	start_idx = node_idx * chunk_size
	end_idx = min((node_idx + 1) * chunk_size, total)
	return data[start_idx:end_idx]

#####################################################################
#####################################################################
# 重掩码 + 解码 infilling 实现用函数

def load_first_round_outputs(config) -> Dict[str, Any]:
	"""加载第一次采样+执行的结果 JSON / 目录结构。

	- 根据 config.experiment.project 和 model / dataset 构造 outputs-*.json 路径；
	- 读取 JSON，返回包含 data 列表和辅助索引的信息。

	具体实现尽量与 pde_execute.py 中的逻辑保持一致，
	这样 round_idx 与 execute 子目录能够一一对应：
	- outputs 文件名: ../{project}/temp_data/outputs-*.json
	- round_idx: 按 data[i]["full_output"][j] / ["extracted_output"][j] 的双重循环顺序递增。

	返回字典结构：
	{
	  "data": 原始 JSON 列表,
	  "round_map": { round_idx: {"prompt": ..., "full_output": ..., "item_index": i, "sample_index": j } },
	  "file_path": outputs JSON 的完整路径,
	  "outputs_name": 基础 outputs 名称
	}
	"""
	project_name = config.experiment.project
	pretrained_model = config.model
	dataset = config.dataset.eval_dataset

	outputs_name = "eval-" + str(pretrained_model).replace("/", ".") + "-" + str(dataset)
	num_node = config.experiment.num_node if config.experiment.num_node is not None else 1
	node_index = config.experiment.node_index if config.experiment.node_index is not None else 0

	if num_node > 1:
		file_path = os.path.join("..", project_name, "temp_data", f"outputs-{node_index}-{outputs_name}.json")
	else:
		file_path = os.path.join("..", project_name, "temp_data", f"outputs-{outputs_name}.json")

	if not os.path.exists(file_path):
		raise FileNotFoundError(f"first-round outputs JSON not found at {file_path}")

	with open(file_path, "r", encoding="utf-8", errors="surrogatepass") as f:
		data = json.load(f)

	round_map: Dict[int, Dict[str, Any]] = {}
	counter = 0
	# 对sample中落盘的 json 文件解析
	# json 文件: 仅包含 1 个 dict 元素的 list
	for item_idx, item in enumerate(data):
		full_outputs = item.get("full_output", []) or []
		extracted_outputs = item.get("extracted_output", []) or []
		m = min(len(full_outputs), len(extracted_outputs))  # 实际上生成出代码块的回复数量
		prompt = item.get("prompt", "")

		for sample_idx in range(m):
			round_map[counter] = {  # 将真正成功的解一一打包
				"prompt": prompt,
				"full_output": full_outputs[sample_idx],
				"sample_index": sample_idx,
			}
			counter += 1

	return {
		"data": data,
		"round_map": round_map,
		"file_path": file_path,
		"outputs_name": outputs_name,
	}

# 读取 csv, 返回信息
def load_execute_results(project_name: str) -> List[Dict[str, Any]]:
	"""读取 project_name/execute/test_results.csv，解析 round / error_type / error_line。

	返回一个字典列表，每个元素至少包含：
	- round_idx
	- error_type
	- error_line（可能为空字符串）

	"""
	exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))
	result_sheet_path = os.path.join(exec_root, "test_results.csv")
	if not os.path.exists(result_sheet_path):
		raise FileNotFoundError(f"test_results.csv not found at {result_sheet_path}")

	results: List[Dict[str, Any]] = []
	with open(result_sheet_path, "r", encoding="utf-8", errors="surrogatepass") as f:
		lines = f.readlines()

	# 第一行是表头: round,nRMSE,elapsed_time,convergence_rate,num_trial,error_type,error_line
	for line in lines[1:]:
		line = line.strip()
		if not line:
			continue
		parts = line.split(",")
		# 兼容性处理: 至少应该有前 5 列, 后 2 列缺失则置空
		if len(parts) < 5:
			continue
		try:
			round_idx = int(parts[0])
		except ValueError:
			# 若 round_idx 无法解析为整数, 跳过该行
			continue

		# 读取错误类型与错误行号（如果存在）
		error_type = parts[5] if len(parts) > 5 else ""
		error_line = parts[6] if len(parts) > 6 else ""

		results.append(
			{
				"round_idx": round_idx,
				"error_type": error_type,
				"error_line": error_line,
			}
		)

	return results

# 定位 solver 需重掩码代码块 起止行
def locate_error_block_in_solver(
	solver_path: str,
	error_line: int,
	k_up: int = 5,
	k_down: int = 5,
) -> Tuple[int, int]:
	"""在 solver.py 中基于“缩进 + 空行 + k 行上限”定位错误代码块。

	设计思路（待实现）：
	- 读取文件到行列表（1-based 索引）。
	- 获取报错行 L 的缩进 indent_L；
	- 向上扩展：
	  * 遇空行停止；
	  * indent < indent_L 时，若是 if/for/while/def/class 等块头，可将其作为块头并停止；否则停止；
	  * 其它行纳入，最多扩展 k_up 行；
	- 向下扩展：
	  * 遇空行或 indent < indent_L 停止；
	  * 其它行纳入，最多扩展 k_down 行；
	- 返回 (start_line, end_line)。
	"""
	if not os.path.exists(solver_path):
		return error_line, error_line

	with open(solver_path, "r", encoding="utf-8", errors="surrogatepass") as f:
		lines = f.readlines()

	if not lines:
		return error_line, error_line

	n_lines = len(lines)
	# 保证 error_line 在有效范围内
	if error_line < 1:
		error_line = 1
	elif error_line > n_lines:
		error_line = n_lines

	# 将索引转换为 0-based
	idx = error_line - 1

	def indent_level(s: str) -> int:
		"""简单计算缩进：统计行首空格和制表符数量。"""
		stripped = s.lstrip("\t ")
		return len(s) - len(stripped)

	# 若报错行是空行，则尝试在附近寻找最近的非空行作为参考
	def find_nearby_nonempty(start_idx: int) -> int:
		if lines[start_idx].strip():
			return start_idx
		# 先向上再向下，限制在 k_up/k_down 范围内
		for offset in range(1, max(k_up, k_down) + 1):
			up = start_idx - offset
			down = start_idx + offset
			if up >= 0 and lines[up].strip():
				return up
			if down < n_lines and lines[down].strip():
				return down
		return start_idx

	idx = find_nearby_nonempty(idx)
	base_line = lines[idx]
	indent_L = indent_level(base_line)

	# 向上扩展
	start_idx = idx
	steps_up = 0
	while start_idx > 0 and steps_up < k_up:
		candidate_idx = start_idx - 1
		line = lines[candidate_idx]
		stripped_line = line.strip()
		if not stripped_line:
			break
		if stripped_line.startswith("#"):
			break
		# 遇到三引号开头/结尾的行(多行注释/docstring边界), 视作块边界, 不再向上扩展
		if '"""' in line or "'''" in line:
			break

		ind = indent_level(line)
		stripped = line.lstrip()

		# 若缩进更少（无论是否为块头），视为离开当前块，直接停止，但不把该行纳入块中
		if ind < indent_L:
			break

		# 同级或更深缩进，纳入块
		start_idx = candidate_idx
		steps_up += 1

	# 向下扩展
	end_idx = idx
	steps_down = 0
	while end_idx < n_lines - 1 and steps_down < k_down:
		candidate_idx = end_idx + 1
		line = lines[candidate_idx]
		stripped_line = line.strip()
		if not stripped_line:
			break
		if stripped_line.startswith("#"):
			break
		# 同样, 三引号行作为块边界
		if '"""' in line or "'''" in line:
			break

		ind = indent_level(line)
		# 碰到更浅缩进，认为已离开当前块
		if ind < indent_L:
			break

		end_idx = candidate_idx
		steps_down += 1

	# 返回 1-based 行号
	return start_idx + 1, end_idx + 1

# 将 solver 中的错误代码块映射回 response
def map_block_back_to_response(
	full_output: str,
	solver_block_lines: List[str],
) -> Tuple[int, int]:
	"""将 solver.py 中的错误代码块映射回原始 response 文本中的位置。

	可能的策略：
	- 先用 extract_long_code(full_output) 拿到代码字符串 code_in_resp；
	- 在 code_in_resp 中用字符串匹配定位 solver_block_lines 拼接后的子串；
	- 再将其转换为 full_output 中的绝对字符区间 (start_char, end_char)。

	返回：在 full_output 中的字符区间 (start_char, end_char)。

	"""
	if not solver_block_lines:
		return 0, 0

	# 1) 在 full_output 中找到与 extract_long_code 相同的最大 python 代码块
	pattern = r"```python(.*?)```"
	matches = list(re.finditer(pattern, full_output, re.DOTALL))
	if not matches:
		return 0, 0

	# 选取与 extract_long_code 同样准则的最大块
	best_match = None
	best_len = -1
	for m in matches:
		code_segment = m.group(1)
		seg_len = len(code_segment)
		if seg_len > best_len:
			best_len = seg_len
			best_match = m

	if best_match is None:
		return 0, 0

	code_region_text = best_match.group(1)
	code_region_start = best_match.start(1)  # 在 full_output 中的起始下标

	# 2) 按行对比 solver_block_lines 与 code_region_text 中的行，找到匹配窗口
	def norm_line(s: str) -> str:
		# 仅去掉行尾空白，保留前导缩进以维持结构
		return s.rstrip("\n\r ")

	resp_lines = code_region_text.split("\n")
	resp_norm = [norm_line(l) for l in resp_lines]
	solver_norm = [norm_line(l) for l in solver_block_lines]

	block_len = len(solver_norm)
	if block_len == 0:
		return code_region_start, code_region_start

	start_idx_in_code = None
	for i in range(0, len(resp_norm) - block_len + 1):
		if resp_norm[i : i + block_len] == solver_norm:
			start_idx_in_code = i
			break

	# 若逐行匹配失败，退化为子串搜索
	if start_idx_in_code is None:
		block_str = "\n".join(solver_block_lines).strip()
		idx = code_region_text.find(block_str)
		if idx == -1:
			return code_region_start, code_region_start
			
		start_char = code_region_start + idx
		end_char = start_char + len(block_str)
		return start_char, end_char

	# 3) 根据行号计算在 code_region_text 内部的字符区间
	# 使用不带换行符的 resp_lines 来计算偏移，再显式加上 "\n"
	char_offset_start = 0
	for j in range(start_idx_in_code):
		char_offset_start += len(resp_lines[j]) + 1  # 加上换行符

	char_offset_end = char_offset_start
	for j in range(start_idx_in_code, start_idx_in_code + block_len):
		char_offset_end += len(resp_lines[j])
		if j < start_idx_in_code + block_len - 1:
			char_offset_end += 1  # 行间的换行符

	start_char = code_region_start + char_offset_start
	end_char = code_region_start + char_offset_end
	return start_char, end_char

# 根据 token_span, 修改原 response 对应的 token ids --> masked token ids
def build_masked_response(
	tokenizer: DreamTokenizer,
	response: str,
	token_span: Tuple[int, int],
	num_mask_tokens: int,
	mask_token_id: int,

) -> List[int]:
	"""基于 token_span 在 response 中构造 *固定长度* 的 mask 区域。

	实现逻辑:
	- 先对整段 response 做一次编码 (add_special_tokens=False)，得到 input_ids；
	- 将 [tok_start, tok_end) 范围内的原 token 全部丢弃；
	- 无论原来有多少个 token，统一插入 num_mask_tokens 个 mask_token_id；
	- 再将新的 token 序列 decode 回文本并返回，用于后续送入模型或调试观察。

	注意: mask_token_id 由上层统一从 model.config.mask_token_id 传入，
	以与 SFT 训练脚本保持一致。
	"""

	# 对完整 response 做一次编码，拿到原始 token 序列
	enc_all = tokenizer(response, add_special_tokens=False)
	input_ids: List[int] = enc_all["input_ids"]

	tok_start, tok_end = token_span
	# 边界保护，避免越界
	if tok_start < 0:
		tok_start = 0
	if tok_end < tok_start:
		tok_end = tok_start
	if tok_start > len(input_ids):
		tok_start = len(input_ids)
	if tok_end > len(input_ids):
		tok_end = len(input_ids)

	# 在 token 级别执行替换: 丢弃原区间，插入固定长度的 mask_token 序列
	prefix_ids = input_ids[:tok_start]
	suffix_ids = input_ids[tok_end:]
	masked_ids = prefix_ids + [mask_token_id] * int(num_mask_tokens) + suffix_ids

	# 若原 block 末尾的换行也被一起移除，则在 mask 段之后补回一个换行 token，
	# 避免 masked 区域与下一行代码直接贴合。
	newline_ids = tokenizer("\n", add_special_tokens=False).get("input_ids", [])
	if newline_ids:
		tail_prefix = input_ids[tok_end : tok_end + len(newline_ids)]
		if tail_prefix != newline_ids:
			masked_ids = prefix_ids + [mask_token_id] * int(num_mask_tokens) + newline_ids + suffix_ids
	# 返回修改后的 token id 序列，供上层根据需要自行 decode 或送入模型
	return masked_ids

# 将 response 中的字符区间映射到 token_span
def map_response_span_to_token_span(
	tokenizer: DreamTokenizer,
	response: str,
	span: Tuple[int, int],
) -> Tuple[int, int]:
	"""将 response 中的字符区间 span 映射到 tokenizer token 序号区间。

	返回值为 (token_start, token_end)，满足：
	- token_start 为第一个受影响 token 的下标（含）；
	- token_end 为最后一个受影响 token 之后的位置（不含）。

	优先尝试使用 offset_mapping；若不可用，则退化为
	"分别对前缀和 span 单独编码" 的近似方法。
	"""
	start_char, end_char = span
	if start_char < 0:
		start_char = 0
	if end_char < start_char:
		end_char = start_char
	if end_char > len(response):
		end_char = len(response)

	# 优先尝试 fast tokenizer 的 offset_mapping
	try:
		enc = tokenizer(
			response,
			return_offsets_mapping=True,
			add_special_tokens=False,
		)
		offsets = enc.get("offset_mapping", None)
		if offsets is not None:
			token_start = None
			token_end = None

			for i, (s, e) in enumerate(offsets):
				# 第一次覆盖到 span 起点的 token
				if token_start is None and s <= start_char < e:
					token_start = i
				# 最后一个与 span 相交的 token
				if s < end_char <= e:
					token_end = i + 1

			if token_start is None:
				for i, (s, e) in enumerate(offsets):
					if s >= start_char:
						token_start = i
						break

			if token_end is None:
				for i in range(len(offsets) - 1, -1, -1):
					s, e = offsets[i]
					if e <= end_char:
						token_end = i + 1
						break

			if token_start is None:
				token_start = 0
			if token_end is None:
				token_end = len(offsets)

			return token_start, token_end
	except (TypeError, NotImplementedError, AttributeError):
		# 部分 tokenizer 可能不支持 return_offsets_mapping, 或者 slow tokenizer 未实现该特性
		pass

	# 退化方案：分别编码前缀与 span，以 token 数量近似映射
	prefix_enc = tokenizer(
		response[:start_char],
		add_special_tokens=False,
	)
	span_enc = tokenizer(
		response[start_char:end_char],
		add_special_tokens=False,
	)

	token_start = len(prefix_enc["input_ids"])
	token_end = token_start + len(span_enc["input_ids"])
	return token_start, token_end

# 调用 pde_sample_dream.py 中的 _sample, 利用全局注意力对 MASK 进行解码
def refine_with_dream_model(
	config,
	tokenizer: DreamTokenizer,
	prompt: str,
	masked_response_ids: List[int],
	pretrained_model: str,
) -> str:
	"""利用 Dream 模型的多步扩散采样，对 [MASK] 段进行与采样脚本一致的单 block 重新生成。"""
	if not masked_response_ids:
		return ""

	model = get_dream_model(pretrained_model)
	device = next(model.parameters()).device

	# 与 SFT 训练脚本对齐: 始终使用 model.config.mask_token_id 作为掩码 token
	mask_token_id = getattr(model.config, "mask_token_id", None)
	if mask_token_id is None:
		raise ValueError("model.config.mask_token_id is required for refine_with_dream_model")

	# 定位 mask 区间
	masked_ids = list(masked_response_ids)
	try:
		mask_start = masked_ids.index(mask_token_id)
	except ValueError:
		return tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

	mask_end = mask_start
	while mask_end < len(masked_ids) and masked_ids[mask_end] == mask_token_id:
		mask_end += 1
	mask_len = mask_end - mask_start
	if mask_len <= 0:
		return tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

	prefix_ids = masked_ids[:mask_start]
	suffix_ids = masked_ids[mask_end:]

	# prompt + 已知前缀 作为输入序列，保持尾部 suffix 以便后续拼接
	prompt_ids = tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
	input_prefix_ids = prompt_ids + prefix_ids
	input_len = len(input_prefix_ids)

	if input_len > 0:
		input_ids = torch.tensor([input_prefix_ids], dtype=torch.long, device=device)
		attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
	else:
		input_ids = torch.empty((1, 0), dtype=torch.long, device=device)
		attention_mask = torch.empty((1, 0), dtype=torch.bool, device=device)

	suffix_tensor = None
	if suffix_ids:
		suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=device)

	# 与 SFT 训练脚本对齐: 始终使用 model.config.pad_token_id 作为 padding token
	pad_token_id = getattr(model.config, "pad_token_id", None)
	if pad_token_id is None:
		raise ValueError("model.config.pad_token_id is required for refine_with_dream_model")

	eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
	if eos_id is None or eos_id < 0:
		eos_id = getattr(tokenizer, "eos_token_id", None)
	if eos_id is None:
		eos_id = getattr(model.config, "eos_token_id", mask_token_id)

	rollout_cfg = config.rollout
	steps = rollout_cfg.steps if rollout_cfg.steps is not None else 32
	temperature = rollout_cfg.temperature if rollout_cfg.temperature is not None else 0.0
	top_p = rollout_cfg.top_p  # 默认 None 即不启用 top-p 约束
	top_k = rollout_cfg.top_k  # 同上，直接读取配置
	target_conf = rollout_cfg.target
	alg_temp = rollout_cfg.alg_temp
	pad_target_penalty = rollout_cfg.pad_target_penalty if rollout_cfg.pad_target_penalty is not None else 1.0
	use_cache = rollout_cfg.use_cache if rollout_cfg.use_cache is not None else True
	further_horizon = rollout_cfg.further_horizon
	if not use_cache:
		further_horizon = None
	remask_strategy = rollout_cfg.remasking_strategy if rollout_cfg.remasking_strategy is not None else ""
	if remask_strategy == "low_confidence_static":
		unmask_threshold = None
	else:
		unmask_threshold = rollout_cfg.dynamic_threshold

	gen_config = DreamGenerationConfig(
		output_history=False,
		return_dict_in_generate=True,
		max_length=input_len + mask_len + len(suffix_ids),
		steps=steps,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
		tar=target_conf,
		alg_temp=alg_temp,
	)

	block_length = mask_len
	if block_length <= 0:
		return tokenizer.decode(
			masked_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

	generation_ids = dream_block_sample(
		model,
		input_ids,
		attention_mask=attention_mask,
		generation_config=gen_config,
		block_length=block_length,
		use_cache=use_cache,
		further_horizon=further_horizon,
		mask_token_id=mask_token_id,
		eos_token_id=eos_id,
		pad_token_id=pad_token_id,
		pad_target_penalty=pad_target_penalty,
		unmask_threshold=unmask_threshold,
		frozen_tail_ids=suffix_tensor,
	)

	sequences_tensor = generation_ids.sequences if hasattr(generation_ids, "sequences") else generation_ids
	sequences = sequences_tensor.detach().cpu()
	prompt_len = len(prompt_ids)
	response_total_len = len(masked_ids)
	response_ids = sequences[0, prompt_len : prompt_len + response_total_len].tolist()

	# 仅删除显式 eos token，保持其余 token 原样
	if eos_id is not None:
		response_ids = [tok for tok in response_ids if tok != eos_id]

	refined_text = tokenizer.decode(
		response_ids,
		skip_special_tokens=False,
		clean_up_tokenization_spaces=False,
	)
	# eos_token 字面值仍可能出现在文本层（如 <|im_end|>），此处仅针对 tokenizer.eos_token 字符串做一次干净移除
	eos_text = tokenizer.eos_token
	if eos_text:
		refined_text = refined_text.replace(eos_text, "")
	return refined_text

#####################################################################
#####################################################################


if __name__ == "__main__":
	config = get_config()
	assigned_gpu = getattr(config, "assigned_gpu", None)
	if assigned_gpu is not None:
		os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(assigned_gpu))

	# multiprocessing 默认使用 fork，某些场景会导致 CUDA 句柄冲突，这里强制使用 spawn
	mp.set_start_method("spawn", force=True)

	cprint("[main] Start PDE refine pipeline.", "cyan")
	start_time = time.time()
	summary = run_multi_step_refine_loop(config)
	total_refine_time = time.time() - start_time
	cprint("[main] PDE refine pipeline finished.", "green")

	# 将 refine 阶段的统计信息与总耗时落盘到 project/results 目录
	project_name = config.experiment.project
	dataset_cfg = getattr(config, "dataset", None)
	dataset_name = getattr(dataset_cfg, "eval_dataset", None) if dataset_cfg is not None else None
	data_type = getattr(dataset_cfg, "data_type", None) if dataset_cfg is not None else None
	pde_type = getattr(dataset_cfg, "pde_type", None) if dataset_cfg is not None else None

	results_dir = os.path.join("..", project_name, "results")
	os.makedirs(results_dir, exist_ok=True)
	metrics_path = os.path.join(results_dir, "refine_metrics.json")

	metrics = {
		"model": config.model,
		"dataset": dataset_name,
		"data_type": data_type,
		"pde_type": pde_type,
		"num_node": getattr(config.experiment, "num_node", None),
		"node_index": getattr(config.experiment, "node_index", None),
		"assigned_gpu": getattr(config, "assigned_gpu", None),
		"refine_enabled": summary.get("refine_enabled", False),
		"refine_max_steps": summary.get("max_steps", 0),
		"actual_iterations": summary.get("actual_iterations", 0),
		"total_refined_rounds": summary.get("total_refined_rounds", 0),
		"refined_rounds_per_step": summary.get("refined_rounds_per_step", []),
		"execute_metrics_before_refine": summary.get("execute_metrics_before_refine", {}),
		"iterations": summary.get("iterations", []),
		"total_refine_time_sec": float(total_refine_time),
	}

	with open(metrics_path, "w", encoding="utf-8") as f_metrics:
		json.dump(metrics, f_metrics, indent=2, ensure_ascii=False)
