from __future__ import annotations

from collections import defaultdict
import json
import math
import os
import re
from typing import Any, Dict, List

from omegaconf import OmegaConf
from termcolor import cprint

if __package__:
	from reward.refine_buffer_utils import append_records
else:
	from refine_buffer_utils import append_records


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


def z_score_normalize(lst: List[float], eps: float = 1e-8) -> List[float]:
	if not lst:
		return []
	mean = sum(lst) / len(lst)
	var = sum((x - mean) ** 2 for x in lst) / len(lst)
	std = var ** 0.5
	denom = std if std > eps else eps
	return [(x - mean) / denom for x in lst]


def compute_single_response_score(metrics: Any, nrmse_threshold: float) -> tuple[float, bool, bool]:
	"""根据 evaluator 返回的 metrics 计算单个候选的 reward。

	规则（假设 nrmse_threshold=0.1）：
	- 成功 + nRMSE <= 阈值          -> reward = 1.0  (加入 buffer )
	- 成功 + nRMSE > 阈值 且 非 NaN -> reward = 0.2

	- 报错(非syntax)  				-> reward = 0.0  (加入 buffer)
	- timeout / nRMSE 为 NaN 		-> reward = 0.0  
	- 报错且是语法类错误           	-> reward = 0.0
	
	- 无法提取 solver (no_solver)  -> reward = -1.0

	额外：该函数还会决定该候选是否应该被加入 refine buffer。
	- should_add_to_buffer=True: 该候选会被挑进 refine buffer（用于后续 refine/infilling 训练数据来源）
	
	返回值:
	- reward: 上述区间内的标量
	- is_bad: 候选质量是否差（用于后续在 refine 训练中区别对待“好/坏”样本）
	- should_add_to_buffer: 是否应加入 refine buffer（独立于 is_bad）
	"""
	# 默认视为“坏样本”，reward=0.0，且默认不加入 refine buffer
	reward = 0.0
	is_bad = True
	should_add_to_buffer = False

	if not isinstance(metrics, dict):
		return reward, is_bad, should_add_to_buffer

	status = metrics.get("status")
	error_type = str(metrics.get("error_type", "") or "")
	nrmse_val = metrics.get("nRMSE")

	# 无法提取 solver：属于格式/抽取失败，不按“报错(非 syntax)”规则进 buffer
	if status == "no_solver":
		return -1.0, True, False

	# 成功跑通的情况
	if status == "ok" and nrmse_val is not None:
		try:
			nrmse = float(nrmse_val)
		except Exception:
			return 0.0, True, False

		# nRMSE 为 NaN/Inf：视作数值失败
		if not math.isfinite(nrmse):
			return 0.0, True, False

		# 好的数值解：nRMSE 较小
		if nrmse <= nrmse_threshold:
			# 仅将 reward=1.0 的最优解加入 refine buffer
			return 1.0, False, True
		# 能跑通但误差较大
		return 0.2, True, False

	# 其余都是报错 / timeout 等
	# 语法类错误单独强惩罚
	if error_type in {"SyntaxError", "IndentationError"}:
		# syntax 类错误不加入 refine buffer
		return 0.0, True, False

	# 报错但非 syntax：加入 refine buffer
	# 覆盖 timeout / no_output / parse_failed / exception / NameError / TypeError / ...
	if status == "failed":
		# timeout 不加入 refine buffer
		if error_type == "timeout":
			return 0.0, True, False
		return 0.0, True, True

	# 其它报错统一 reward = 0.0
	return reward, is_bad, should_add_to_buffer


def _read_text(path: str) -> str:
	try:
		with open(path, "r", encoding="utf-8", errors="surrogatepass") as f:
			return f.read()
	except Exception:
		return ""


def _normalized_error_len(text: str) -> int:
	"""将 errors 文本做轻量归一化后计长。

	目的：降低空白符差异的影响，更贴近“报错项是否变短”。
	"""
	if not text:
		return 0
	return len(" ".join(str(text).split()))


_SYNTAX_ERROR_TYPES = {
	"SyntaxError",
	"IndentationError",
	"TabError",
}


_EXC_LINE_RE = re.compile(
	r"^\s*(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception))\s*:\s*(?P<msg>.*)\s*$"
)


def _normalize_error_message(msg: str, *, max_len: int = 200) -> str:
	if not msg:
		return ""
	s = " ".join(str(msg).split())
	# normalize volatile tokens (line numbers, ids, addresses)
	s = re.sub(r"0x[0-9a-fA-F]+", "<hex>", s)
	s = re.sub(r"\b\d+\b", "<num>", s)
	# reduce long paths (best-effort)
	s = re.sub(r"(/[^\s]+)+", "<path>", s)
	if len(s) > int(max_len):
		s = s[: int(max_len)]
	return s


def _extract_error_signatures(text: str) -> tuple[set[str], set[str]]:
	"""从 errors 文本中抽取可比对的“错误指纹”。

	返回 (syntax_sigs, non_syntax_sigs)。
	"""
	syntax_sigs: set[str] = set()
	non_syntax_sigs: set[str] = set()
	if not text:
		return syntax_sigs, non_syntax_sigs

	for raw_line in str(text).splitlines():
		line = raw_line.strip()
		if not line:
			continue
		m = _EXC_LINE_RE.match(line)
		if not m:
			continue
		exc_full = str(m.group("exc") or "").strip()
		exc = exc_full.split(".")[-1]
		msg = _normalize_error_message(str(m.group("msg") or ""))
		sig = f"{exc}:{msg}" if msg else f"{exc}:"
		if exc in _SYNTAX_ERROR_TYPES:
			syntax_sigs.add(sig)
		else:
			non_syntax_sigs.add(sig)

	# fallback：如果没抽到任何 exception 行，用首行文本做一个粗粒度指纹
	if not syntax_sigs and not non_syntax_sigs:
		for raw_line in str(text).splitlines():
			line = raw_line.strip()
			if not line:
				continue
			non_syntax_sigs.add(f"TEXT:{_normalize_error_message(line)}")
			break

	return syntax_sigs, non_syntax_sigs


def _is_partial_improved(
	*,
	origin_errors_path: str,
	new_errors_path: str,
	allowed_new_non_syntax_errors: int,
	min_shorten_chars: int,
	min_shorten_ratio: float,
) -> bool:
	old_text = _read_text(origin_errors_path)
	new_text = _read_text(new_errors_path)

	# 方案：优先用“错误指纹集合”做子集/容错判定；如果无法抽取指纹，再回退到“长度变短”启发式
	old_syntax, old_non_syntax = _extract_error_signatures(old_text)
	new_syntax, new_non_syntax = _extract_error_signatures(new_text)

	old_has_sigs = bool(old_syntax or old_non_syntax)
	new_has_sigs = bool(new_syntax or new_non_syntax)

	if old_has_sigs and new_has_sigs:
		# 不允许引入新的语法类错误（语法错误在 reward 里通常会单独强惩罚）
		added_syntax = new_syntax - old_syntax
		if added_syntax:
			return False

		allowed = max(0, int(allowed_new_non_syntax_errors))
		added_non_syntax = new_non_syntax - old_non_syntax
		if len(added_non_syntax) > allowed:
			return False

		# 至少需要“消除/减少”一些旧错误指纹，才算明显改进
		removed_any = (old_syntax | old_non_syntax) - (new_syntax | new_non_syntax)
		if not removed_any:
			return False
		return True

	# fallback: length heuristic
	old_len = _normalized_error_len(old_text)
	new_len = _normalized_error_len(new_text)
	if old_len <= 0:
		return False
	if new_len < 0:
		return False
	if new_len >= old_len:
		return False
	delta = old_len - new_len
	if delta >= max(0, int(min_shorten_chars)):
		return True
	# ratio
	try:
		r = float(delta) / float(old_len)
	except Exception:
		r = 0.0
	return r >= float(min_shorten_ratio)


def compute_refine_response_score(
	metrics: Any,
	*,
	nrmse_threshold: float,
	origin_is_bad: bool,
	partial_improved: bool,
	ok_high_nrmse_reward_when_good: float,
	partial_improve_reward_when_bad: float,
) -> tuple[float, bool, bool]:
	"""refine 相位专用 reward。

	约束：
	- scratch 的奖励规则保持在 compute_single_response_score() 中。
	- refine 的 is_bad 来自 rollout_refine 的 refine_meta.is_bad（即“修复前的样本是否坏”）。
	"""
	# 默认：refine 不写入 buffer（buffer 写入只在 train_scratch 阶段发生）
	should_add_to_buffer = False
	is_bad = True

	if not isinstance(metrics, dict):
		return 0.0, True, should_add_to_buffer

	status = metrics.get("status")
	error_type = str(metrics.get("error_type", "") or "")
	nrmse_val = metrics.get("nRMSE")

	if status == "no_solver":
		return -1.0, True, should_add_to_buffer

	# 成功跑通
	if status == "ok" and nrmse_val is not None:
		try:
			nrmse = float(nrmse_val)
		except Exception:
			return 0.0, True, should_add_to_buffer
		if not math.isfinite(nrmse):
			return 0.0, True, should_add_to_buffer
		# ===== 两类 refine 样本分开定义 =====
		# (1) origin_is_bad=false：属于“简单任务/好样本”，仍以 nRMSE 阈值为主
		if not bool(origin_is_bad):
			if nrmse <= float(nrmse_threshold):
				return 1.0, False, should_add_to_buffer
			# 能跑通但误差较大：给一个中等奖励（默认 0.5）
			return float(ok_high_nrmse_reward_when_good), True, should_add_to_buffer

		# (2) origin_is_bad=true：属于“困难任务/坏样本”，只要能跑通就算成功
		return 1.0, False, should_add_to_buffer

	# 语法类错误：困难任务强惩罚（-1.0）；简单任务保持 0.0
	if error_type in {"SyntaxError", "IndentationError"}:
		if bool(origin_is_bad):
			return -1.0, True, should_add_to_buffer
		return 0.0, True, should_add_to_buffer

	# failed：简单任务保持 0.0；困难任务允许“部分改进”加分（作为过程奖励 bonus）
	if status == "failed":
		base = 0.0
		bonus = 0.0
		if bool(origin_is_bad) and partial_improved and error_type != "timeout":
			bonus = float(partial_improve_reward_when_bad)
		reward = float(base + bonus)
		# 避免出现比 1.0 更高的 reward
		reward = min(1.0, reward)
		return reward, True, should_add_to_buffer

	# 其它报错统一 0.0；困难任务允许“部分改进”bonus
	base = 0.0
	bonus = 0.0
	if bool(origin_is_bad) and partial_improved:
		bonus = float(partial_improve_reward_when_bad)
	reward = float(base + bonus)
	reward = min(1.0, reward)
	return reward, True, should_add_to_buffer


def compute_group_rewards_for_grpo(scores: List[float]) -> List[float]:
	"""对一个 group 内的单样本 reward 做后处理。

	当前实现：对 group 内 reward 做 z-score 归一化（GRPO 常用的组内相对优势）。
	使用一个极小的 eps 防止 std=0 时分母为 0。
	"""
	return z_score_normalize(scores, eps=1e-8)


def main() -> None:
	##############################
	##############################
	# 读取 config + JSON文件 ( output-rl-{pde}-step*.json )

	cfg = get_config()
	project_name = str(cfg.experiment.project)
	current_epoch = int(getattr(cfg.experiment, "current_epoch", 1) or 1)
	pde_type = str(cfg.dataset.pde_type)
	phase = str(getattr(cfg.experiment, "function", ""))

	nrmse_threshold = float(OmegaConf.select(cfg, "reward.nrmse_threshold", default=1.0))

	num_node = int(getattr(cfg.experiment, "num_node", 1) or 1)
	node_index = int(getattr(cfg.experiment, "node_index", 0) or 0)

	# refine attempt（多轮 refine 时用于隔离 outputs/execute 目录）
	attempt_raw = OmegaConf.select(cfg, "experiment.refine_attempt", default=None)
	try:
		attempt_raw_s = str(attempt_raw).strip() if attempt_raw is not None else None
	except Exception:
		attempt_raw_s = None

	def _is_aggregate_attempt(x: Any) -> bool:
		if x is None:
			return True
		try:
			xs = str(x).strip().lower()
		except Exception:
			xs = ""
		if xs in {"all", "agg", "aggregate"}:
			return True
		try:
			return int(x) == 0
		except Exception:
			return False

	def _outputs_file_path(outputs_name: str) -> str:
		if num_node > 1:
			return f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
		return f"../{project_name}/temp_data/outputs-{outputs_name}.json"

	suffix = ""
	if phase in {"train_refine", "refine"}:
		suffix = "-refine"

	# 多轮 refine attempt：由 refine_rollout.multi_attempt.* 控制；聚合模式由 experiment.refine_attempt=0/all 触发
	multi_attempt_enable = bool(
		OmegaConf.select(cfg, "refine_rollout.multi_attempt.enable", default=False)
	)
	num_attempts = int(
		OmegaConf.select(cfg, "refine_rollout.multi_attempt.num_attempts", default=1) or 1
	)
	aggregate_refine = bool(
		suffix == "-refine" and multi_attempt_enable and num_attempts > 1 and _is_aggregate_attempt(attempt_raw)
	)

	# outputs data
	data: List[Dict[str, Any]] = []
	attempt_datas: List[tuple[int, List[Dict[str, Any]]]] = []
	if aggregate_refine:
		loaded = 0
		for a in range(1, int(num_attempts) + 1):
			outputs_name_a = f"rl-{pde_type}-step{current_epoch}-refine-a{int(a)}"
			file_name_a = _outputs_file_path(outputs_name_a)
			# backward compatibility：如果 a1 不存在，尝试无 attempt 后缀
			if (not os.path.exists(file_name_a)) and int(a) == 1:
				outputs_name_fallback = f"rl-{pde_type}-step{current_epoch}-refine"
				file_name_fallback = _outputs_file_path(outputs_name_fallback)
				if os.path.exists(file_name_fallback):
					file_name_a = file_name_fallback
					cprint(
						f"[pde_rl_reward] refine multi-attempt: fallback to {file_name_a}",
						"yellow",
					)

			if not os.path.exists(file_name_a):
				continue
			cprint(f"[pde_rl_reward] loading {file_name_a}", "green")
			try:
				with open(file_name_a, "r", encoding="utf-8") as f:
					d = json.load(f)
			except Exception:
				d = []
			if isinstance(d, list) and d:
				attempt_datas.append((int(a), d))
				loaded += 1
		cprint(
			f"[pde_rl_reward] refine multi-attempt aggregate: loaded {loaded} attempts (expected={int(num_attempts)})",
			"green",
		)
	else:
		attempt_tag = ""
		if suffix == "-refine" and attempt_raw is not None:
			try:
				attempt_i = int(attempt_raw)
			except Exception:
				attempt_i = None
			if attempt_i is not None and int(attempt_i) > 0:
				attempt_tag = f"-a{int(attempt_i)}"
		outputs_name = f"rl-{pde_type}-step{current_epoch}{suffix}{attempt_tag}"
		file_name = _outputs_file_path(outputs_name)
		cprint(f"[pde_rl_reward] loading {file_name}", "green")
		with open(file_name, "r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, list):
			data = []

	##############################
	##############################


	##############################
	##############################
	# 计算 reward 并 重构数据格式用于训练 --> task/response 结构 改为 每个 response 单独记录, 并存储为一个新的 JSON 文件

	final_data: List[Dict[str, Any]] = []
	refine_records: List[Dict[str, Any]] = []
	sample_type = "refine" if suffix else "scratch"

	# execute 目录（用于 refine 的“部分改进”判定）
	# 单 attempt 时 exec_dir_name 不带 -a；多 attempt 聚合时会在循环里按 a1/a2/... 切换
	exec_dir_name = f"step-{current_epoch}{suffix}"
	if suffix == "-refine" and (not aggregate_refine) and attempt_raw is not None:
		try:
			attempt_i = int(attempt_raw)
		except Exception:
			attempt_i = None
		if attempt_i is not None and int(attempt_i) > 0:
			exec_dir_name = f"step-{current_epoch}{suffix}-a{int(attempt_i)}"
	exec_root = os.path.abspath(os.path.join("..", project_name, "execute", exec_dir_name, pde_type))

	# refine reward 参数
	ok_high_nrmse_reward_when_good = float(
		OmegaConf.select(cfg, "reward.refine.good.ok_high_nrmse_reward", default=0.5)
	)
	partial_improve_reward_when_bad = float(
		OmegaConf.select(cfg, "reward.refine.bad.partial_improve_reward", default=0.1)
	)
	allowed_new_non_syntax_errors = int(
		OmegaConf.select(cfg, "reward.refine.bad.allowed_new_non_syntax_errors", default=0) or 0
	)
	min_shorten_chars = int(
		OmegaConf.select(cfg, "reward.refine.bad.min_shorten_chars", default=20) or 20
	)
	min_shorten_ratio = float(
		OmegaConf.select(cfg, "reward.refine.bad.min_shorten_ratio", default=0.05)
	)

	# Attempt selection knobs (only used for refine multi-attempt aggregation)
	# New name (more explicit): reward.refine.attempt_selection.*
	# Backward compatible: reward.refine.schemeA.*
	def _select_attempt_selection(path_new: str, path_old: str, *, default: Any) -> Any:
		v = OmegaConf.select(cfg, path_new, default=None)
		if v is None:
			v = OmegaConf.select(cfg, path_old, default=default)
		return v

	attempt_selection_enable = bool(
		_select_attempt_selection(
			"reward.refine.attempt_selection.enable",
			"reward.refine.schemeA.enable",
			default=True,
		)
	)
	keep_final_attempt = bool(
		_select_attempt_selection(
			"reward.refine.attempt_selection.keep_final_attempt",
			"reward.refine.schemeA.keep_final_attempt",
			default=True,
		)
	)
	keep_improved_attempts = bool(
		_select_attempt_selection(
			"reward.refine.attempt_selection.keep_improved_attempts",
			"reward.refine.schemeA.keep_improved_attempts",
			default=True,
		)
	)
	max_improved_attempts = int(
		_select_attempt_selection(
			"reward.refine.attempt_selection.max_improved_attempts_per_sample",
			"reward.refine.schemeA.max_improved_attempts_per_sample",
			default=1,
		)
		or 1
	)
	max_improved_attempts = max(0, int(max_improved_attempts))

	# ===== multi-attempt refine aggregation + scheme-A selection =====
	if sample_type == "refine" and aggregate_refine:
		# pack per group
		group_packs: List[Dict[str, Any]] = []
		for attempt_i, d in attempt_datas:
			exec_dir_name_a = f"step-{current_epoch}-refine-a{int(attempt_i)}"
			exec_root_a = os.path.abspath(
				os.path.join("..", project_name, "execute", exec_dir_name_a, pde_type)
			)
			for _gid, item in enumerate(d):
				prompt = str(item.get("prompt", ""))
				full_outputs = item.get("full_output", []) or []
				metrics_list = item.get("pde_metrics", []) or []
				if not full_outputs:
					continue

				refine_meta = item.get("refine_meta") if isinstance(item, dict) else None
				origin_is_bad = True
				sample_key = None
				if isinstance(refine_meta, dict):
					origin_is_bad = bool(refine_meta.get("is_bad", True))
					sample_key = refine_meta.get("sample_key")
					if not sample_key:
						try:
							os0 = int(refine_meta.get("origin_step"))
							or0 = int(refine_meta.get("origin_round"))
							op0 = str(refine_meta.get("pde_type") or pde_type)
							sample_key = f"{os0}:{op0}:{or0}"
						except Exception:
							sample_key = None
				if not sample_key:
					sample_key = f"unknown:{int(attempt_i)}:{id(item)}"

				scores: List[float] = []
				entries_tmp: List[Dict[str, Any]] = []
				group_improved = False
				for j in range(len(full_outputs)):
					metrics = metrics_list[j] if j < len(metrics_list) else None
					partial_improved = False
					if bool(origin_is_bad) and isinstance(metrics, dict):
						round_val = metrics.get("round")
						try:
							round_i = int(round_val)
						except Exception:
							round_i = None
						if round_i is not None:
							sub_dir = os.path.join(exec_root_a, str(round_i))
							origin_err = os.path.join(sub_dir, "origin_errors.txt")
							new_err = os.path.join(sub_dir, "errors.txt")
							if os.path.exists(origin_err) and os.path.exists(new_err):
								partial_improved = _is_partial_improved(
									origin_errors_path=origin_err,
									new_errors_path=new_err,
									allowed_new_non_syntax_errors=allowed_new_non_syntax_errors,
									min_shorten_chars=min_shorten_chars,
									min_shorten_ratio=min_shorten_ratio,
								)
								group_improved = group_improved or bool(partial_improved)

					score, _is_bad2, _should_add = compute_refine_response_score(
						metrics,
						nrmse_threshold=nrmse_threshold,
						origin_is_bad=bool(origin_is_bad),
						partial_improved=bool(partial_improved),
						ok_high_nrmse_reward_when_good=ok_high_nrmse_reward_when_good,
						partial_improve_reward_when_bad=partial_improve_reward_when_bad,
					)
					scores.append(float(score))

				rewards = compute_group_rewards_for_grpo(scores)
				for j, response in enumerate(full_outputs):
					entry: Dict[str, Any] = {
						"prompt": prompt,
						"response": str(response),
						"reward": float(rewards[j]) if j < len(rewards) else 0.0,
						"sample_type": "refine",
						"group_id": -1,  # fill later
						"index_in_group": int(j),
						"refine_attempt": int(attempt_i),
						"sample_key": str(sample_key),
					}
					entries_tmp.append(entry)

				group_packs.append(
					{
						"sample_key": str(sample_key),
						"attempt": int(attempt_i),
						"origin_is_bad": bool(origin_is_bad),
						"group_improved": bool(group_improved),
						"entries": entries_tmp,
					}
				)

		# select packs per sample_key
		packs_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
		for p in group_packs:
			packs_by_key[str(p.get("sample_key"))].append(p)

		selected_packs: List[Dict[str, Any]] = []
		for _k, packs in packs_by_key.items():
			packs.sort(key=lambda x: int(x.get("attempt", 0)))
			if not packs:
				continue

			if not attempt_selection_enable:
				selected = list(packs)
			else:
				selected = []
				final_pack = packs[-1]
				# intermediate improved packs
				if keep_improved_attempts and max_improved_attempts > 0:
					cnt = 0
					for p in packs[:-1]:
						if not bool(p.get("origin_is_bad", True)):
							continue
						if bool(p.get("group_improved", False)):
							selected.append(p)
							cnt += 1
							if cnt >= max_improved_attempts:
								break
				# final pack
				if keep_final_attempt:
					selected.append(final_pack)
				else:
					# if not keeping final, still keep a fallback pack to avoid empty selection
					if not selected:
						selected.append(final_pack)

			# de-dup by attempt id
			seen_attempts: set[int] = set()
			for p in selected:
				a = int(p.get("attempt", 0))
				if a in seen_attempts:
					continue
				seen_attempts.add(a)
				selected_packs.append(p)

		# assign new group_id sequentially
		gid = 0
		for pack in sorted(selected_packs, key=lambda x: (str(x.get("sample_key")), int(x.get("attempt", 0)))):
			entries = pack.get("entries") or []
			if not isinstance(entries, list) or not entries:
				continue
			for j, e in enumerate(entries):
				if isinstance(e, dict):
					e["group_id"] = int(gid)
					e["index_in_group"] = int(j)
					final_data.append(e)
			gid += 1

		cprint(
			f"[pde_rl_reward] refine multi-attempt: packs={len(group_packs)} selected={len(selected_packs)} final_entries={len(final_data)}",
			"green",
		)
	else:

		# ===== single pass (scratch or single-attempt refine) =====
		# 根据 file_name 读取的数据
		for group_id, item in enumerate(data):
			# list 中每个 item 是一个 group(task) 的结果
			prompt = str(item.get("prompt", ""))
			full_outputs = item.get("full_output", []) or []
			metrics_list = item.get("pde_metrics", []) or []
			if not full_outputs:
				continue

			# 对当前 group 内的每个 response 单独计算 reward, 并决定是否加入 refine buffer
			scores: List[float] = []
			is_bad_flags: List[bool] = []
			buffer_indices: List[int] = []
			refine_meta = item.get("refine_meta") if isinstance(item, dict) else None
			origin_is_bad = True
			if sample_type == "refine" and isinstance(refine_meta, dict):
				origin_is_bad = bool(refine_meta.get("is_bad", True))
			for j in range(len(full_outputs)):
				metrics = metrics_list[j] if j < len(metrics_list) else None
				if sample_type == "scratch":
					score, is_bad, should_add_to_buffer = compute_single_response_score(metrics, nrmse_threshold)
				else:
					# refine：可选的“部分改进”奖励（仅针对 origin_is_bad=true 的样本）
					partial_improved = False
					if bool(origin_is_bad) and isinstance(metrics, dict):
						round_val = metrics.get("round")
						try:
							round_i = int(round_val)
						except Exception:
							round_i = None
						if round_i is not None:
							sub_dir = os.path.join(exec_root, str(round_i))
							origin_err = os.path.join(sub_dir, "origin_errors.txt")
							new_err = os.path.join(sub_dir, "errors.txt")
							if os.path.exists(origin_err) and os.path.exists(new_err):
								partial_improved = _is_partial_improved(
									origin_errors_path=origin_err,
									new_errors_path=new_err,
									allowed_new_non_syntax_errors=allowed_new_non_syntax_errors,
									min_shorten_chars=min_shorten_chars,
									min_shorten_ratio=min_shorten_ratio,
								)

					score, is_bad, should_add_to_buffer = compute_refine_response_score(
						metrics,
						nrmse_threshold=nrmse_threshold,
						origin_is_bad=bool(origin_is_bad),
						partial_improved=bool(partial_improved),
						ok_high_nrmse_reward_when_good=ok_high_nrmse_reward_when_good,
						partial_improve_reward_when_bad=partial_improve_reward_when_bad,
					)
				scores.append(float(score))
				is_bad_flags.append(bool(is_bad))
				if should_add_to_buffer:
					buffer_indices.append(j)

			# 组内 reward 归一化(GRPO)
			rewards = compute_group_rewards_for_grpo(scores)

			# 扁平样本（供当前 PPO 训练脚本使用）
			for j, response in enumerate(full_outputs):
				entry: Dict[str, Any] = {
					"prompt": prompt,
					"response": str(response),
					"reward": float(rewards[j]) if j < len(rewards) else 0.0,
					"sample_type": sample_type,
					"group_id": int(group_id),
					"index_in_group": int(j),
				}
				final_data.append(entry)

			# 从当前 group 中抽取需要加入 refine buffer 的候选
			# 新约定：buffer 只存“原子化样本指针”（origin_step/origin_round/is_bad/+pde_type），
			# 不存 prompt/response/reward/metrics/group_id 等。
			if buffer_indices:
				buffer_indices_sorted = sorted(set(int(x) for x in buffer_indices))
				for j in buffer_indices_sorted:
					m = metrics_list[j] if j < len(metrics_list) else None
					m_dict = m if isinstance(m, dict) else {}
					round_val = m_dict.get("round")
					try:
						round_i = int(round_val)
					except Exception:
						# 无法定位 execute/step-{origin_step}/{origin_round} 时直接跳过
						continue
					refine_records.append(
						{
							"origin_step": int(current_epoch),
							"origin_round": int(round_i),
							"is_bad": bool(is_bad_flags[j]) if j < len(is_bad_flags) else True,
							"pde_type": pde_type,
						}
					)

	# 写训练数据（train/rl_pde_dream.py 读取 ./{project}/temp_data/{optimization_data}.json）
	opt_name = str(cfg.dataset.optimization_data)
	base_out_path = f"../{project_name}/temp_data/{opt_name}.json"
	scratch_step_path = f"../{project_name}/temp_data/{opt_name}_scratch_step{current_epoch}.json"
	refine_step_path = f"../{project_name}/temp_data/{opt_name}_refine_step{current_epoch}.json"

	phase_out_path = refine_step_path if sample_type == "refine" else scratch_step_path
	os.makedirs(os.path.dirname(phase_out_path), exist_ok=True)
	with open(phase_out_path, "w", encoding="utf-8") as f:
		json.dump(final_data, f, indent=2, ensure_ascii=False)

	merged: List[Dict[str, Any]] = []
	if sample_type == "scratch":
		merged = list(final_data)
	else:
		try:
			with open(scratch_step_path, "r", encoding="utf-8") as f_in:
				scratch_data = json.load(f_in)
			if isinstance(scratch_data, list):
				merged.extend(scratch_data)
		except Exception:
			pass
		merged.extend(final_data)

	os.makedirs(os.path.dirname(base_out_path), exist_ok=True)
	with open(base_out_path, "w", encoding="utf-8") as f:
		json.dump(merged, f, indent=2, ensure_ascii=False)

	cprint(
		f"[pde_rl_reward] wrote phase data: {phase_out_path} (n={len(final_data)}); "
		f"merged -> {base_out_path} (n={len(merged)})",
		"green",
	)

	# ===== 更新 refine buffer（默认仅在 train_scratch 阶段写入，避免 refine 回灌导致 buffer 膨胀） =====
	max_size = int(getattr(getattr(cfg, "refine_buffer", {}), "max_size", 0) or 0)
	if phase in {"train_scratch", "train", ""}:
		if refine_records and max_size > 0:
			try:
				append_records(project_name, refine_records, max_size=max_size)
			except Exception as e:
				cprint(f"[pde_rl_reward] failed to append refine records: {e}", "red")
		else:
			cprint("[pde_rl_reward] no refine records to append or max_size <= 0", "yellow")
	else:
		cprint(f"[pde_rl_reward] skip appending refine buffer for phase={phase!r}", "yellow")
	##############################
	##############################


if __name__ == "__main__":
	main()
