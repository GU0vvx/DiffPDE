# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for the PDE reinforcement-learning controller.
import os
import sys
import subprocess
import uuid
import math
import random
from termcolor import cprint
import json
from pathlib import Path
import wandb

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    import os
    # 1) 读取 CLI 与基础 YAML 配置
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)

    # 使用一个临时合并（基础 + CLI）来判定 data_type 与 pde_type，
    # 但实际合并顺序将采用 base -> pde -> cli（保证 CLI 覆盖权最高）。
    tmp_conf = OmegaConf.merge(yaml_conf, cli_conf)
    data_type = OmegaConf.select(tmp_conf, "dataset.data_type", default=None)
    pde_type  = OmegaConf.select(tmp_conf, "dataset.pde_type",  default=None)

    # 以基础配置为起点
    conf = yaml_conf

    # 2) 若为 PDE 任务，则追加合并对应 pde_type 的子配置（合并到 pde: 节点下）
    if data_type == "pde":
        if not pde_type:
            raise ValueError("dataset.pde_type is required when dataset.data_type == 'pde'")

        # 以传入的基础配置文件所在目录为根，定位到 configs/pde_type/<pde_type>.yaml
        base_conf_dir = os.path.dirname(os.path.abspath(str(cli_conf.config)))
        pde_conf_path = os.path.join(base_conf_dir, "pde_type", f"{pde_type}.yaml")
        if not os.path.exists(pde_conf_path):
            raise FileNotFoundError(f"PDE config not found for type '{pde_type}': {pde_conf_path}")

        pde_conf = OmegaConf.load(pde_conf_path)
        # 将子配置挂载到 pde 节点，便于使用 ${pde.*} 的插值
        conf = OmegaConf.merge(conf, {"pde": pde_conf})

    # 3) 最后合并 CLI，确保命令行覆盖最高优先级
    conf = OmegaConf.merge(conf, cli_conf)
    return conf


def _forwarded_cli_args(parent_args: list[str]) -> list[str]:
	"""Forward the original config path and OmegaConf overrides to children."""
	forwarded: list[str] = []
	for arg in parent_args:
		if arg.startswith("config="):
			config_path = arg.split("=", 1)[1]
			if not config_path:
				raise ValueError("The config= CLI argument must not be empty.")
			config_path = str(Path(config_path).expanduser().resolve())
			forwarded.append(f"config={config_path}")
		else:
			forwarded.append(arg)
	return forwarded

if __name__ == "__main__":
	###############################
	###############################
	# 读取 Config 并在终端打印
	parent_cli_args = list(sys.argv[1:])
	child_cli_args = _forwarded_cli_args(parent_cli_args)
	config = get_config()

	os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.assigned_gpu))
	os.environ.setdefault("NCCL_P2P_LEVEL", os.environ.get("NCCL_P2P_LEVEL", "NVL"))
	os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

	cprint("==== Loaded PDE-RL config ====", color="green")
	try:
		cfg_yaml = OmegaConf.to_yaml(config, resolve=True)
	except TypeError:
		cfg_yaml = OmegaConf.to_yaml(config)
	print(cfg_yaml)
	###############################
	###############################





	###############################
	###############################
	# wandb offline 相关设置初始化

	project_name = str(config.experiment.project)
	# 让整个 RL 流程复用同一个 wandb run（每个 step 的训练脚本会 resume 该 run）
	wandb_run_id = OmegaConf.select(config, "wandb.run_id", default=None)
	if not wandb_run_id:
		wandb_run_id = uuid.uuid4().hex
		try:
			config.wandb.run_id = wandb_run_id
		except Exception:
			pass
	cprint(f"[PDE-RL] wandb.run_id={wandb_run_id}", color="green")

	# ===== 单进程 wandb 聚合（适配 WANDB_MODE=offline） =====
	# wandb 在 offline 模式下会忽略 resume（见日志警告），因此不要依赖“子进程 resume”。
	# 这里在父进程只 init 一次 run；子进程训练将写 jsonl 指标，由父进程回灌。
	_wandb_enabled = str(os.environ.get("WANDB_MODE", "")).strip().lower() != "disabled"
	if _wandb_enabled:
		try:
			cfg_dict = OmegaConf.to_container(config, resolve=True)
		except Exception:
			cfg_dict = None
		wandb.init(
			project=str(project_name),
			name=str(project_name),
			id=str(wandb_run_id),
			config=cfg_dict,
		)

	_logged_steps: set[int] = set()

	def _replay_train_metrics(step_idx: int) -> None:
		"""将子进程训练写出的 jsonl 指标回灌到父进程 wandb run。"""
		if not _wandb_enabled:
			return
		metrics_path = Path(project_name) / "temp_data" / f"train_metrics_step{int(step_idx)}.jsonl"
		if not metrics_path.exists():
			cprint(f"[PDE-RL] metrics jsonl not found: {metrics_path}", color="yellow")
			return
		with metrics_path.open("r", encoding="utf-8", errors="surrogatepass") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					rec = json.loads(line)
					wandb_step = int(rec.get("step"))
					metrics = rec.get("metrics", {})
				except Exception:
					continue
				if wandb_step in _logged_steps:
					continue
				if isinstance(metrics, dict) and metrics:
					wandb.log(metrics, step=wandb_step)
					_logged_steps.add(wandb_step)



	###############################
	###############################
	

	
	###############################
	###############################
	# 导入模型
	model_base = getattr(config, "model_base", None)
	if model_base is None and hasattr(config, "model"):
		model_base = getattr(config.model, "model_base", None)

	from omegaconf import MISSING

	if OmegaConf.select(config, "model.value_base_model", default=MISSING) is not MISSING:
		have_value_model = True
	else:
		have_value_model = False
	###############################
	###############################




	###############################
	###############################
	# rl rollout 流程相关函数定义

	def _run(cmd: list[str], *, cwd: str) -> None:
		subprocess.run(cmd, cwd=cwd, check=True)

	def _uniform_split(total: int, k: int, offset: int = 0) -> dict[int, int]:
		"""将 total 均匀分到 k 份；余数按 offset 轮转分配，避免固定顺序偏置。"""
		total = max(0, int(total))
		k = max(1, int(k))
		base, rem = divmod(total, k)
		out = {i: base for i in range(k)}
		for r in range(rem):
			out[(offset + r) % k] += 1
		return out

	def _load_json_list(path: Path) -> list:
		try:
			with path.open("r", encoding="utf-8", errors="surrogatepass") as f:
				data = json.load(f)
			return data if isinstance(data, list) else []
		except Exception:
			return []

	def _write_json_list(path: Path, data: list) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8", errors="surrogatepass") as f:
			json.dump(data, f, indent=2, ensure_ascii=False)
	def begin_with(file_name: str) -> None:
		"""将给定文件清空并创建，用于结果日志初始化。"""
		os.makedirs(os.path.dirname(file_name), exist_ok=True)
		with open(file_name, "w") as f:
			f.write("")

	def rollout_scratch(step_idx: int, *, pde_type: str | None = None, num_tasks: int | None = None) -> None:
		"""Phase A: 从头生成 PDE solver 的 rollout（scratch）。"""

		script_name = "pde_rl_rollout.py"
		cmd = [
			sys.executable,
			script_name,
			*child_cli_args,
			f"experiment.current_epoch={step_idx}",
		]
		if pde_type:
			cmd.append(f"dataset.pde_type={pde_type}")
		if num_tasks is not None:
			cmd.append(f"rollout.num_task_per_step={int(num_tasks)}")
		_run(cmd, cwd="sample")


	def rollout_refine(
		step_idx: int,
		*,
		pde_type: str | None = None,
		num_samples: int | None = None,
		refine_attempt: int | None = None,
	) -> None:
		"""Phase B: 对已有错误/效果差的 solver 做 refine infilling 的 rollout。"""

		script_name = "pde_rl_rollout_refine.py"
		cmd = [
			sys.executable,
			script_name,
			*child_cli_args,
			f"experiment.current_epoch={step_idx}",
		]
		if pde_type:
			cmd.append(f"dataset.pde_type={pde_type}")
		if num_samples is not None:
			cmd.append(f"refine_rollout.num_samples_per_step={int(num_samples)}")
		# optional: multi-attempt refine
		if refine_attempt is not None:
			cmd.append(f"experiment.refine_attempt={int(refine_attempt)}")
		_run(cmd, cwd="sample")


	def execute_pde(step_idx: int, phase: str, *, pde_type: str | None = None, refine_attempt: int | None = None) -> None:
		"""在 PDE 仿真环境中执行生成的 solver 代码，收集数值结果 / 错误信息。"""

		cmd = [
			sys.executable,
			"pde_rl_execute.py",
			*child_cli_args,
			f"experiment.function={phase}",
			f"experiment.current_epoch={step_idx}",
		]
		if refine_attempt is not None:
			cmd.append(f"experiment.refine_attempt={int(refine_attempt)}")
		if pde_type:
			cmd.append(f"dataset.pde_type={pde_type}")
		_run(cmd, cwd="reward")


	def compute_reward(
		step_idx: int,
		phase: str,
		*,
		pde_type: str | None = None,
		optimization_data: str | None = None,
		refine_attempt: int | None = None,
	) -> None:
		"""根据 PDE 仿真结果计算 reward，并将 (prompt, response, reward, meta)
		写入 project/temp_data 下供训练脚本使用。
		"""

		cmd = [
			sys.executable,
			"pde_rl_reward.py",
			*child_cli_args,
			f"experiment.function={phase}",
			f"experiment.current_epoch={step_idx}",
		]
		if refine_attempt is not None:
			cmd.append(f"experiment.refine_attempt={int(refine_attempt)}")
		if pde_type:
			cmd.append(f"dataset.pde_type={pde_type}")
		if optimization_data:
			cmd.append(f"dataset.optimization_data={optimization_data}")
		_run(cmd, cwd="reward")


    ###############################
    ###############################



	###############################
	###############################
	# rl train 流程相关函数定义

	def _lr_for_step(step_idx: int) -> float:
		"""计算当前 RL step 的学习率（跨 step 退火）。

		说明：每个 RL step 都会重新 launch 一次训练脚本，所以这里做“跨进程/跨 step”的 lr 调度。
		"""
		base_lr = OmegaConf.select(config, "lr_anneal.base_lr", default=None)
		if base_lr is None:
			base_lr = OmegaConf.select(config, "optimizer.params.learning_rate", default=0.0)
		base_lr = float(base_lr)

		enable = bool(OmegaConf.select(config, "lr_anneal.enable", default=False))
		if not enable:
			return base_lr

		schedule = str(OmegaConf.select(config, "lr_anneal.schedule", default="cosine")).strip().lower()
		min_lr_scale = float(OmegaConf.select(config, "lr_anneal.min_lr_scale", default=0.0))
		min_lr_scale = max(0.0, min(1.0, min_lr_scale))

		total_step_cfg = int(getattr(config.experiment, "total_step", 1) or 1)
		denom = max(1, total_step_cfg - 1)
		progress = float(int(step_idx) - 1) / float(denom)  # 0..1
		progress = max(0.0, min(1.0, progress))

		if schedule in ("none", "constant", "const"):
			scale = 1.0
		elif schedule == "linear":
			scale = 1.0 - (1.0 - min_lr_scale) * progress
		elif schedule == "cosine":
			cos = 0.5 * (1.0 + math.cos(math.pi * progress))
			scale = min_lr_scale + (1.0 - min_lr_scale) * cos
		elif schedule == "exp":
			gamma = float(OmegaConf.select(config, "lr_anneal.exp_gamma", default=1.0))
			# step_idx 从 1 开始，因此衰减次数为 (step_idx-1)
			scale = gamma ** max(0, int(step_idx) - 1)
			scale = max(min_lr_scale, float(scale))
		else:
			raise ValueError(f"Unknown lr_anneal.schedule={schedule!r}")

		return base_lr * float(scale)

	# ===== 训练阶段（调用单独的分布式训练脚本） =====
	def train_policy(step_idx: int) -> None:
		"""在第 step_idx 轮 rollout 产生的数据上，做一次策略 RL 更新。"""

		if model_base == "dream":
			script_name = "rl_pde_dream.py"
		else:
			raise NotImplementedError(f"PDE-RL not implemented for model_base={model_base!r}")

		lr_this_step = _lr_for_step(step_idx)
		lr_cli = f"{float(lr_this_step):.10g}"
		cprint(f"[PDE-RL] lr(step={int(step_idx)}) = {lr_cli}", color="green")

		cmd = [
			"accelerate",
			"launch",
			"--num_machines",
			"1",
			"--machine_rank",
			"0",
			"--main_process_ip",
			"127.0.0.1",
			"--main_process_port",
			"8888",
			"--config_file",
			f"accelerate_configs/{config.experiment.deepspeed_file}.yaml",
			f"train/{script_name}",
			*child_cli_args,
			f"experiment.current_epoch={step_idx}",
			f"optimizer.params.learning_rate={lr_cli}",
			f"lr_scheduler.params.learning_rate={lr_cli}",
			f"wandb.run_id={wandb_run_id}",
			"wandb.resume=allow",
		]
		# 子进程禁用 wandb，避免 offline 模式下每次 launch 都新开一个 offline-run 目录
		child_env = os.environ.copy()
		child_env["WANDB_MODE"] = "disabled"
		subprocess.run(cmd, check=True, env=child_env)

	
    
    # ===== 可选：周期性 PDE 评估 =====
	def eval_current_policy(step_idx: int) -> None:
		"""使用当前 ckpt 对 PDE 任务做一次完整评估，记录数值精度等指标。"""

		cmd = [sys.executable, "pde_eval.py", *child_cli_args]
		subprocess.run(cmd, check=True)

    ###############################
    ###############################





	###############################
    ###############################
    # 主训练循环

	# ===== 顶层 RL 循环 =====

	i = config.experiment.current_epoch
	total_step = config.experiment.total_step
	eval_every = getattr(config.experiment, "eval_every", 0) or 0

	# 是否在 RL 中启用 refine 相位，沿用 rollout.enable_refine 语义
	rollout_cfg = getattr(config, "rollout", None)
	enable_refine_phase = False
	if rollout_cfg is not None:
		enable_refine_phase = bool(getattr(rollout_cfg, "enable_refine", False))

	while i <= total_step:
		cprint(f"[PDE-RL] ===== Step {i} / {total_step} =====", color="cyan")

		multi_enable = bool(OmegaConf.select(config, "dataset.multi_pde.enable", default=False))
		pde_types = OmegaConf.select(config, "dataset.multi_pde.pde_types", default=None)
		if not isinstance(pde_types, (list, ListConfig)):
			pde_types = None
		if not pde_types:
			pde_types = [str(getattr(config.dataset, "pde_type", "burgers"))]
		pde_types = [str(x) for x in list(pde_types)]

		# 基准优化数据名（训练脚本读取这个）
		base_opt_name = str(config.dataset.optimization_data)
		project_temp = Path(project_name) / "temp_data"
		combined_path = project_temp / f"{base_opt_name}.json"

		if not multi_enable or len(pde_types) <= 1:
			# 单一 pde 训练
			# from-scratch rollout + reward 计算
			rollout_scratch(i)
			execute_pde(i, phase="train_scratch")
			compute_reward(i, phase="train_scratch")

			# Phase B: 对不佳 / 报错解做 refine（若启用）
			if enable_refine_phase:
				multi_attempt_enable = bool(
					OmegaConf.select(config, "refine_rollout.multi_attempt.enable", default=False)
				)
				num_attempts = int(
					OmegaConf.select(config, "refine_rollout.multi_attempt.num_attempts", default=1)
					or 1
				)
				if multi_attempt_enable and num_attempts > 1:
					for a in range(1, int(num_attempts) + 1):
						rollout_refine(i, refine_attempt=int(a))
						execute_pde(i, phase="train_refine", refine_attempt=int(a))
					# aggregate in reward stage
					compute_reward(i, phase="train_refine", refine_attempt=0)
				else:
					rollout_refine(i)
					execute_pde(i, phase="train_refine")
					compute_reward(i, phase="train_refine")
		else:
			# ===== 多 PDE：逐类跑 rollout/execute/reward，但训练阶段统一（合并后训练一次） =====
			total_tasks_step = int(getattr(getattr(config, "rollout", {}), "num_task_per_step", 1) or 1)
			split = _uniform_split(total_tasks_step, len(pde_types), offset=(int(i) - 1) % len(pde_types))
			per_type_opt_names: list[str] = []

			# refine samples 也做均匀切分（如果启用 refine）
			total_refine_samples = int(
				OmegaConf.select(config, "refine_rollout.num_samples_per_step", default=0) or 0
			)
			ref_split = _uniform_split(
				total_refine_samples, len(pde_types), offset=(int(i) - 1) % len(pde_types)
			)

			cprint(
				f"[PDE-RL] multi_pde enabled: step_tasks={total_tasks_step} types={pde_types}",
				color="green",
			)

			for idx_t, pde_t in enumerate(pde_types):
				tasks_t = int(split.get(idx_t, 0))
				ref_samples_t = int(ref_split.get(idx_t, 0))
				if tasks_t <= 0 and (not enable_refine_phase or ref_samples_t <= 0):
					continue

				opt_name_t = f"{base_opt_name}__{pde_t}"
				per_type_opt_names.append(opt_name_t)

				if tasks_t > 0:
					rollout_scratch(i, pde_type=pde_t, num_tasks=tasks_t)
					execute_pde(i, phase="train_scratch", pde_type=pde_t)
					compute_reward(i, phase="train_scratch", pde_type=pde_t, optimization_data=opt_name_t)

				if enable_refine_phase and ref_samples_t > 0:
					multi_attempt_enable = bool(
						OmegaConf.select(config, "refine_rollout.multi_attempt.enable", default=False)
					)
					num_attempts = int(
						OmegaConf.select(config, "refine_rollout.multi_attempt.num_attempts", default=1)
						or 1
					)
					if multi_attempt_enable and num_attempts > 1:
						for a in range(1, int(num_attempts) + 1):
							# attempt>1 仍复用同一份 buffer 抽样（由 refine rollout 内部固定 seed 保证）
							rollout_refine(
								i,
								pde_type=pde_t,
								num_samples=ref_samples_t,
								refine_attempt=int(a),
							)
							execute_pde(i, phase="train_refine", pde_type=pde_t, refine_attempt=int(a))
						compute_reward(
							i,
							phase="train_refine",
							pde_type=pde_t,
							optimization_data=opt_name_t,
							refine_attempt=0,
						)
					else:
						rollout_refine(i, pde_type=pde_t, num_samples=ref_samples_t)
						execute_pde(i, phase="train_refine", pde_type=pde_t)
						compute_reward(i, phase="train_refine", pde_type=pde_t, optimization_data=opt_name_t)

			# 合并各 PDE 的训练数据到 base_opt_name，并打乱顺序避免“先后顺序过于清晰”
			merged_all: list = []
			for opt_name_t in per_type_opt_names:
				merged_all.extend(_load_json_list(project_temp / f"{opt_name_t}.json"))

			shuffle_enable = bool(OmegaConf.select(config, "dataset.multi_pde.shuffle_before_train", default=True))
			if shuffle_enable and merged_all:
				seed_cfg = OmegaConf.select(config, "dataset.multi_pde.seed", default=None)
				try:
					seed_base = int(seed_cfg) if seed_cfg is not None else 0
				except Exception:
					seed_base = 0
				seed_eff = int(seed_base) + 100000 * int(i)
				rng = random.Random(seed_eff)
				rng.shuffle(merged_all)

			_write_json_list(combined_path, merged_all)
			cprint(
				f"[PDE-RL] merged multi_pde train data -> {combined_path} (n={len(merged_all)})",
				color="green",
			)

		# 基于本轮 scratch + refine 的数据做一次策略更新（无论单 PDE 还是多 PDE，都只训练一次）
		train_policy(i)
		_replay_train_metrics(i)

		# 周期性 PDE 评估（可选）
		if eval_every > 0 and (i % eval_every == 0):
			eval_current_policy(i)

		i += 1

	if _wandb_enabled:
		try:
			wandb.finish()
		except Exception:
			pass
