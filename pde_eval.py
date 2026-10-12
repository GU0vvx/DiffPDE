# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE evaluation control.
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from termcolor import cprint
from typing import Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


def _coerce_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

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


if __name__ == "__main__":
    # Capture original CLI args so we can forward overrides to child scripts.
    # This is important when running multiple PDE types via a bash loop.
    parent_cli_args = sys.argv[1:]

    config = get_config()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.assigned_gpu))
    
    # Print the merged/effective config to the terminal at startup
    cprint("==== Loaded config ====", color="green")
    try:
        # Prefer resolving interpolations if supported by current OmegaConf
        cfg_yaml = OmegaConf.to_yaml(config, resolve=True)
    except TypeError:
        # Fallback for older OmegaConf versions without the 'resolve' arg
        cfg_yaml = OmegaConf.to_yaml(config)
    print(cfg_yaml)

    project_name = config.experiment.project
    eval_type = config.dataset.data_type

    good_nrmse_threshold = _coerce_float(
        OmegaConf.select(config, "execute.good_nrmse_threshold", default=None)
    )
    if good_nrmse_threshold is None:
        good_nrmse_threshold = _coerce_float(
            OmegaConf.select(config, "execute.nrmse_threshold", default=None)
        )
    def _child_cli_args(child_cwd: str):
        """Forward parent's CLI args to child scripts.

        - Keeps all CLI overrides (e.g. dataset.pde_type, experiment.project, rollout.*)
        - Rewrites `config=...` so it is valid relative to the child cwd.
        """
        rewritten = []
        for arg in parent_cli_args:
            if arg.startswith("config="):
                cfg_path = arg.split("=", 1)[1]
                if cfg_path and not os.path.isabs(cfg_path):
                    # Child scripts run from subfolders like sample/ or reward/.
                    cfg_path = os.path.normpath(os.path.join("..", cfg_path))
                rewritten.append(f"config={cfg_path}")
            else:
                rewritten.append(arg)
        return rewritten

    def _read_json_if_exists(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _compute_execute_pass_metrics(
        proj_name: str,
        nrmse_threshold: Optional[float] = None,
        nrmse_col: str = "nRMSE",
    ):
        """Read execute/test_results.csv and compute pass rate.

        Passing = error_type is empty or 'none'.
        """
        import csv

        exec_root = os.path.abspath(os.path.join(proj_name, "execute"))
        csv_path = os.path.join(exec_root, "test_results.csv")
        metrics = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "good": 0,
            "good_rate": 0.0,
            "good_passed_rate": 0.0,
            "nrmse_threshold": nrmse_threshold,
            "nrmse_col": nrmse_col,
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
                        if nrmse_threshold is not None:
                            raw = row.get(nrmse_col)
                            nrmse_val = _coerce_float(raw)
                            if nrmse_val is not None and nrmse_val < nrmse_threshold:
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

    def _write_json(path: str, obj) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base):
        cprint(f"This is sampling.", color = "green")
        if model_base == "dream":
            subprocess.run(
                [sys.executable, "pde_sample_dream.py", *_child_cli_args("sample")],
                cwd='sample',
                check=True,
            )
        elif model_base == "llada":
            subprocess.run(
                [sys.executable, "pde_sample_llada.py", *_child_cli_args("sample")],
                cwd='sample',
                check=True,
            )
        elif model_base == "trado" or model_base == "sdar":
            subprocess.run( # 子进程
                [sys.executable, "pde_sample_sdar.py", *_child_cli_args("sample")],
                cwd='sample', # 切换路径
                check=True,
            )

    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        subprocess.run(
            [sys.executable, "reward.py", *_child_cli_args("reward")],
            cwd='reward',
            check=True,
        )
    
    def execute():
        cprint(f"This is the execution.", color = "green")
        subprocess.run(
            [sys.executable, "pde_execute.py", *_child_cli_args("reward")],
            cwd='reward',
            check=True,
        )

    def refine(model_base):
        """Run PDE refine script once if enabled in config.

        按当前 YAML 的结构优先读取 rollout.enable_refine / rollout.refine_max_steps，
        若不存在 rollout 节点，则回退到顶层 enable_refine 以兼容旧配置。"""
        # 1) 根据 config.rollout 或顶层 enable_refine 判断是否开启 refine
        enabled = False
        max_steps = 0
        rollout_cfg = getattr(config, "rollout", None)
        if rollout_cfg is not None:
            enabled = bool(getattr(rollout_cfg, "enable_refine", False))
            max_steps = int(getattr(rollout_cfg, "refine_max_steps", 0) or 0)
        else:
            enabled = bool(getattr(config, "enable_refine", False))
            max_steps = int(getattr(config, "refine_max_steps", 0) or 0)

        if not enabled:
            cprint("[refine] Refine is disabled in config, skip.", "yellow")
            return
        if max_steps <= 0:
            cprint("[refine] refine_max_steps <= 0, skip refine.", "yellow")
            return

        # 2) Only Dream model is currently supported
        if model_base != "dream":
            cprint("[refine] Refine is only supported for Dream model, skip.", "yellow")
            return

        # 3) Launch refine script once
        refine_cmd = [sys.executable, "pde_refine_dream.py", *_child_cli_args("sample")]
        cprint("[refine] Starting PDE refine pipeline...", "cyan")
        try:
            subprocess.run(
                refine_cmd,
                cwd='sample',
                check=True,
            )
            cprint("[refine] PDE refine pipeline finished.", "green")
        except subprocess.CalledProcessError as err:
            raise RuntimeError("[refine] Refine script failed, please check logs.") from err
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)

    # One run summary file (overwritten each run)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_metrics_path = os.path.join(project_name, "results", "run_metrics.json")
    run_metrics = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": project_name,
        "model": getattr(config, "model", None),
        "model_base": getattr(config, "model_base", None),
        "dataset": getattr(config.dataset, "eval_dataset", None),
        "data_type": getattr(config.dataset, "data_type", None),
        "pde_type": getattr(config.dataset, "pde_type", None),
        "num_node": getattr(config.experiment, "num_node", None),
        "node_index": getattr(config.experiment, "node_index", None),
        "stages": {},
    }

    # 1) 第一次采样与执行
    t_sample = time.time()
    sample(config.model_base)
    run_metrics["stages"]["sample_wall_time_sec"] = float(time.time() - t_sample)

    # sample metrics from pde_sample_dream.py
    gen_metrics_path = os.path.join(project_name, "results", "generation_metrics.json")
    gen_metrics = _read_json_if_exists(gen_metrics_path)
    if gen_metrics is not None:
        run_metrics["stages"]["sample"] = gen_metrics

    t_execute = time.time()
    execute()
    run_metrics["stages"]["execute_round0_wall_time_sec"] = float(time.time() - t_execute)
    run_metrics["stages"]["execute_round0"] = _compute_execute_pass_metrics(
        project_name, nrmse_threshold=good_nrmse_threshold
    )

    # 2) 若在配置中开启 refine, 则调用 refine 函数，
    #    多轮归档/掩码/重解码/重新评测的具体流程由 pde_refine_dream.py 控制。
    t_refine = time.time()
    refine(config.model_base)
    run_metrics["stages"]["refine_wall_time_sec"] = float(time.time() - t_refine)

    refine_metrics_path = os.path.join(project_name, "results", "refine_metrics.json")
    refine_metrics = _read_json_if_exists(refine_metrics_path)
    if refine_metrics is not None:
        run_metrics["stages"]["refine"] = refine_metrics

    # Final execute pass-rate after refine (if refine happened)
    run_metrics["stages"]["execute_final"] = _compute_execute_pass_metrics(
        project_name, nrmse_threshold=good_nrmse_threshold
    )

    _write_json(run_metrics_path, run_metrics)
    cprint(f"[metrics] Run summary written to {run_metrics_path}", "cyan")
    




