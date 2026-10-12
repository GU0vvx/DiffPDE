# Adapted from Gen-Verse/dLLM-RL:
# https://github.com/Gen-Verse/dLLM-RL
# Based on commit bca67b12f700bec4a3fb2dfe50adf04eb62cad5a.
# Upstream portions licensed under the Apache License, Version 2.0.
# Modified by the DiffPDE authors for PDE solver evaluation.
import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import shutil
import concurrent.futures as cf
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf
import subprocess
import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pde.paths import get_evaluator_path, get_evaluator_python


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

import textwrap


# function for pde-execute



def main():
    cfg          = get_config()
    evaluator_python = get_evaluator_python(cfg)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.assigned_gpu))

    project_name = cfg.experiment.project
    outputs_name = "eval-" + cfg.model.replace("/", ".") + "-" + cfg.dataset.eval_dataset


    num_node = cfg.experiment.num_node
    node_index = cfg.experiment.node_index

    if num_node > 1:
        file_name    = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:# 绝对路径查找结果
        file_name    = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)


    if cfg.dataset.data_type == "pde":
        # 1) 在 ../{project_name} 下创建 execute 目录
        exec_root = os.path.abspath(os.path.join("..", project_name, "execute"))
        os.makedirs(exec_root, exist_ok=True)

        # 1.1) 在 execute 根目录下重建一个结果汇总表，用于记录各子目录评测结果
        result_sheet_path = os.path.join(exec_root, "test_results.csv")
        print("Generating PDE result sheet at", result_sheet_path)
        with open(result_sheet_path, "w", encoding="utf-8", errors="surrogatepass") as f:
            f.write("round,nRMSE,elapsed_time,convergence_rate,num_trial,error_type,error_line\n")

        # 2) 将 JSON 中每条样本的 full_output / extracted_output 逐一写入编号子目录
        # 目录命名从 0 开始依次递增：{exec_root}/0, {exec_root}/1, ...
        counter = 0
        for item in data:
            full_outputs = item.get("full_output", []) or []
            extracted_outputs = item.get("extracted_output", []) or []
            m = min(len(full_outputs), len(extracted_outputs))

            for j in range(m):
                # 子目录
                sub_dir = os.path.join(exec_root, str(counter))
                os.makedirs(sub_dir, exist_ok=True)

                # 写入原始回复
                responses_path = os.path.join(sub_dir, "responses.txt")
                with open(responses_path, "w", encoding="utf-8", errors="surrogatepass") as f:
                    f.write(str(full_outputs[j]))

                # 写入对应 prompt
                prompt_path = os.path.join(sub_dir, "prompt.txt")
                prompt_text = str(item.get("prompt", ""))
                with open(prompt_path, "w", encoding="utf-8", errors="surrogatepass") as f:
                    f.write(prompt_text)

                # 写入提取出的代码为 solver.py
                code_text = str(extracted_outputs[j])
                placeholder_msg = "We can not extract the code in the output."
                if code_text.strip() == placeholder_msg:
                    cprint(f"Skip solver.py for index {counter} (no code extracted)", color="yellow")
                else:
                    solver_path = os.path.join(sub_dir, "solver.py")
                    with open(solver_path, "w", encoding="utf-8", errors="surrogatepass") as f:
                        f.write(code_text)

                # 复制对应 pde_type 的 evaluator.py 到子目录
                evaluator_src = get_evaluator_path(str(cfg.dataset.pde_type), cfg)
                evaluator_dst = os.path.join(sub_dir, "evaluator.py")
                shutil.copyfile(evaluator_src, evaluator_dst)

                # 预先创建 output.txt 和 errors.txt 文件
                output_path = os.path.join(sub_dir, "output.txt")
                errors_path = os.path.join(sub_dir, "errors.txt")
                open(output_path, "w", encoding="utf-8", errors="surrogatepass").close()
                open(errors_path, "w", encoding="utf-8", errors="surrogatepass").close()

                counter += 1

        # 评测数据集路径构造
        eval_dataset = os.path.join(str(cfg.dataset.root_dataset_folder), str(cfg.pde.dataset_for_eval))

        def cleanup_gpu_processes(gpu_id):
            """
            Clean up any orphaned processes still using the specified GPU
            """
            try:
                # Find all processes using this GPU
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(gpu_id),
                    ],
                    capture_output=True,
                    text=True
                )
                
                # Extract process IDs
                pids = result.stdout.strip().split('\n')
                
                # Kill each process
                for pid in pids:
                    if pid and pid.isdigit():
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"Killed GPU process with PID {pid}")
                        except ProcessLookupError:
                            pass  # Process already terminated
            except Exception as e:
                print(f"Error during GPU cleanup: {e}")

        def run_evaluator_in_subdir(sub_dir: str, eval_dataset: str, gpu_id: str):
            '''


            sub_dir: evaluator.py 所在子目录路径
            eval_dataset: 评测数据集路径
            '''
            job_out = open(os.path.join(sub_dir, f'output.txt'), 'w')
            job_err = open(os.path.join(sub_dir, f'errors.txt'), 'w')

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
            
            # Append PDE-specific hyperparameters to the command
            pde_name = cfg.dataset.pde_type

            if pde_name == 'advection':
                cmd.extend(["--beta", str(cfg.pde.beta)])
            elif pde_name == 'burgers':
                cmd.extend(["--nu", str(cfg.pde.nu)])
            elif pde_name == 'reacdiff1d':
                cmd.extend(["--nu", str(cfg.pde.nu), "--rho", str(cfg.pde.rho)])
            elif pde_name == 'cns1d':
                cmd.extend(["--eta", str(cfg.pde.eta)])
            elif pde_name in ['darcy', 'ins2d']:
                pass  # No hyperparameters for these two
            else:
                raise ValueError(f'PDE {pde_name} not recognized')
            
            
            # 为子进程设置独立环境, 指定使用的 GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            try:
                # Start process using Popen
                process = subprocess.Popen(
                    cmd,
                    stdout=job_out,
                    stderr=job_err,
                    text=True,
                    cwd=sub_dir,
                    preexec_fn=os.setsid,  # Create a new process group
                    env=env,
                )
                
                # Wait for the process with timeout
                exit_code = process.wait(timeout=cfg.pde.timeout)
                stderr = None
                status = "completed"
        
            except subprocess.TimeoutExpired:
                # Kill the entire process group on timeout
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait a moment for graceful termination
                time.sleep(2)
                
                # If still running, use SIGKILL
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                
                # Clean up any GPU processes that might still be running
                cleanup_gpu_processes(cfg.assigned_gpu)
                
                job_out.write(f"Process exceeded the {cfg.pde.timeout}-second timeout limit.\n")
                job_err.write(f"Process exceeded the {cfg.pde.timeout}-second timeout limit.\n")
                exit_code = -1
                stderr = "TimeoutExpired: Process exceeded the timeout limit."
                status = "timeout"
                
            finally:
                # Always close the files
                job_out.close()
                job_err.close()

            return {
                "exit_code": exit_code,
                "stderr": stderr,
                "status": status
            }

        # 3) 遍历 exec_root 下的每个子目录，运行 evaluator 并将结果写入汇总 CSV
        sub_dirs = [
            os.path.join(exec_root, d)
            for d in os.listdir(exec_root)
            if os.path.isdir(os.path.join(exec_root, d)) and d.isdigit()
        ]
        sub_dirs.sort(key=lambda p: int(os.path.basename(p)))

        def get_last_line(output_file: str) -> str | None:
            if not os.path.exists(output_file):
                return None
            with open(output_file, "r", encoding="utf-8", errors="surrogatepass") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if line.strip():  # skip trailing blank lines
                    return line
            return None

        def parse_results_from_line(line: str):
            import re
            rel_err = float(re.search(r"nRMSE: (.*?)\t", line).group(1))
            elapsed = float(re.search(r"Time: (.*?)s", line).group(1))
            avg_rate = float(re.search(r"Average convergence rate: (.*?)\t", line).group(1))
            return rel_err, elapsed, avg_rate

        def extract_error_info(errors_path: str) -> tuple[str, str]:
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
        # 并行执行 evaluator, 自适应在多 GPU 上调度
        # cfg.assigned_gpu 形如 "0,1,2,3" 或 "0"
        gpu_list = [g.strip() for g in str(cfg.assigned_gpu).split(",") if g.strip() != ""]
        if not gpu_list:
            gpu_list = ["0"]
        num_gpus = len(gpu_list)

        # 每个 GPU 上允许同时运行的 evaluator 数量, 可在配置中覆盖
        total_workers = max(1, num_gpus * cfg.max_workers_per_gpu)

        def eval_one_subdir(sub_dir: str, eval_dataset: str, gpu_id: str):
            round_idx = int(os.path.basename(sub_dir))

            # 若没有 solver.py，直接返回 failed
            solver_path = os.path.join(sub_dir, "solver.py")
            if not os.path.exists(solver_path):
                # 若没有成功提取出代码块, 记为 none, 保留子目录以便查看 responses/prompt
                return round_idx, "none", "none", "none", "no_solver", ""

            result = run_evaluator_in_subdir(sub_dir, eval_dataset, gpu_id)
            errors_path = os.path.join(sub_dir, "errors.txt")
            error_type = "none"
            error_line = ""

            if result["exit_code"] != 0:
                err_type, err_line = extract_error_info(errors_path)
                if result.get("status") == "timeout" and err_type == "none":
                    err_type = "timeout"
                return round_idx, "failed", "failed", "failed", err_type, err_line

            # 解析 output.txt 最后一行的 nRMSE/Time/Rate
            output_file = os.path.join(sub_dir, "output.txt")
            last_line = get_last_line(output_file)
            if not last_line:
                err_type, err_line = extract_error_info(errors_path)
                return round_idx, "failed", "failed", "failed", err_type, err_line

            try:
                rel_err, elapsed, avg_rate = parse_results_from_line(last_line)
                return round_idx, rel_err, elapsed, avg_rate, error_type, error_line
            except Exception:
                # 正则解析失败也记为 failed
                err_type, err_line = extract_error_info(errors_path)
                if err_type == "none":
                    err_type = "parse_failed"
                return round_idx, "failed", "failed", "failed", err_type, err_line

        tasks = []
        for i, sub_dir in enumerate(sub_dirs):
            gpu_id = gpu_list[i % num_gpus]
            tasks.append((sub_dir, eval_dataset, gpu_id))

        results = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            future_to_round = {
                executor.submit(eval_one_subdir, sub_dir, eval_dataset, gpu_id): int(os.path.basename(sub_dir))
                for (sub_dir, eval_dataset, gpu_id) in tasks
            }

            for fut in as_completed(future_to_round):
                try:
                    round_idx, rel_err, elapsed, avg_rate, err_type, err_line = fut.result()
                except Exception:
                    round_idx = future_to_round[fut]
                    results.append((round_idx, "failed", "failed", "failed", "exception", ""))
                else:
                    results.append((round_idx, rel_err, elapsed, avg_rate, err_type, err_line))

        # 按 round_idx 排序后写入 CSV, 保持顺序一致
        results.sort(key=lambda x: x[0])
        with open(result_sheet_path, "a", encoding="utf-8", errors="surrogatepass") as csv_f:
            for round_idx, rel_err, elapsed, avg_rate, err_type, err_line in results:
                csv_f.write(f"{round_idx},{rel_err},{elapsed},{avg_rate},1,{err_type},{err_line}\n")


        # PDE 分支：完成落盘与评测，并保持 JSON 一致性
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  
    main()
