#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch filter for refine_traversal results.

Given a CodePDE run root, this script finds all immediate subdirectories
matching ``seed_*`` and applies ``filter_trajectory.py`` sequentially.

Usage:
  python data/sft/filter_trajectories.py /path/to/CodePDE/results/run \
	  [--nrmse-threshold 0.001] [--time-threshold 600] [--cr-threshold 0]




!!! only filter REFINE-TRAVERSAL results !!!

Notes:
- The single-seed filter script must be colocated as ``filter_trajectory.py``.
- This script prints a per-seed status line and a final summary.
- The default writes ``test_results.filtered.csv``. Use ``--in-place`` only
  when a downstream legacy tool requires the name ``test_results.csv``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch filter seed_* folders under a run root directory")
    p.add_argument("run_root_dir", type=str, help="Path to a refine_traversal run root directory")
    p.add_argument("--nrmse-threshold", type=float, default=1e-3,
                   help="Max allowed nRMSE; rows with nRMSE greater than this are removed (default: 0.001)")
    p.add_argument("--time-threshold", type=float, default=600.0,
                   help="Max allowed elapsed_time in seconds; rows greater than this are removed (default: 600)")
    p.add_argument("--cr-threshold", type=float, default=0.0,
                   help="Min allowed convergence_rate; rows lower than this are removed (default: 0)")
    p.add_argument("--in-place", action="store_true",
                   help="Explicitly overwrite each seed's test_results.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    run_root = Path(args.run_root_dir)
    if not run_root.exists() or not run_root.is_dir():
        print(f"[ERROR] Not a directory: {run_root}", file=sys.stderr)
        return 2

    this_dir = Path(__file__).resolve().parent
    single_script = this_dir / "filter_trajectory.py"
    if not single_script.exists():
        print(f"[ERROR] Missing single filter script: {single_script}", file=sys.stderr)
        return 2

    # Collect seed_* subdirectories (only one level deep)
    seed_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        print(f"[WARN] No seed_* subdirectories found under {run_root}")
        return 0

    total = 0
    ok = 0
    failed = 0

    for sd in seed_dirs:
        total += 1
        cmd = [sys.executable, str(single_script), str(sd),
               "--nrmse-threshold", str(args.nrmse_threshold),
               "--time-threshold", str(args.time_threshold),
               "--cr-threshold", str(args.cr_threshold)]
        if args.in_place:
            cmd.append("--in-place")
        print(f"[RUN] {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {sd} exception: {e}", file=sys.stderr)
            continue

        if proc.returncode == 0:
            ok += 1
            print(f"[OK] {sd}\n{proc.stdout.strip()}\n")
            if proc.stderr:
                print(f"[STDERR] {sd}\n{proc.stderr.strip()}\n")
        else:
            failed += 1
            print(f"[FAIL] {sd} (code={proc.returncode})\n{proc.stdout.strip()}\n", file=sys.stderr)
            if proc.stderr:
                print(f"[STDERR] {sd}\n{proc.stderr.strip()}\n", file=sys.stderr)

    print(f"[SUMMARY] total={total} ok={ok} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
