#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-generate Alpaca datasets from all seed_* folders under a run root,
then concatenate into a single dataset JSON.

Example:
  python data/sft/build_sft_dataset.py /path/to/CodePDE/results/run \
	--output-dir /path/to/output

Behavior:
  - For each immediate seed_* subdirectory, call trajectory_to_alpaca.py with --output
	to create/overwrite an Alpaca dataset named: <seed_dir>/alpaca_dataset_<seed_name>.json
  - After all seeds processed, concatenate all generated JSON arrays in order
	and save to <output_dir>/alpaca_dataset_merged.json (overwrite if exists).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Batch-generate and merge Alpaca datasets from seed_* folders")
	p.add_argument("run_root_dir", type=str, help="Run root directory that contains seed_* subfolders")
	p.add_argument(
		"--output-dir", type=str, default=None,
		help="Directory to place the merged dataset (default: run_root_dir)",
	)
	p.add_argument(
		"--merged-filename", type=str, default="alpaca_dataset_merged.json",
		help="Merged dataset filename (default: alpaca_dataset_merged.json)",
	)
	# Cleaning options
	p.add_argument(
		"--use-reply-cleaner", action="store_true",
		help="Use clean_trajectory.py instead of normalize_sections.py."
	)
	return p.parse_args()


def main() -> int:
	args = parse_args()

	run_root = Path(args.run_root_dir).resolve()
	if not run_root.exists() or not run_root.is_dir():
		print(f"[ERROR] Not a directory: {run_root}", file=sys.stderr)
		return 2

	output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root
	output_dir.mkdir(parents=True, exist_ok=True)

	this_dir = Path(__file__).resolve().parent
	single_script = this_dir / "trajectory_to_alpaca.py"
	# Choose cleaner
	cleaner_script = (this_dir / "clean_trajectory.py") if args.use_reply_cleaner else (this_dir / "normalize_sections.py")
	if not single_script.exists():
		print(f"[ERROR] Missing script: {single_script}", file=sys.stderr)
		return 2
	if not cleaner_script.exists():
		print(f"[WARN] Cleaning script not found, skipping cleaning step: {cleaner_script}", file=sys.stderr)

	seed_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
	if not seed_dirs:
		print(f"[WARN] No seed_* subdirectories found under {run_root}")
		# still produce an empty merged file to be explicit
		merged_path = output_dir / args.merged_filename
		merged_path.write_text("[]", encoding="utf-8")
		print(f"[OK] Wrote empty merged dataset: {merged_path}")
		return 0

	generated_files = []
	total = 0
	ok = 0
	failed = 0

	for sd in seed_dirs:
		total += 1
		seed_name = sd.name  # e.g., seed_0000

		# Cleaning step: run selected cleaner
		if cleaner_script.exists():
			clean_cmd = [sys.executable, str(cleaner_script), str(sd)]
			# The builder requires stable source filenames for the following conversion.
			clean_cmd.append("--in-place")
			print(f"[CLEAN] {' '.join(clean_cmd)}")
			try:
				clean_proc = subprocess.run(clean_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
				if clean_proc.returncode == 0:
					print(f"[CLEAN OK] {seed_name}\n{clean_proc.stdout.strip()}\n")
					if clean_proc.stderr:
						print(f"[CLEAN STDERR] {seed_name}\n{clean_proc.stderr.strip()}\n")
				else:
					print(f"[CLEAN FAIL] {seed_name} (code={clean_proc.returncode})\n{clean_proc.stdout.strip()}\n", file=sys.stderr)
					if clean_proc.stderr:
						print(f"[CLEAN STDERR] {seed_name}\n{clean_proc.stderr.strip()}\n", file=sys.stderr)
			except Exception as e:
				print(f"[CLEAN EXCEPTION] {seed_name} exception: {e}", file=sys.stderr)
		out_path = sd / f"alpaca_dataset_{seed_name}.json"
		cmd = [
			sys.executable,
			str(single_script),
			str(sd),
			"--output",
			str(out_path),
		]
		print(f"[RUN] {' '.join(cmd)}")
		try:
			proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		except Exception as e:
			failed += 1
			print(f"[FAIL] {seed_name} exception: {e}", file=sys.stderr)
			continue

		if proc.returncode == 0:
			ok += 1
			print(f"[OK] {seed_name}\n{proc.stdout.strip()}\n")
			if proc.stderr:
				print(f"[STDERR] {seed_name}\n{proc.stderr.strip()}\n")
			if out_path.exists():
				generated_files.append(out_path)
		else:
			failed += 1
			print(f"[FAIL] {seed_name} (code={proc.returncode})\n{proc.stdout.strip()}\n", file=sys.stderr)
			if proc.stderr:
				print(f"[STDERR] {seed_name}\n{proc.stderr.strip()}\n", file=sys.stderr)

	# Merge all generated datasets
	merged = []
	for jf in generated_files:
		try:
			txt = jf.read_text(encoding="utf-8")
			arr = json.loads(txt)
			if isinstance(arr, list):
				merged.extend(arr)
			else:
				print(f"[WARN] Not a list in {jf}; skipping", file=sys.stderr)
		except Exception as e:
			print(f"[WARN] Failed to read {jf}: {e}", file=sys.stderr)

	merged_path = output_dir / args.merged_filename
	try:
		merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
	except Exception as e:
		print(f"[ERROR] Failed to write merged dataset: {e}", file=sys.stderr)
		return 2

	print(f"[SUMMARY] seeds_total={total} ok={ok} failed={failed}")
	print(f"[OK] Wrote merged dataset with {len(merged)} samples to: {merged_path}")
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	sys.exit(main())
