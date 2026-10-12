#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter a single seed folder by test_results.csv according to given rules.

Rules:
1) Remove rows containing any of the tokens {failed, nan, inf} (case-insensitive) in any field.
2) Remove rows with nRMSE > threshold_nrmse (default 0.001).
3) Remove rows with elapsed_time > threshold_time (default 600).
4) Remove rows with convergence_rate < threshold_cr (default 0).

By default, the filtered rows are written to ``test_results.filtered.csv`` and
the input is left unchanged. Use ``--in-place`` for the legacy overwrite mode.

Usage:
  python data/sft/filter_trajectory.py /path/to/CodePDE/results/seed_0000 \
	--nrmse-threshold 0.15 --time-threshold 1200 --cr-threshold -0.15
"""

import argparse
import csv
import math
import sys
from pathlib import Path


REQUIRED_HEADER = ["round", "nRMSE", "elapsed_time", "convergence_rate", "num_trial"]
INVALID_TOKENS = {"failed", "nan", "inf"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Filter test_results.csv for a single seed folder")
	parser.add_argument(
		"seed_dir",
		type=str,
		help="Path to a seed folder that contains test_results.csv",
	)
	parser.add_argument(
		"--nrmse-threshold",
		type=float,
		default=1e-3,
		help="Max allowed nRMSE; rows with nRMSE greater than this are removed (default: 0.001)",
	)
	parser.add_argument(
		"--time-threshold",
		type=float,
		default=600.0,
		help="Max allowed elapsed_time in seconds; rows greater than this are removed (default: 600)",
	)
	parser.add_argument(
		"--cr-threshold",
		type=float,
		default=0.0,
		help="Min allowed convergence_rate; rows lower than this are removed (default: 0)",
	)
	output_group = parser.add_mutually_exclusive_group()
	output_group.add_argument(
		"--output",
		type=Path,
		help="Output CSV path (default: <seed_dir>/test_results.filtered.csv)",
	)
	output_group.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite test_results.csv (legacy behavior; explicit opt-in)",
	)
	return parser.parse_args()


def has_invalid_token(row: dict) -> bool:
	for v in row.values():
		if v is None:
			continue
		s = str(v).strip().lower()
		# If any field contains failed/nan/inf, drop this row
		if any(tok in s for tok in INVALID_TOKENS):
			return True
	return False


def to_float_safe(s: str):
	try:
		return float(s)
	except Exception:
		return None


def validate_header(fieldnames: list[str]) -> bool:
	return fieldnames is not None and list(fieldnames) == REQUIRED_HEADER


def main() -> int:
	args = parse_args()

	seed_dir = Path(args.seed_dir)
	if not seed_dir.exists() or not seed_dir.is_dir():
		print(f"[ERROR] Provided path is not a directory: {seed_dir}", file=sys.stderr)
		return 2

	csv_path = seed_dir / "test_results.csv"
	if not csv_path.exists():
		print(f"[ERROR] Missing test_results.csv in: {seed_dir}", file=sys.stderr)
		return 2

	# Read all rows first
	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		if not validate_header(reader.fieldnames):
			print(
				f"[ERROR] Invalid CSV header in {csv_path}.\n"
				f"Expected: {REQUIRED_HEADER}\nGot: {reader.fieldnames}",
				file=sys.stderr,
			)
			return 2

		rows = list(reader)

	if len(rows) == 0:
		print(f"[ERROR] No data rows in {csv_path}", file=sys.stderr)
		return 2

	output_path = csv_path if args.in_place else (args.output or seed_dir / "test_results.filtered.csv")
	output_path = Path(output_path).expanduser()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	kept_rows: list[dict] = []
	removed_invalid_token = 0
	removed_threshold_nrmse = 0
	removed_threshold_time = 0
	removed_threshold_cr = 0
	removed_parse_error = 0

	for row in rows:
		# 1) invalid tokens
		if has_invalid_token(row):
			removed_invalid_token += 1
			continue

		nrmse = to_float_safe(row.get("nRMSE", ""))
		etime = to_float_safe(row.get("elapsed_time", ""))
		crate = to_float_safe(row.get("convergence_rate", ""))

		# Parse errors (non-numeric) → remove
		if nrmse is None or etime is None or crate is None:
			removed_parse_error += 1
			continue

		# Also guard against non-finite values
		if not (math.isfinite(nrmse) and math.isfinite(etime) and math.isfinite(crate)):
			removed_invalid_token += 1
			continue

		# 2) nRMSE threshold (strictly greater)
		if nrmse > args.nrmse_threshold:
			removed_threshold_nrmse += 1
			continue

		# 3) elapsed_time threshold (strictly greater)
		if etime > args.time_threshold:
			removed_threshold_time += 1
			continue

		# 4) convergence_rate threshold (strictly smaller)
		if crate < args.cr_threshold:
			removed_threshold_cr += 1
			continue

		kept_rows.append(row)

	# Write filtered rows without changing the input unless --in-place was explicit.
	with output_path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=REQUIRED_HEADER)
		writer.writeheader()
		for r in kept_rows:
			# Ensure output keeps the original string formatting for consistency
			writer.writerow({k: r.get(k, "") for k in REQUIRED_HEADER})

	print(
		"[SUMMARY]",
		f"output={output_path}",
		f"total={len(rows)}",
		f"kept={len(kept_rows)}",
		f"removed_invalid_token={removed_invalid_token}",
		f"removed_parse_error={removed_parse_error}",
		f"removed_nrmse={removed_threshold_nrmse}",
		f"removed_time={removed_threshold_time}",
		f"removed_cr={removed_threshold_cr}",
	)

	if len(kept_rows) == 0:
		print(
			f"[WARNING] All rows were filtered out. Empty output: {output_path}",
			file=sys.stderr,
		)

	return 0


if __name__ == "__main__":
	sys.exit(main())
