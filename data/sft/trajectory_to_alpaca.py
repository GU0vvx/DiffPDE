#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate an Alpaca-format dataset from a refine_traversal seed folder.

Input folder example: ``/path/to/CodePDE/results/run/seed_0000``.

This script will:
  1) Read test_results.csv to get the list of valid rounds (rows).
  2) For each round r, read messages_r.json and responses_r.txt.
  3) Convert them to a single Alpaca sample:
	 - system: messages[0].content if role=='system'
	 - history: all complete (user,assistant) pairs except the final user
	 - instruction: the last user content
	 - output: content of responses_r.txt (latest assistant output)
	 - input: set to "" by default (optional field)

Notes:
  - messages_r.json is expected to be a list of {role, content} dicts.
  - The last assistant of the latest iteration is NOT included in messages_r.json; it is stored in responses_r.txt.
  - If structure is unexpected, the script will try a best-effort extraction and log warnings.

Additional behavior:
	- When parsing messages_r.json, the FIRST encountered user message's content will be overridden (once)
		by the content of the FIRST user message found in the parent directory's file 'message.json' (if available).
		This override applies regardless of whether that first user message eventually becomes part of history or
		becomes the final instruction. If parent message.json is missing or malformed, the original content is kept.

Usage:
  python data/sft/trajectory_to_alpaca.py /path/to/seed_dir \
	  --output /path/to/out.json

If --output is not provided, a default file 'alpaca_dataset.json' will be created under the seed_dir.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


REQUIRED_HEADER = ["round", "nRMSE", "elapsed_time", "convergence_rate", "num_trial"]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate Alpaca dataset from a seed folder")
	p.add_argument("seed_dir", type=str, help="Path to a seed folder with test_results.csv and messages_*.json/responds_*.txt")
	p.add_argument("--output", type=str, default=None, help="Output JSON file path (default: <seed_dir>/alpaca_dataset.json)")
	return p.parse_args()


def read_csv_rounds(csv_path: Path):
	rounds = []
	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		if list(reader.fieldnames) != REQUIRED_HEADER:
			print(
				f"[ERROR] Invalid CSV header in {csv_path}.\n"
				f"Expected: {REQUIRED_HEADER}\nGot: {reader.fieldnames}",
				file=sys.stderr,
			)
			return []
		for row in reader:
			r = row.get("round", "").strip()
			# Only keep integer rounds
			try:
				rnd = int(float(r))  # tolerate '12.0'
				rounds.append(rnd)
			except Exception:
				# Skip non-numeric (e.g., 'failed')
				continue
	return rounds


def load_messages(messages_path: Path):
	try:
		data = json.loads(messages_path.read_text(encoding="utf-8"))
		if not isinstance(data, list):
			print(f"[WARN] messages not a list: {messages_path}", file=sys.stderr)
			return []
		# normalize entries to dicts with role/content
		norm = []
		for m in data:
			if isinstance(m, dict) and "role" in m and "content" in m:
				norm.append({"role": str(m["role"]).strip(), "content": str(m["content"])})
			else:
				print(f"[WARN] skip malformed message in {messages_path}: {m}", file=sys.stderr)
		return norm
	except Exception as e:
		print(f"[ERROR] Failed to load {messages_path}: {e}", file=sys.stderr)
		return []


def load_parent_first_user_content(seed_dir: Path) -> str | None:
	"""Read seed_dir.parent / 'message.json' and return the content of the first entry
	with role == 'user'. If not found or errors occur, return None.
	"""
	parent = seed_dir.parent
	# Prefer 'message.json', fallback to 'messages.json'
	ref_path = parent / "message.json"
	if not ref_path.exists():
		alt_path = parent / "messages.json"
		ref_path = alt_path if alt_path.exists() else ref_path
	try:
		if not ref_path.exists():
			return None
		data = json.loads(ref_path.read_text(encoding="utf-8"))
		if isinstance(data, list):
			for m in data:
				if isinstance(m, dict) and str(m.get("role", "")).strip() == "user":
					return str(m.get("content", ""))
		else:
			# Some runs might store an object { messages: [...] }
			msgs = data.get("messages") if isinstance(data, dict) else None
			if isinstance(msgs, list):
				for m in msgs:
					if isinstance(m, dict) and str(m.get("role", "")).strip() == "user":
						return str(m.get("content", ""))
	except Exception as e:
		print(f"[WARN] Failed to read parent user content from {ref_path}: {e}", file=sys.stderr)
	return None


def build_alpaca_sample(seed_dir: Path, rnd: int):
	messages_path = seed_dir / f"messages_{rnd}.json"
	responses_path = seed_dir / f"responses_{rnd}.txt"

	if not messages_path.exists() or not responses_path.exists():
		print(f"[WARN] Missing messages or responses for round {rnd}, skip.")
		return None

	messages = load_messages(messages_path)
	if not messages:
		print(f"[WARN] Empty or invalid messages for round {rnd}, skip.")
		return None

	# Extract system (optional)
	system_text = ""
	idx = 0
	if messages[0].get("role") == "system":
		system_text = messages[0].get("content", "")
		idx = 1

	history = []
	last_user = None

	# Prepare override for the FIRST encountered user message
	parent_user_override = load_parent_first_user_content(seed_dir)
	first_user_replaced = False

	# Walk through remaining messages in (user,assistant) pairs, leaving the final user as instruction
	n = len(messages)
	while idx < n:
		if messages[idx].get("role") != "user":
			# Unexpected; try to skip until we find a user
			print(f"[WARN] Round {rnd}: expected 'user' at idx={idx}, got {messages[idx].get('role')}. Trying to continue.", file=sys.stderr)
			idx += 1
			continue
		user_content = messages[idx].get("content", "")
		# Override only for the very first encountered user
		if not first_user_replaced and parent_user_override is not None:
			user_content = parent_user_override
			first_user_replaced = True
			print(f"[INFO] Round {rnd}: first user content overridden from parent 'message.json'", file=sys.stderr)
		idx += 1

		if idx >= n:
			# This user is the last one → instruction
			last_user = user_content
			break

		if messages[idx].get("role") == "assistant":
			assistant_content = messages[idx].get("content", "")
			history.append([user_content, assistant_content])
			idx += 1
		else:
			# Next is not assistant → treat current user as final instruction
			print(f"[WARN] Round {rnd}: user not followed by assistant at idx={idx}. Treat as final instruction.", file=sys.stderr)
			last_user = user_content
			# do not advance idx here; next loop will attempt to parse further but commonly ends
			break

	# Fallback if we never found a trailing user for instruction
	if last_user is None:
		# If there is at least one user-assistant pair, take the last user of last pair as instruction
		if history:
			last_user = history.pop()[0]
		else:
			# As a very last resort, use the very first non-system message
			# (only happens in malformed inputs)
			first_non_system = next((m for m in messages if m.get("role") != "system"), None)
			last_user = first_non_system.get("content", "") if first_non_system else ""

	# Load output from responses
	try:
		output_text = responses_path.read_text(encoding="utf-8", errors="ignore")
	except Exception as e:
		print(f"[WARN] Round {rnd}: cannot read responses file: {e}", file=sys.stderr)
		output_text = ""

	sample = {
		"instruction": last_user or "",
		"input": "",
		"output": output_text,
		"system": system_text,
		"history": history,
	}
	return sample


def main() -> int:
	args = parse_args()
	seed_dir = Path(args.seed_dir)
	if not seed_dir.exists() or not seed_dir.is_dir():
		print(f"[ERROR] Not a directory: {seed_dir}", file=sys.stderr)
		return 2

	csv_path = seed_dir / "test_results.csv"
	if not csv_path.exists():
		print(f"[ERROR] Missing test_results.csv under {seed_dir}", file=sys.stderr)
		return 2

	rounds = read_csv_rounds(csv_path)
	if not rounds:
		print(f"[ERROR] No valid rounds found in {csv_path}", file=sys.stderr)
		return 2

	samples = []
	for rnd in rounds:
		s = build_alpaca_sample(seed_dir, rnd)
		if s is not None:
			samples.append(s)

	if not samples:
		print(f"[ERROR] No valid samples were built. Aborting.", file=sys.stderr)
		return 2

	out_path = Path(args.output) if args.output else (seed_dir / "alpaca_dataset.json")
	try:
		out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
	except Exception as e:
		print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
		return 2

	print(f"[OK] Wrote {len(samples)} samples to: {out_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
