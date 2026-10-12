#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# Non-destructive by default; writes <input>.normalized.json.
python data/sft/normalize_prompts.py \
--input /path/to/sft_dataset.json \
--reference /path/to/CodePDE/results/messages_0.json

'''


import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(obj: Any, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_user_prompt_from_reference(ref_data: Any) -> str:
	"""Extract the user content string from reference messages list.

	Expected structure: a list with at least two items, where the second item
	has keys {"role": "user", "content": <str>}.
	If the second item is not the user, fallback to the first item whose role=="user".
	"""
	if not isinstance(ref_data, list) or len(ref_data) < 2:
		raise ValueError("Reference file must be a list with at least two elements.")

	# Prefer the second element's content
	second = ref_data[1]
	if isinstance(second, dict) and "content" in second:
		return str(second["content"])  # Ensure string

	# Fallback: find first user role
	for item in ref_data:
		if isinstance(item, dict) and item.get("role") == "user" and "content" in item:
			return str(item["content"])  # Ensure string

	raise ValueError("Could not find a user 'content' entry in reference file.")


def modify_dataset_inplace(dataset: Any, user_prompt: str) -> Tuple[int, int, int]:
	"""Modify dataset in-place following the rules:
	- If item['history'] == [] (empty list), replace item['instruction'] with user_prompt
	- Else (history not empty), replace history[0] with user_prompt

	Returns (total, empty_history_count, non_empty_history_count)
	"""
	if not isinstance(dataset, list):
		raise ValueError("Target dataset must be a list of items.")

	total = len(dataset)
	empty_hist = 0
	non_empty_hist = 0

	for idx, item in enumerate(dataset):
		if not isinstance(item, dict):
			# Skip invalid entries, but keep counting
			continue

		hist = item.get("history", [])
		if not isinstance(hist, list):
			hist = []  # Coerce to empty list if malformed

		if len(hist) == 0:
			item["instruction"] = user_prompt
			item["history"] = []  # ensure it's exactly an empty list
			empty_hist += 1
		else:
			# Replace the first element regardless of its previous type
			hist[0] = user_prompt
			item["history"] = hist
			non_empty_hist += 1

	return total, empty_hist, non_empty_hist


def backup_file(path: str) -> str:
	bak_path = path + ".bak"
	shutil.copy2(path, bak_path)
	return bak_path


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Modify SFT dataset instructions/history via a reference user prompt.")
	p.add_argument(
		"--input",
		required=True,
		help="Path to the target dataset JSON to modify (list of items).",
	)
	p.add_argument(
		"--reference",
		required=True,
		help="Path to the reference messages JSON (list with system+user).",
	)
	output_group = p.add_mutually_exclusive_group()
	output_group.add_argument(
		"--output",
		default=None,
		help="Output path (default: <input_stem>.normalized.json).",
	)
	output_group.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite input after creating a .bak backup (explicit legacy opt-in).",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()

	target_path = os.path.abspath(args.input)
	ref_path = os.path.abspath(args.reference)
	if args.in_place:
		out_path = target_path
	elif args.output:
		out_path = os.path.abspath(args.output)
	else:
		base, ext = os.path.splitext(target_path)
		out_path = base + ".normalized" + (ext or ".json")

	dataset = load_json(target_path)
	ref_data = load_json(ref_path)

	user_prompt = extract_user_prompt_from_reference(ref_data)
	print(f"Loaded reference user prompt, length={len(user_prompt)}")

	total, empty_hist, non_empty_hist = modify_dataset_inplace(dataset, user_prompt)
	print(
		f"Processed {total} items: empty_history={empty_hist}, non_empty_history={non_empty_hist}"
	)

	if args.in_place:
		bak = backup_file(target_path)
		print(f"Backup created: {bak}")
	save_json(dataset, out_path)
	print(f"Written to: {out_path}")


if __name__ == "__main__":
	main()
