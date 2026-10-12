#!/usr/bin/env python
"""Convert a generic SFT dataset (instruction/input/output/system/history) into
the format required by `sft_trado.py` (a JSON list of {prompt, response}).

Rules (per user request):
  - Iterate all samples in source JSON (list of dicts).
  - Keep only samples where `history` is an empty list (== []).
  - For kept samples, build `prompt` = system + instruction (concatenated in order).
    (If either field is missing or empty, treat it as empty string.)
  - Use original `output` field as `response`.
  - Skip samples where `output` is missing / empty after stripping.
  - Output is a JSON list placed at target path.

CLI Usage:
  python data/sft/to_prompt_response.py \
      --input /path/to/alpaca_dataset.json \
      --output /path/to/output/prompt_response.json

Optional Flags:
  --strip          : strip leading/trailing whitespace (default True)
  --newline-sep    : insert a newline between system and instruction instead of a space
  --lowercase      : lowercase prompt (not enabled by default)

Notes:
  - This script does NOT tokenize; it only restructures JSON.
  - Ensure the output filename matches `training.dataset.optimization_data` in config (without .json).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(system: str, instruction: str, use_newline: bool) -> str:
    sep = "\n" if use_newline else " "
    if not system and not instruction:
        return ""
    if system and instruction:
        return f"{system}{sep}{instruction}".strip()
    return (system or instruction).strip()


def convert(samples: List[Dict[str, Any]], *, strip_ws: bool, newline_sep: bool, lowercase: bool) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, ex in enumerate(samples):
        # Validate history empty
        history = ex.get("history", None)
        if history != []:  # Only keep strictly empty list
            continue

        system = ex.get("system", "") or ""
        instruction = ex.get("instruction", "") or ""
        output = ex.get("output", "") or ""

        if strip_ws:
            system = system.strip()
            instruction = instruction.strip()
            output = output.strip()

        if not output:  # Skip if no usable response
            continue

        prompt = build_prompt(system, instruction, newline_sep)
        if lowercase:
            prompt = prompt.lower()

        out.append({"prompt": prompt, "response": output})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert dataset to TraDo SFT format")
    p.add_argument("--input", required=True, help="Source JSON file path")
    p.add_argument("--output", required=True, help="Destination JSON file path")
    p.add_argument("--strip", action="store_true", help="Strip whitespace (default True)")
    p.add_argument("--no-strip", action="store_true", help="Disable stripping whitespace")
    p.add_argument("--newline-sep", action="store_true", help="Use newline between system and instruction")
    p.add_argument("--lowercase", action="store_true", help="Lowercase prompt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Default strip True unless explicitly disabled
    strip_ws = True
    if args.no_strip:
        strip_ws = False
    elif args.strip:
        strip_ws = True

    samples = load_json(args.input)
    converted = convert(samples, strip_ws=strip_ws, newline_sep=args.newline_sep, lowercase=args.lowercase)
    save_json(converted, args.output)
    print(f"Converted {len(converted)} samples (history == [] and output present). Saved to: {args.output}")

    # Simple stats
    if converted:
        max_prompt_len_chars = max(len(x["prompt"]) for x in converted)
        max_resp_len_chars = max(len(x["response"]) for x in converted)
        print(f"Max prompt chars: {max_prompt_len_chars}; Max response chars: {max_resp_len_chars}")
    else:
        print("Warning: No samples converted.")


if __name__ == "__main__":
    main()
