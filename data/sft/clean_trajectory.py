#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean the first reply to the first user message per round by removing the first numbered section
and renumbering the remaining sections.

Target locations per round r under a seed directory:
- messages_r.json: If there exists an assistant reply following the FIRST role=="user",
  clean that assistant's content in-place (or write with suffix when not in-place).
- Otherwise, responses_r.txt: Clean its content instead (in-place or with suffix).

Detection and cleaning rules:
- The text is typically in a three-section format using Markdown headings like:
  "# 1. ...", "## 2. ...", or "### 3. ...".
- Remove the entire first numbered section and decrement the numbers of subsequent sections by 1.
- Ignore headings inside fenced code blocks delimited by triple backticks (```),
  i.e., do not treat them as section headings for removal/renumbering purposes.

Default behavior (no --in-place):
    - The original file is left unchanged.
    - Cleaned content is written to a sibling with suffix ``.cleaned``.

Usage examples:
    # Non-destructive by default: writes messages_0.cleaned.json (or responses_0.cleaned.txt)
    python data/sft/clean_trajectory.py /path/to/seed_0000

    # In-place overwrite without creating backup copy
    python data/sft/clean_trajectory.py /path/to/seed_0000 --in-place

    # Recursively over a run root that contains seed_* dirs
    python data/sft/clean_trajectory.py /path/to/CodePDE/results/run

    # Single round file processing
    python data/sft/clean_trajectory.py /path/to/messages_0.json

Exit code is non-zero if any file fails to process.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Header or plain numbered line like: "# 1. Title" or "1. Title"
HEADER_OR_PLAIN_RE = re.compile(r"^(?P<prefix>\s*(?P<hash>#{1,6}\s*)?)(?P<num>\d+)\.(?P<rest>\s+.*)$")
FENCE_RE = re.compile(r"^\s*```")

MSG_FILE_RE = re.compile(r"messages_(?P<round>\d+)\.json$")


def _collect_header_candidates(lines: List[str]) -> List[Tuple[int, int, bool, bool]]:
    """Collect potential numbered header candidates outside code fences.
    Returns (line_idx, num, has_hash, has_blank_before)."""
    candidates: List[Tuple[int, int, bool, bool]] = []
    in_code = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEADER_OR_PLAIN_RE.match(line)
        if m:
            has_hash = m.group('hash') is not None
            j = i - 1
            has_blank_before = False
            while j >= 0 and lines[j].strip() == "":
                has_blank_before = True
                j -= 1
            if i == 0:
                has_blank_before = True
            candidates.append((i, int(m.group('num')), has_hash, has_blank_before))
    return candidates

def _build_top_level_sequence(candidates: List[Tuple[int, int, bool, bool]], require_start_at: int | None = 1) -> List[Tuple[int, int]]:
    """Build an increasing top-level header sequence (line_idx, num).
    Prefer '#' headers; else in plain-number mode require blank line before.
    If require_start_at is None, start from the minimal number present."""
    if not candidates:
        return []
    prefer_hash = any(has_hash for _, _, has_hash, _ in candidates)
    filtered = [c for c in candidates if (c[2] if prefer_hash else True)]
    if not filtered:
        return []
    if not prefer_hash:
        filtered = [c for c in filtered if c[3]]
        if not filtered:
            return []
    start_num = require_start_at if require_start_at is not None else min(num for _, num, _, _ in filtered)
    headers: List[Tuple[int, int]] = []
    start_pos = None
    for idx, (li, num, _, _) in enumerate(filtered):
        if num == start_num:
            headers.append((li, num))
            start_pos = idx
            break
    if start_pos is None:
        return []
    want = start_num + 1
    for li, num, _, _ in filtered[start_pos + 1:]:
        if num == want:
            headers.append((li, num))
            want += 1
    return headers

def _find_top_level_header_indices(lines: List[str]) -> List[Tuple[int, int]]:
    """Find indices and numbers of top-level 1., 2., 3., ... headers outside code fences."""
    return _build_top_level_sequence(_collect_header_candidates(lines), require_start_at=1)


def process_text(text: str) -> Optional[str]:
    """Remove the first numbered section and decrement following section numbers by 1.
    Ignore headings found inside fenced code blocks.
    Return cleaned text if any change applied; otherwise return None.
    """
    lines = text.splitlines(keepends=True)

    headers_pairs = _find_top_level_header_indices(lines)
    # Idempotent guard: only clean when we still have at least 3 top-level sections
    if not headers_pairs or len(headers_pairs) < 3:
        return None  # nothing to clean

    # Identify first section span
    first_start = headers_pairs[0][0]
    first_end = headers_pairs[1][0] if len(headers_pairs) > 1 else len(lines)

    # Build content without the first section
    new_lines: List[str] = []
    new_lines.extend(lines[:first_start])
    new_lines.extend(lines[first_end:])

    # Renumber remaining top-level headers after removal (may now start at 2)
    result_lines: List[str] = new_lines[:]
    headers_after = _build_top_level_sequence(_collect_header_candidates(result_lines), require_start_at=None)
    for (li, num) in headers_after:
        m = HEADER_OR_PLAIN_RE.match(result_lines[li])
        if not m:
            continue
        prefix = m.group('prefix')
        rest = m.group('rest')
        new_num = max(1, num - 1)
        if new_num != num:
            result_lines[li] = f"{prefix}{new_num}.{rest}"

    return ''.join(result_lines)


def _normalize_top_level_numbering(text: str) -> Optional[str]:
    """Normalize top-level numbered headers to start from 1 without removing any content.
    Prefer '#' headers; skip headings inside code fences. If minimal header number > 1
    and we have at least two headers, decrement all by (min_num - 1).
    Returns updated text or None if no change is needed.
    """
    lines = text.splitlines(keepends=True)
    candidates = _collect_header_candidates(lines)
    headers = _build_top_level_sequence(candidates, require_start_at=None)
    if not headers or len(headers) < 2:
        return None
    min_num = min(num for _, num in headers)
    if min_num <= 1:
        return None
    shift = min_num - 1
    new_lines = lines[:]
    for (li, num) in headers:
        m = HEADER_OR_PLAIN_RE.match(new_lines[li])
        if not m:
            continue
        prefix = m.group('prefix')
        rest = m.group('rest')
        new_num = num - shift
        new_lines[li] = f"{prefix}{new_num}.{rest}"
    return ''.join(new_lines) if new_lines != lines else None


def find_first_user_index(messages: List[dict]) -> Optional[int]:
    for i, m in enumerate(messages):
        if isinstance(m, dict) and str(m.get('role', '')).strip() == 'user':
            return i
    return None


def find_first_assistant_after(messages: List[dict], start_idx: int) -> Optional[int]:
    for j in range(start_idx + 1, len(messages)):
        m = messages[j]
        if isinstance(m, dict) and str(m.get('role', '')).strip() == 'assistant':
            return j
    return None


def _write_cleaned_text(path: Path, content: str, in_place: bool, output_suffix: str) -> bool:
    """Write to a sibling output by default, or overwrite with explicit opt-in."""
    try:
        output_path = path if in_place else path.with_name(f"{path.stem}{output_suffix}{path.suffix}")
        output_path.write_text(content, encoding='utf-8')
        action = "Overwrote" if in_place else "Wrote cleaned output"
        print(f"[ok] {action}: {output_path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to write cleaned output for {path}: {e}", file=sys.stderr)
        return False


def clean_round(seed_dir: Path, msg_path: Path, in_place: bool, output_suffix: str, normalize_only: bool = False) -> bool:
    """Clean one round according to the described rule.
    Returns True if a change was written successfully, else False.
    """
    try:
        messages = json.loads(msg_path.read_text(encoding='utf-8'))
        if not isinstance(messages, list):
            print(f"[WARN] messages not a list: {msg_path}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[WARN] Failed to load {msg_path}: {e}", file=sys.stderr)
        return False

    first_user_idx = find_first_user_index(messages)
    if first_user_idx is None:
        print(f"[skip] No user message in {msg_path}")
        return False

    # Locate assistant reply to the first user
    asst_idx = find_first_assistant_after(messages, first_user_idx)

    # Derive round number for responses path
    m = MSG_FILE_RE.search(msg_path.name)
    round_str = m.group('round') if m else None
    responses_path = seed_dir / f"responses_{round_str}.txt" if round_str else None

    if asst_idx is not None:
        content = str(messages[asst_idx].get('content', ''))
        new_content = _normalize_top_level_numbering(content) if normalize_only else process_text(content)
        if new_content is None:
            action = "normalize numbering" if normalize_only else "clean"
            print(f"[skip] No numbered sections to {action} in assistant reply: {msg_path}")
            return False
        messages[asst_idx]['content'] = new_content
        content_json = json.dumps(messages, ensure_ascii=False, indent=2)
        return _write_cleaned_text(msg_path, content_json, in_place=in_place, output_suffix=output_suffix)
    else:
        # Fall back to responses_r.txt
        if not responses_path or not responses_path.exists():
            print(f"[skip] No assistant reply in messages and missing responses file for round {round_str}: {msg_path}")
            return False
        try:
            txt = responses_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[WARN] Failed to read {responses_path}: {e}", file=sys.stderr)
            return False
        new_txt = _normalize_top_level_numbering(txt) if normalize_only else process_text(txt)
        if new_txt is None:
            action = "normalize numbering" if normalize_only else "clean"
            print(f"[skip] No numbered sections to {action} in {responses_path}")
            return False
        return _write_cleaned_text(responses_path, new_txt, in_place=in_place, output_suffix=output_suffix)


def iter_targets(paths: List[Path]) -> List[Path]:
    targets: List[Path] = []
    for p in paths:
        if p.is_file() and p.name.startswith('messages_') and p.name.endswith('.json') and '.cleaned.' not in p.name:
            targets.append(p)
        elif p.is_dir():
            # recurse seed_* dirs or any with messages_*.json
            targets.extend(f for f in p.rglob('messages_*.json') if '.cleaned.' not in f.name)
    return targets


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='Seed directories or messages_*.json files to process')
    parser.add_argument('--in-place', action='store_true', help='Overwrite files in place (explicit legacy opt-in)')
    parser.add_argument('--output-suffix', default='.cleaned', help='Suffix before the extension for non-destructive output')
    parser.add_argument('--dry-run', action='store_true', help='Parse and report but do not write any files')
    parser.add_argument('--normalize-only', action='store_true', help='Only renumber top-level headers to start from 1; do not remove any section')
    args = parser.parse_args(argv)

    targets = iter_targets(args.paths)
    if not targets:
        print('No target messages_*.json found.', file=sys.stderr)
        return 2

    rc = 0
    for msg_path in targets:
        seed_dir = msg_path.parent
        if args.dry_run:
            print(f"[dry-run] Would process round file: {msg_path}")
            continue
        ok = clean_round(
            seed_dir,
            msg_path,
            in_place=args.in_place,
            output_suffix=args.output_suffix,
            normalize_only=args.normalize_only,
        )
        if not ok:
            rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
