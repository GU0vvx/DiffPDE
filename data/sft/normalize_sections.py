#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strip the first numbered section from response files and renumber remaining sections.

- Targets files that contain three sections with numbered headings like:
  "# 1. ...", "## 2. ...", or "### 3. ...".
- Removes the first section and decrements the numbers of subsequent sections by 1.
- Ignores headings inside fenced code blocks (```).
- Works recursively on directories, processing files matching pattern 'responses_*.txt'.

Usage examples:
  python data/sft/normalize_sections.py /path/to/file.txt
  python data/sft/normalize_sections.py /dir/with/responses --suffix .cleaned.txt
  python data/sft/normalize_sections.py /dir --in-place

Exit code is non-zero if any file fails to process.
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Match a top-level numbered header line, either as Markdown heading (e.g., "# 1. Title")
# or as a plain numbered line (e.g., "1. Title"). We capture an optional heading prefix
# to preserve formatting when rewriting numbers.
HEADER_OR_PLAIN_RE = re.compile(r"^(?P<prefix>\s*(?P<hash>#{1,6}\s*)?)(?P<num>\d+)\.(?P<rest>\s+.*)$")
FENCE_RE = re.compile(r"^\s*```")

def _collect_header_candidates(lines: List[str]) -> List[Tuple[int, int, bool, bool]]:
    """Collect potential numbered header candidates outside code fences.
    Returns tuples: (line_idx, num, has_hash, has_blank_before).
    """
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
            # check blank line before (or start of file) ignoring pure whitespace
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
    Preference: headers with '#'. Otherwise, for plain numbers require blank line before.
    If require_start_at is None, start from the minimal number present.
    """
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
    """Find the indices and numbers of the top-level section headers: 1., 2., 3., ...

    We deliberately look for the first occurrence of a header numbered 1, and then
    the next header numbered 2 that appears after it, then 3, and so on. This helps
    avoid accidentally treating unrelated ordered lists as top-level sections.
    Returns a list of tuples (line_idx, number). Inside fenced code blocks is ignored.
    """
    candidates = _collect_header_candidates(lines)
    return _build_top_level_sequence(candidates, require_start_at=1)


def find_sections(lines: List[str]) -> List[Tuple[int, int, int]]:
    """Return (start_idx, end_idx_exclusive, number) for each top-level section."""
    headers = _find_top_level_header_indices(lines)
    if not headers:
        return []
    sections: List[Tuple[int, int, int]] = []
    for idx, (start, num) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        sections.append((start, end, num))
    return sections


essential_warn = (
    "No eligible 3-section structure found outside code fences; skipping. "
    "Expected top-level sections like '# 1. ...', '# 2. ...', '# 3. ...' or '1. ...', '2. ...', '3. ...'."
)


def process_text(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    sections = find_sections(lines)
    # Only clean responses that still have at least three top-level sections (idempotent guard)
    if not sections or len(sections) < 3:
        return None

    # Identify the first section by order of appearance (usually number==1)
    first_start, first_end, _ = sections[0]

    # Build new content: remove first section
    new_lines: List[str] = []
    new_lines.extend(lines[:first_start])
    new_lines.extend(lines[first_end:])

    # Renumber remaining top-level headers (may start at 2 now) by -1, without touching inner lists
    result_lines: List[str] = new_lines[:]  # copy
    candidates_after = _collect_header_candidates(result_lines)
    sequence_after = _build_top_level_sequence(candidates_after, require_start_at=None)
    for (li, num) in sequence_after:
        m = HEADER_OR_PLAIN_RE.match(result_lines[li])
        if not m:
            continue
        prefix = m.group('prefix')
        rest = m.group('rest')
        new_num = max(1, num - 1)
        if new_num != num:
            result_lines[li] = f"{prefix}{new_num}.{rest}"

    return ''.join(result_lines)


def iter_target_files(paths: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_file():
            # Skip files that look like previously generated cleaned outputs
            if ".cleaned." in p.name:
                continue
            files.append(p)
        elif p.is_dir():
            for f in p.rglob('responses_*.txt'):
                if ".cleaned." in f.name:
                    continue
                files.append(f)
    return files


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='Files or directories to process')
    parser.add_argument('--in-place', action='store_true', help='Overwrite files in place')
    parser.add_argument('--suffix', default='.cleaned.txt', help='Suffix for output files when not in-place')
    parser.add_argument('--dry-run', action='store_true', help='Print planned changes without writing files')
    args = parser.parse_args(argv)

    targets = iter_target_files(args.paths)
    if not targets:
        print('No target files found.', file=sys.stderr)
        return 2

    rc = 0
    for path in targets:
        try:
            text = path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Failed to read {path}: {e}", file=sys.stderr)
            rc = 1
            continue

        new_text = process_text(text)
        if new_text is None:
            print(f"[skip] {path}: {essential_warn}")
            continue

        if args.dry_run:
            print(f"[dry-run] Would process {path} -> in-place={args.in_place}, suffix={args.suffix}")
            continue

        if args.in_place:
            try:
                path.write_text(new_text, encoding='utf-8')
                print(f"[ok] Overwrote {path}")
            except Exception as e:
                print(f"Failed to write {path}: {e}", file=sys.stderr)
                rc = 1
        else:
            out_path = path.with_name(path.stem + args.suffix)
            try:
                out_path.write_text(new_text, encoding='utf-8')
                print(f"[ok] Wrote {out_path}")
            except Exception as e:
                print(f"Failed to write {out_path}: {e}", file=sys.stderr)
                rc = 1

    return rc


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
