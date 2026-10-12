from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from termcolor import cprint


REFINE_BUFFER_FILENAME = "refine_buffer.jsonl"


def _get_buffer_path(project_name: str) -> Path:
	"""Return the path to the refine buffer file for a given project."""
	return Path("..") / project_name / "temp_data" / REFINE_BUFFER_FILENAME


def _load_all_entries(path: Path) -> List[Dict[str, Any]]:
	"""Load all refine-buffer entries from a JSONL file.

	If the file does not exist, return an empty list.
	"""
	if not path.exists():
		return []

	entries: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8", errors="surrogatepass") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			if isinstance(obj, dict):
				entries.append(obj)
	return entries


def _save_all_entries(path: Path, entries: List[Dict[str, Any]]) -> None:
	"""Overwrite the JSONL file with the provided entries."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", errors="surrogatepass") as f:
		for obj in entries:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_records(
	project_name: str,
	records: Iterable[Dict[str, Any]],
	max_size: int,
) -> None:
	"""Append sample-level pointer records to the refine buffer.

	New schema: one JSON object per line.
	Required fields: origin_step, origin_round, is_bad
	Optional fields: pde_type

	This buffer intentionally does NOT store prompt/response/reward/metrics.
	"""
	path = _get_buffer_path(project_name)
	if max_size is not None and max_size <= 0:
		cprint("[refine_buffer] max_size <= 0, skip appending records", "yellow")
		return

	existing = _load_all_entries(path)
	if existing:
		last_id = max(int(e.get("buffer_id", i)) for i, e in enumerate(existing))
	else:
		last_id = 0

	new_entries: List[Dict[str, Any]] = []
	for r in records:
		if not isinstance(r, dict):
			continue
		origin_step = r.get("origin_step")
		origin_round = r.get("origin_round")
		if origin_step is None or origin_round is None:
			continue
		try:
			origin_step_i = int(origin_step)
			origin_round_i = int(origin_round)
		except Exception:
			continue

		entry: Dict[str, Any] = {
			"origin_step": origin_step_i,
			"origin_round": origin_round_i,
			"is_bad": bool(r.get("is_bad", True)),
		}
		if r.get("pde_type") is not None:
			entry["pde_type"] = str(r.get("pde_type"))

		last_id += 1
		entry["buffer_id"] = int(last_id)
		new_entries.append(entry)

	if not new_entries:
		return

	all_entries = existing + new_entries
	if max_size is not None and len(all_entries) > max_size:
		all_entries = all_entries[-int(max_size) :]

	_save_all_entries(path, all_entries)
	cprint(
		f"[refine_buffer] appended {len(new_entries)} record(s); total={len(all_entries)} (limit={max_size})",
		"green",
	)


def append_groups(
	project_name: str,
	groups: Iterable[Dict[str, Any]],
	max_size: int,
) -> None:
	"""Append one or more refine groups to the buffer with a FIFO max size.

	This helper assigns a monotonically increasing "buffer_id" field to newly
	added groups when it is missing.
	"""
	path = _get_buffer_path(project_name)
	if max_size is not None and max_size <= 0:
		# Max size <= 0 means "do not store"; treat as no-op.
		cprint("[refine_buffer] max_size <= 0, skip appending groups", "yellow")
		return

	existing = _load_all_entries(path)
	if existing:
		last_id = max(int(e.get("buffer_id", i)) for i, e in enumerate(existing))
	else:
		last_id = 0

	new_entries: List[Dict[str, Any]] = []
	for g in groups:
		if not isinstance(g, dict):
			continue
		if "buffer_id" not in g:
			last_id += 1
			g = {**g, "buffer_id": int(last_id)}
		new_entries.append(g)

	if not new_entries:
		return

	all_entries = existing + new_entries
	if max_size is not None and len(all_entries) > max_size:
		# Keep the newest max_size entries.
		all_entries = all_entries[-int(max_size) :]

	_save_all_entries(path, all_entries)
	cprint(
		f"[refine_buffer] appended {len(new_entries)} group(s); total={len(all_entries)} (limit={max_size})",
		"green",
	)


def sample_groups(
	project_name: str,
	num_groups: int,
) -> List[Dict[str, Any]]:
	"""Sample up to `num_groups` groups uniformly at random from the buffer.

	This helper is provided for future GRPO training (rl_pde.py / train scripts)
	and is not yet wired into the main RL loop.
	"""
	from random import sample

	path = _get_buffer_path(project_name)
	entries = _load_all_entries(path)
	if not entries or num_groups <= 0:
		return []
	if len(entries) <= num_groups:
		return list(entries)
	return sample(entries, int(num_groups))


def _coerce_to_pointer_records(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Coerce mixed buffer entries into sample-level pointer records.

	Supported shapes:
	- New schema entry: {origin_step, origin_round, is_bad, pde_type?, buffer_id?}
	- Legacy group entry: {samples:[{origin_step, origin_round|metrics.round, is_bad, metrics?}], meta:{pde_type?}, buffer_id?}
	"""
	flat: List[Dict[str, Any]] = []
	for e in entries:
		if not isinstance(e, dict):
			continue

		# New schema
		if "origin_step" in e and "origin_round" in e:
			try:
				origin_step = int(e.get("origin_step"))
				origin_round = int(e.get("origin_round"))
			except Exception:
				continue
			rec: Dict[str, Any] = {
				"buffer_id": e.get("buffer_id"),
				"origin_step": origin_step,
				"origin_round": origin_round,
				"is_bad": bool(e.get("is_bad", True)),
			}
			if e.get("pde_type") is not None:
				rec["pde_type"] = str(e.get("pde_type"))
			flat.append(rec)
			continue

		# Legacy group schema
		meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
		pde_type = meta.get("pde_type")
		buffer_id = e.get("buffer_id")
		samples = e.get("samples")
		if not isinstance(samples, list):
			continue
		for s in samples:
			if not isinstance(s, dict):
				continue
			origin_step = s.get("origin_step")
			origin_round = s.get("origin_round")
			if origin_round is None:
				m = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
				origin_round = m.get("round")
			try:
				origin_step_i = int(origin_step)
				origin_round_i = int(origin_round)
			except Exception:
				continue
			rec = {
				"buffer_id": buffer_id,
				"origin_step": origin_step_i,
				"origin_round": origin_round_i,
				"is_bad": bool(s.get("is_bad", True)),
			}
			if pde_type is not None:
				rec["pde_type"] = str(pde_type)
			flat.append(rec)
	return flat


def sample_samples(
	project_name: str,
	num_samples: int,
	good_ratio: float = 0.5,
	seed: Optional[int] = None,
	require_round_for_bad: bool = True,
) -> List[Dict[str, Any]]:
	"""Backward compatible alias.

	New code should use sample_records(...) to get sample-level pointer records.
	"""
	return sample_records(
		project_name=project_name,
		num_samples=num_samples,
		good_ratio=good_ratio,
		seed=seed,
		require_round_for_bad=require_round_for_bad,
		pde_type=None,
	)


def sample_records(
	project_name: str,
	num_samples: int,
	good_ratio: float = 0.5,
	seed: Optional[int] = None,
	require_round_for_bad: bool = True,
	pde_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
	"""Sample up to `num_samples` pointer records from refine buffer.

	- good/bad is determined by record['is_bad'] (missing -> bad)
	- pde_type filters records if provided
	"""
	if num_samples <= 0:
		return []
	good_ratio = max(0.0, min(1.0, float(good_ratio)))

	path = _get_buffer_path(project_name)
	entries = _load_all_entries(path)
	flat = _coerce_to_pointer_records(entries)
	if pde_type is not None:
		flat = [r for r in flat if str(r.get("pde_type", "")) == str(pde_type)]
	if not flat:
		return []

	good: List[Dict[str, Any]] = []
	bad: List[Dict[str, Any]] = []
	for rec in flat:
		try:
			origin_step = int(rec.get("origin_step"))
			origin_round = int(rec.get("origin_round"))
		except Exception:
			continue
		is_bad = bool(rec.get("is_bad", True))
		if is_bad and require_round_for_bad:
			if origin_step is None or origin_round is None:
				continue
		if is_bad:
			bad.append(rec)
		else:
			good.append(rec)

	rng = random.Random(seed)
	need_good = int(round(num_samples * good_ratio))
	need_bad = int(num_samples - need_good)

	need_good = min(need_good, len(good))
	need_bad = min(need_bad, len(bad))

	if need_good + need_bad < num_samples:
		remain = num_samples - (need_good + need_bad)
		if len(good) - need_good >= remain:
			need_good += remain
		elif len(bad) - need_bad >= remain:
			need_bad += remain
		else:
			need_good = len(good)
			need_bad = len(bad)

	selected: List[Dict[str, Any]] = []
	if need_good > 0:
		selected.extend(rng.sample(good, need_good) if len(good) > need_good else list(good))
	if need_bad > 0:
		selected.extend(rng.sample(bad, need_bad) if len(bad) > need_bad else list(bad))

	rng.shuffle(selected)
	return selected
