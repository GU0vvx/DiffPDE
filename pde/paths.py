"""Resolve external CodePDE assets and evaluator runtime paths."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _config_value(config: Any, dotted_key: str) -> Any:
	current = config
	for part in dotted_key.split("."):
		if current is None:
			return None
		if isinstance(current, Mapping):
			current = current.get(part)
		else:
			current = getattr(current, part, None)
	return current


def get_codepde_root(config: Any = None) -> Path:
	"""Return the configured CodePDE checkout root.

	``CODEPDE_ROOT`` has priority over the optional
	``paths.codepde_root`` config field.
	"""
	root_value = os.environ.get("CODEPDE_ROOT") or _config_value(config, "paths.codepde_root")
	if not root_value:
		raise RuntimeError(
			"CodePDE root is not configured. Set CODEPDE_ROOT or "
			"paths.codepde_root in the config."
		)

	root = Path(str(root_value)).expanduser().resolve()
	if not root.is_dir():
		raise FileNotFoundError(f"Configured CodePDE root is not a directory: {root}")
	return root


def _solver_asset_path(pde_type: str, filename: str, config: Any = None) -> Path:
	path = get_codepde_root(config) / "solvers" / str(pde_type) / filename
	if not path.is_file():
		raise FileNotFoundError(
			f"CodePDE asset not found for PDE type '{pde_type}': {path}"
		)
	return path


def get_solver_template_path(pde_type: str, config: Any = None) -> Path:
	return _solver_asset_path(pde_type, "solver_template.py", config)


def get_evaluator_path(pde_type: str, config: Any = None) -> Path:
	return _solver_asset_path(pde_type, "evaluator.py", config)


def get_evaluator_python(config: Any = None) -> str:
	"""Resolve the Python executable used to run CodePDE evaluators.

	Resolution order: ``paths.evaluator_python`` config,
	``DIFFPDE_EVALUATOR_PYTHON`` environment variable, then ``sys.executable``.
	"""
	value = (
		_config_value(config, "paths.evaluator_python")
		or os.environ.get("DIFFPDE_EVALUATOR_PYTHON")
		or sys.executable
	)
	executable = str(value).strip()
	if not executable:
		raise RuntimeError("Evaluator Python executable resolved to an empty value.")

	if os.path.isabs(executable) or os.sep in executable:
		path = Path(executable).expanduser().resolve()
		if not path.is_file():
			raise FileNotFoundError(f"Evaluator Python executable not found: {path}")
		return str(path)

	resolved = shutil.which(executable)
	if not resolved:
		raise FileNotFoundError(f"Evaluator Python executable not found on PATH: {executable}")
	return resolved
