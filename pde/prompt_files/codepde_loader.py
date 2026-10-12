"""Load CodePDE prompt text from the separately cloned CodePDE checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pde.paths import get_codepde_root


PDE_DESCRIPTION_SYMBOLS = {
    "advection": "advection_description",
    "burgers": "burgers_description",
    "reacdiff1d": "reacdiff_1d_description",
    "cns1d": "cns1d_description",
    "darcy": "darcy_description",
}


def _load_prompt_module(filename: str, config: Any = None):
    path = get_codepde_root(config) / "prompt_files" / filename
    if not path.is_file():
        raise FileNotFoundError(f"CodePDE prompt source is missing: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_diffpde_external_codepde_{Path(filename).stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CodePDE prompt source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_generation_prompts(config: Any = None) -> tuple[str, str]:
    """Return CodePDE's base system and solver-generation prompt templates."""
    module = _load_prompt_module("general_prompt.py", config)
    try:
        return module.system_prompt, module.code_generation_without_seed_prompt
    except AttributeError as exc:
        raise AttributeError(
            "CodePDE general_prompt.py lacks a required prompt symbol"
        ) from exc


def get_pde_description(pde_type: str, config: Any = None) -> str:
    """Return the unformatted description template for a supported PDE."""
    try:
        symbol = PDE_DESCRIPTION_SYMBOLS[str(pde_type)]
    except KeyError as exc:
        raise KeyError(f"Unsupported CodePDE PDE description: {pde_type}") from exc
    module = _load_prompt_module("pde_descriptions.py", config)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise AttributeError(
            f"CodePDE pde_descriptions.py lacks required symbol: {symbol}"
        ) from exc
