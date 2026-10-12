# Third-party notices

## dLLM-RL

- Repository: https://github.com/Gen-Verse/dLLM-RL
- Pinned base commit: `bca67b12f700bec4a3fb2dfe50adf04eb62cad5a`
- License: Apache License 2.0. A faithful copy is included at
  `THIRD_PARTY_LICENSES/dLLM-RL-APACHE-2.0.txt`.
- Role: DiffPDE overlays PDE-specific sampling, controller, execution, and
  training code onto a separately cloned dLLM-RL checkout. Some of those
  components are adapted from dLLM-RL and were modified by the DiffPDE
  authors. The adapted files contain file-level provenance and modification
  notices; the upstream Dream framework remains in the separate checkout.

## CodePDE

- Repository: https://github.com/LithiumDA/CodePDE
- Pinned commit: `32abe0730d7177269079fdf82b6154763b68ea78`
- Role: external solver templates, numerical evaluators, base generation
  prompts, and PDE task descriptions.
- Prompt and task text are loaded from the user's separate CodePDE checkout
  at runtime; the publishable DiffPDE tree does not redistribute that text
  or CodePDE evaluator and solver source.
- No explicit software license was found at the pinned commit during this
  audit. Users should review CodePDE's terms separately.

## Dream-Coder

- Model and implementation: https://github.com/DreamLM/Dream-Coder
- Role: base masked diffusion code model used to initialize the DiffPDE
  training pipeline. DiffPDE does not redistribute Dream-Coder weights or
  its separate repository.

## PDEBench

- Project: https://github.com/pdebench/PDEBench
- Dataset: https://doi.org/10.18419/darus-2986
- Role: external benchmark data used by the CodePDE numerical evaluators.
  DiffPDE does not redistribute the dataset.
