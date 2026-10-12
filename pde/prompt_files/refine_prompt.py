"""DiffPDE-authored localized repair and traversal prompt extensions.

Shared CodePDE wording in the optional extensions is composed from the
external pinned checkout at runtime.
"""

from pde.prompt_files.codepde_loader import get_generation_prompts

refine_prompt = '''
You are an intelligent AI assistant for code infilling and local repair in scientific computing and numerical solver code.
Your role is to carefully complete and fix only the missing code pieces in an existing solution, instead of rewriting the whole answer from scratch.

You will be given:
- An existing answer that already contains natural-language reasoning and one or more ```python``` code blocks.
- Inside these code blocks, some contiguous spans are replaced by MASK tokens, indicating holes that must be filled.

Your tasks are:
1. Read and understand the existing answer, including the intended data shapes, control flow, and interfaces (such as solver functions, helper utilities, and return values).
2. For each MASK span, infill the missing Python code so that:
   - The resulting code is syntactically correct and runs without obvious errors.
   - The logic, structure, and intent of the surrounding implementation are preserved.
   - Function signatures, input/output shapes, and public interfaces remain unchanged.
3. Do NOT rewrite or restructure the parts of the answer that are not masked:
   - Do not change existing variable names, function names, comments, or explanations outside MASK spans.
   - Do not introduce completely new solution strategies that conflict with the existing design, unless it is clearly required to make the code consistent and runnable.
   - Match the existing coding style (indentation, naming conventions, use of NumPy/PyTorch/JAX, etc.).

When a MASK span is longer than what is actually needed:
- Stop once the necessary code for a correct and coherent implementation is complete.
- Conceptually treat the remaining masked positions as padding; do not add extra, unnecessary statements, prints, or comments just to fill space.

Your output should be the same reasoning and code as the input, except that every MASK span has been replaced by your completed code while all non-masked content remains unchanged.
'''

def get_refine_system_prompt(config=None) -> str:
    """Compose the original refine system prompt using CodePDE wording."""
    base_system_prompt, _ = get_generation_prompts(config)
    shared_opening = base_system_prompt.split(".\n", 1)[0]
    return shared_opening + ', with a special focus on code infilling.\nYour goal is to improve and complete existing PDE-solving code by carefully filling in missing pieces while preserving correct logic, structure, and intent of the surrounding solution.\nYou are given a PDE problem description and an existing answer that already contains reasoning and ```python``` code blocks, where some code spans are replaced by MASK tokens.\nYour job is to infill only these masked code spans, without rewriting the rest of the answer or generating a completely new solution from scratch.\nIf a MASK span is longer than the code that is actually needed, stop once the necessary code is complete and conceptually treat the remaining masked positions as padding (do not add extra code or comments there).\n'

def get_traversal_problem_prompt(config=None) -> str:
    """Compose the original traversal prompt without bundling CodePDE prose."""
    _, generation_prompt = get_generation_prompts(config)
    start = generation_prompt.index(". Your python implementation")
    end = generation_prompt.index("\n```\n", start) + len("\n```\n")
    shared_footer = generation_prompt[start:end]
    return '\nYour task is to solve a partial differential equation (PDE) using Python in batch mode.\n{pde_description}\n\nYou will LEARN from existing materials rather than merely "improve" them. Carefully study both the Primary Target and the Supplementary References below, extract the core ideas, and synthesize a solution tailored to this problem.\n\nPrimary Target:\n{target_code_samples}\n\nSupplementary References:\n{supplementary_code_samples}\n\nYour tasks are:\n1. Learn from the Primary Target and Supplementary References by extracting the core techniques, stability/accuracy tricks, performance optimizations, and design choices. For each key idea, briefly note when it is effective and any trade-offs. Provide a concise, actionable list of possible solution strategies you could adopt for this PDE.\n2. Plan for Implementation:\n\t- Present a concrete plan tailored to this PDE, similar in style to "Plan for Implementation". Include at least: spatial discretization, time integration scheme, boundary conditions handling, stabilization/dealiasing or filtering if relevant, batch processing strategy, mapping to requested output times (t_coordinate), parameter choices (e.g., dt/CFL/viscosity handling), and any device/precision considerations.\n3. Implementation:\n\t- Provide a clean Python implementation for the solver function (you must not change its signature). You may implement small auxiliary functions or add internal arguments to modularize the code.\n\t- You can use NumPy or PyTorch/JAX with GPU acceleration.\n\t- Keep the code simple, robust, and efficient; aim for low nRMSE and a runtime under 600s for typical test sizes.\n\t- Include a few lightweight print statements to aid debugging (avoid verbose logs).\n\nYour generated code will be executed and evaluated. Ensure the solver runs correctly and efficiently.\n\nYour output must follow this exact structure:\n1. Learned solution strategies (from Primary Target and Supplementary References).\n2. A plan on the implementation idea.\n3' + shared_footer + ''
