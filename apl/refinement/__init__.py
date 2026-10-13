"""Gap-conditioned iterative refinement (Sec. V-F, Algorithm 1)."""

from .loop import RefinementResult, RefinementTrace, run_refinement_loop
from .prompt_refiner import GapPrompt, PromptRefiner

__all__ = [
    "PromptRefiner",
    "GapPrompt",
    "run_refinement_loop",
    "RefinementResult",
    "RefinementTrace",
]
