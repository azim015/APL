"""APL - Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models.

Reference implementation of the framework described in
*"Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models"*
(IEEE Transactions on Big Data), by A. Rezaei, M. Haghparast, A. Patooghy and
M. Sookhak.

Typical use::

    from apl import APLConfig, APLModel

    model = APLModel(APLConfig())          # scenario-bank categories by default
    out = model.infer(images[:1])          # Algorithm 1 + rule-based decision
    print(out.description.text)
    print(out.engine.report())
"""

from .config import APLConfig, parse_overrides
from .describe import DescriptionComposer, SceneDescription, sanitize
from .model import APLModel, APLOutput, InferenceOutput

__version__ = "1.0.0"

__all__ = [
    "APLConfig",
    "APLModel",
    "APLOutput",
    "InferenceOutput",
    "DescriptionComposer",
    "SceneDescription",
    "sanitize",
    "parse_overrides",
    "__version__",
]
