"""
APL Framework — Models Package

Implements the five decoupled modules from the paper:
    1. VisionLanguageEncoder
    2. AdaptivePromptLearning (APL) with cross-attention Feature Generator
    3. EvidenceExtractor
    4. ScenarioBankMatcher
    5. RuleEngine
"""

from .vision_language_encoder import VisionLanguageEncoder
from .feature_generator import FeatureGenerator
from .adaptive_prompt_learning import AdaptivePromptLearning
from .evidence_extractor import EvidenceExtractor
from .scenario_bank import ScenarioBankMatcher, DMVScenario
from .rule_engine import RuleEngine

__all__ = [
    "VisionLanguageEncoder",
    "FeatureGenerator",
    "AdaptivePromptLearning",
    "EvidenceExtractor",
    "ScenarioBankMatcher",
    "DMVScenario",
    "RuleEngine",
]
