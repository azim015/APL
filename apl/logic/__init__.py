"""Discrete symbolic layer: scenario bank, soft logic, evidence, rule engine."""

from .evidence import EvidenceExtractor, StructuredDescription, TextFeatureSet
from .rule_engine import (
    INSUFFICIENT,
    NO_VIOLATION,
    VIOLATION,
    EngineOutput,
    RuleEngine,
    ScenarioDecision,
)
from .scenario_bank import Predicate, Rule, Scenario, ScenarioBank
from .soft_logic import SoftLogicGate, compatibility_scores, gap_mask

__all__ = [
    "ScenarioBank",
    "Scenario",
    "Predicate",
    "Rule",
    "SoftLogicGate",
    "compatibility_scores",
    "gap_mask",
    "EvidenceExtractor",
    "StructuredDescription",
    "TextFeatureSet",
    "RuleEngine",
    "EngineOutput",
    "ScenarioDecision",
    "VIOLATION",
    "INSUFFICIENT",
    "NO_VIOLATION",
]
