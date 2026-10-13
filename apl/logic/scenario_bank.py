"""DMV-aligned violation scenario bank (Sec. IV, Sec. V-E).

A scenario is the triple ``s_j = (nu_j, E_j, r_j)``:

``nu_j``  violation type,
``E_j``   the set of evidentiary predicates required to assess it,
``r_j``   a deterministic rule mapping evidence assignments to a decision.

The bank is *data*, not a neural module: it is compiled from state law, federal
law and driver-handbook material into machine-checkable predicates.  The
default bank ships ``M = 12`` scenarios (Table I).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_BANK = (
    Path(__file__).resolve().parents[2] / "data" / "scenarios" / "dmv_scenario_bank.json"
)


@dataclass(frozen=True)
class Predicate:
    """One evidentiary predicate ``e_jq`` with its privacy-neutral phrasing."""

    id: str
    text: str

    def question(self) -> str:
        return f"Is it visible in the scene that {self.text}?"


@dataclass
class Rule:
    """Deterministic rule ``r_j`` over predicate assignments."""

    type: str = "all"
    required: List[str] = field(default_factory=list)
    any_of: List[str] = field(default_factory=list)

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        required_ok = all(assignment.get(p, False) for p in self.required)
        if self.type == "all":
            return required_ok
        if self.type == "any":
            pool = self.any_of or self.required
            return any(assignment.get(p, False) for p in pool)
        if self.type == "all_and_any":
            return required_ok and any(assignment.get(p, False) for p in self.any_of)
        raise ValueError(f"unsupported rule type: {self.type}")


@dataclass
class Scenario:
    id: str
    violation_type: str
    predicates: List[Predicate]
    rule: Rule
    legal_reference: str = ""

    @property
    def num_predicates(self) -> int:
        return len(self.predicates)

    @property
    def weights(self) -> List[float]:
        """``omega_jq = 1 / Q_j`` - the uniform baseline of Table I."""
        q = max(1, self.num_predicates)
        return [1.0 / q] * self.num_predicates

    def predicate_ids(self) -> List[str]:
        return [p.id for p in self.predicates]


class ScenarioBank:
    """Container over scenarios plus the global predicate inventory."""

    def __init__(self, scenarios: Sequence[Scenario], meta: Optional[Dict] = None):
        self.scenarios: List[Scenario] = list(scenarios)
        self.meta = meta or {}
        inventory: Dict[str, Predicate] = {}
        for scenario in self.scenarios:
            for predicate in scenario.predicates:
                inventory.setdefault(predicate.id, predicate)
        self.predicates: List[Predicate] = list(inventory.values())
        self.predicate_index: Dict[str, int] = {
            p.id: i for i, p in enumerate(self.predicates)
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None) -> "ScenarioBank":
        path = Path(path) if path else DEFAULT_BANK
        if not path.is_absolute():
            candidate = Path.cwd() / path
            path = candidate if candidate.exists() else DEFAULT_BANK
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict) -> "ScenarioBank":
        scenarios = []
        for item in raw.get("scenarios", []):
            predicates = [
                Predicate(id=p["id"], text=p["text"]) for p in item["predicates"]
            ]
            rule_raw = dict(item.get("rule", {}))
            rule = Rule(
                type=rule_raw.get("type", "all"),
                required=list(rule_raw.get("required", [p.id for p in predicates])),
                any_of=list(rule_raw.get("any_of", [])),
            )
            scenarios.append(
                Scenario(
                    id=item["id"],
                    violation_type=item["violation_type"],
                    predicates=predicates,
                    rule=rule,
                    legal_reference=item.get("legal_reference", ""),
                )
            )
        return cls(scenarios, raw.get("meta"))

    def to_dict(self) -> Dict:
        return {
            "meta": self.meta,
            "scenarios": [
                {
                    "id": s.id,
                    "violation_type": s.violation_type,
                    "legal_reference": s.legal_reference,
                    "predicates": [{"id": p.id, "text": p.text} for p in s.predicates],
                    "rule": {
                        "type": s.rule.type,
                        "required": s.rule.required,
                        "any_of": s.rule.any_of,
                    },
                }
                for s in self.scenarios
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self) -> Iterable[Scenario]:
        return iter(self.scenarios)

    def __getitem__(self, key) -> Scenario:
        if isinstance(key, int):
            return self.scenarios[key]
        for scenario in self.scenarios:
            if scenario.id == key or scenario.violation_type == key:
                return scenario
        raise KeyError(key)

    @property
    def num_predicates(self) -> int:
        return len(self.predicates)

    def predicate_texts(self) -> List[str]:
        return [p.text for p in self.predicates]

    def violation_types(self) -> List[str]:
        return [s.violation_type for s in self.scenarios]

    def semantic_categories(self) -> List[str]:
        """Category names ``C`` used as the rows of ``w``.

        The evidentiary predicates *are* the semantic categories of the ITS
        task: every row of the generated text features is therefore directly
        comparable with one predicate of the scenario bank, which is what makes
        the continuous features addressable by the discrete rule engine.
        """
        return [p.text for p in self.predicates]

    def scenario_predicate_matrix(self):
        """``[M, P]`` membership matrix with the normalised weights ``omega_jq``."""
        import torch

        matrix = torch.zeros(len(self.scenarios), self.num_predicates)
        for j, scenario in enumerate(self.scenarios):
            for predicate, weight in zip(scenario.predicates, scenario.weights):
                matrix[j, self.predicate_index[predicate.id]] = weight
        return matrix
