"""
TORIS field types — Relator, Contradiction, Field, Goal.

Used by all Section 12 engine modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import IntEnum


class RelationType(IntEnum):
    CAUSAL = 1
    CONDITIONAL = 2
    CONTRADICTS = 3
    CONTAINS = 4
    ENABLES = 5
    VIOLATES = 6
    ANALOGOUS = 7
    REFINES = 8
    TEMPORAL_BEFORE = 9
    EVIDENCES = 10
    NEGATES = 11
    INSTANTIATES = 12


@dataclass
class Relator:
    depth: int
    sigma: float  # strength ∈ [0, 1]
    kappa: float  # connectivity ∈ [0, 1]
    relation_type: RelationType = RelationType.CAUSAL

    @property
    def tau_index(self) -> int:
        return int(self.relation_type)


@dataclass
class Contradiction:
    relator_a: Relator
    relator_b: Relator
    productive: bool = True

    @property
    def tau_diff(self) -> int:
        return abs(self.relator_a.tau_index - self.relator_b.tau_index)

    @property
    def kappa_pole(self) -> float:
        return (self.relator_a.kappa + self.relator_b.kappa) / 2


@dataclass
class Field:
    relators: list[Relator] = dc_field(default_factory=list)
    contradictions: list[Contradiction] = dc_field(default_factory=list)

    def relators_at_depth(self, d: int) -> list[Relator]:
        return [r for r in self.relators if r.depth == d]

    def productive_contradictions(self) -> list[Contradiction]:
        return [c for c in self.contradictions if c.productive]

    def delta_S_components(self, d: int) -> tuple[float, float, float]:
        """Returns (delta_S_struct, delta_S_type, delta_S_strength)."""
        rels = self.relators_at_depth(d)
        if not rels:
            return 0.0, 0.0, 0.0
        n = len(rels)
        delta_S_struct = n * 0.1
        delta_S_type = sum(r.tau_index / 12.0 for r in rels) / n
        delta_S_strength = sum(r.sigma for r in rels) / n
        return delta_S_struct, delta_S_type, delta_S_strength


@dataclass
class Goal:
    q: float = 0.5           # quality metric Q(G) ∈ [0, 1]
    kappa_max: float = 1.0   # maximum kappa value for contour integrals
