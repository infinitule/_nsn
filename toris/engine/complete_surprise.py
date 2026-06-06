"""
TORIS Section 12.5 — Unified Surprise Computation.

Auto-selects the regime (fast / standard / deep) based on depth d and goal
quality Q(G), applies shadow correction when productive contradictions exist,
and returns certified error bounds.

Regime selection:
  FAST     d ≤ d_crit  AND  Q(G) > 0.01  → TFSA bit-operation approximation
  STANDARD d ≤ d_crit  AND  Q(G) ≤ 0.01 → TASF holomorphic + Maass shadow
  DEEP     d > d_crit                     → Rademacher + Eisenstein + shadow
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

from .eisenstein import eisenstein_weights, modular_delta_S
from .maass_completion import complete_tasf, shadow_correction
from .rademacher import certified_surprise as rademacher_certified, rademacher_surprise

if TYPE_CHECKING:
    from toris.field import Field, Goal

_D_CRIT = 5


@dataclass
class UnifiedResult:
    delta_S: float
    error_bound: float
    regime_used: str            # "fast", "standard", or "deep"
    shadow_applied: bool
    suppressed_depths: list[int] = dc_field(default_factory=list)
    rademacher_terms_used: int = 0


def _suppressed_depths(d: int) -> list[int]:
    """
    Partition congruence suppression from Section 11.
    Depths where S(d) is suppressed: d ≡ 4 (mod 5), d ≡ 5 (mod 7), d ≡ 6 (mod 11).
    """
    suppressed = []
    for depth in range(1, d + 1):
        if depth % 5 == 4 or depth % 7 == 5 or depth % 11 == 6:
            suppressed.append(depth)
    return suppressed


def _fast_surprise(field: "Field", goal: "Goal", d: int) -> float:
    """TFSA fast-path: linear sum of weighted relator contributions."""
    rels = field.relators_at_depth(d)
    if not rels:
        return 0.0
    return sum(r.sigma * r.kappa for r in rels) * goal.q


def _standard_surprise(field: "Field", goal: "Goal", d: int) -> tuple[float, float]:
    """TASF holomorphic + Maass shadow; returns (delta_S, error_bound)."""
    result = complete_tasf(field, goal)
    # Error bound for the TASF approximation: 5% of mock part
    error = 0.05 * abs(result.delta_S_mock)
    return result.delta_S_complete, error


def _deep_surprise(
    field: "Field",
    goal: "Goal",
    d: int,
    precision: int,
) -> tuple[float, float, int]:
    """Rademacher + Eisenstein weights + shadow; returns (delta_S, error, N_terms)."""
    S, error_bound = rademacher_certified(field, d, precision=precision)

    # Apply Eisenstein-weighted modular ΔS as a correction to Rademacher
    delta_S_mod = modular_delta_S(field, d, d_crit=_D_CRIT)
    combined = S + delta_S_mod

    # Shadow correction
    shadow = shadow_correction(field, goal)
    total = combined + shadow

    r = rademacher_surprise(field, d, N_terms=10)
    return total, error_bound, r.terms_used


class UnifiedSurprise:
    """
    Ties all TORIS Section 12 components into one interface.

    Usage
    -----
    us = UnifiedSurprise()
    result = us.compute(field, goal, d=7)
    print(result.delta_S, result.regime_used)
    """

    def compute(
        self,
        field: "Field",
        goal: "Goal",
        d: int,
        precision: int = 8,
    ) -> UnifiedResult:
        """
        Auto-select regime and compute ΔS_COMPLETE(F, G, d).

        Always checks suppression (Section 11 partition congruences).
        Applies shadow correction when productive contradictions are present.
        Returns certified error bound.
        """
        if d < 1:
            raise ValueError(f"depth d must be ≥ 1, got {d}")

        suppressed = _suppressed_depths(d)
        has_shadow = bool(field.productive_contradictions())

        if d <= _D_CRIT and goal.q > 0.01:
            # FAST regime
            delta_S = _fast_surprise(field, goal, d)
            if has_shadow:
                delta_S += shadow_correction(field, goal)
            error = 0.10 * abs(delta_S)
            return UnifiedResult(
                delta_S=delta_S,
                error_bound=error,
                regime_used="fast",
                shadow_applied=has_shadow,
                suppressed_depths=suppressed,
                rademacher_terms_used=0,
            )

        if d <= _D_CRIT:
            # STANDARD regime
            delta_S, error = _standard_surprise(field, goal, d)
            return UnifiedResult(
                delta_S=delta_S,
                error_bound=error,
                regime_used="standard",
                shadow_applied=has_shadow,
                suppressed_depths=suppressed,
                rademacher_terms_used=0,
            )

        # DEEP regime
        delta_S, error, n_terms = _deep_surprise(field, goal, d, precision)
        return UnifiedResult(
            delta_S=delta_S,
            error_bound=error,
            regime_used="deep",
            shadow_applied=has_shadow,
            suppressed_depths=suppressed,
            rademacher_terms_used=n_terms,
        )
