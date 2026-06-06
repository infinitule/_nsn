"""
Exp 15 — Shadow Correction Magnitude.

Hypothesis: fields with productive contradictions have measurably wrong ΔS
without shadow correction; the Maass shadow correction fixes this.

Setup:
  - Build field with 1 PRODUCTIVE contradiction (σ_a=0.7, σ_b=0.6, κ=0.5)
  - Compute ΔS_mock (no shadow)
  - Compute ΔS_shadow via approximate closed-form
  - Compute ΔS_complete = ΔS_mock + ΔS_shadow
  - Print the shadow fraction

Success criterion:
  - |ΔS_shadow| > 1.0  (shadow is significant, not negligible)
  - ΔS_complete > ΔS_mock  (shadow always adds to total)
  - shadow_fraction > 0.30  (at least 30% of surprise is non-holomorphic)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from toris.field import (
    Contradiction, Field, Goal, Relator, RelationType,
)
from toris.engine.maass_completion import (
    shadow_correction, complete_tasf, shadow_cusp_form,
)

# ── Build field with one productive contradiction ─────────────────────────────
# Both relators live at depth 3; σ_a=0.7, σ_b=0.6, κ=0.5
R_a = Relator(depth=3, sigma=0.7, kappa=0.5, relation_type=RelationType.CAUSAL)
R_b = Relator(depth=3, sigma=0.6, kappa=0.5, relation_type=RelationType.CONTRADICTS)
contradiction = Contradiction(relator_a=R_a, relator_b=R_b, productive=True)

# Also some background relators for a realistic field
field = Field(
    relators=[
        R_a, R_b,
        Relator(depth=1, sigma=0.5, kappa=0.8, relation_type=RelationType.CAUSAL),
        Relator(depth=2, sigma=0.6, kappa=0.7, relation_type=RelationType.ENABLES),
    ],
    contradictions=[contradiction],
)

goal = Goal(q=0.5, kappa_max=1.0)


def run():
    print("\n=== Exp 15: Shadow Correction Magnitude ===\n")

    # Shadow cusp form at z=0.1 (sample point)
    g = shadow_cusp_form(contradiction, z=0.1)
    print(f"Shadow cusp form g_C(0.1) = {g:.4f}")

    # Complete TASF
    result = complete_tasf(field, goal)

    print(f"\nΔS_mock     = {result.delta_S_mock:.4f}")
    print(f"ΔS_shadow   = {result.delta_S_shadow:.4f}")
    print(f"ΔS_complete = {result.delta_S_complete:.4f}")
    print(f"Shadow fraction = {result.shadow_fraction:.2%}")

    # Approximate theoretical prediction from spec §12.3.4:
    # |ΔS_shadow| ≈ 2π · σ_a · σ_b (for κ_C=0.5, κ_max=1.0)
    theoretical = 2.0 * math.pi * R_a.sigma * R_b.sigma
    print(f"\nTheoretical |ΔS_shadow| ≈ 2π·σ_a·σ_b = {theoretical:.4f}")

    # Assertions
    assert result.delta_S_shadow > 1.0, (
        f"Shadow should be > 1.0, got {result.delta_S_shadow:.4f}"
    )
    print("\nPASS: |ΔS_shadow| > 1.0")

    assert result.delta_S_complete > result.delta_S_mock, (
        "ΔS_complete must exceed ΔS_mock"
    )
    print("PASS: ΔS_complete > ΔS_mock (shadow adds to total)")

    assert result.shadow_fraction > 0.30, (
        f"shadow_fraction = {result.shadow_fraction:.2%}, expected > 30%"
    )
    print(f"PASS: shadow_fraction = {result.shadow_fraction:.2%} > 30%")

    # Internal consistency: complete = mock + shadow
    assert abs(result.delta_S_complete - (result.delta_S_mock + result.delta_S_shadow)) < 1e-10
    print("PASS: ΔS_complete = ΔS_mock + ΔS_shadow (internal consistency)")

    print("\n[Exp 15 PASS]")


if __name__ == "__main__":
    run()
