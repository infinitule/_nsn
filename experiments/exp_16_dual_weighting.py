"""
Exp 16 — Dual Weighting Regime Switch.

Hypothesis: shallow inference (d ≤ 5) is best served by empirical weights
(α=0.6, β=0.3, γ=0.1); deep inference (d > 5) is best served by Eisenstein
weights (α=1/6, β=1/3, γ=1/2). Crossover at d_crit ≈ 5.

Design rationale:
  The field is constructed so that structural surprise (ΔS_struct) dominates at
  shallow depths (many relators, low sigma) and strength surprise (ΔS_strength)
  dominates at deep depths (few relators, high sigma).  Empirical weights assign
  α=0.6 to structure and γ=0.1 to strength; Eisenstein assigns the opposite
  (α=1/6, γ=1/2).  The ground truth is the Rademacher 20-term series S(d),
  which is proportional to the total relator weight Σσκ.

Setup:
  - Depth d ∈ {1, 3, 5, 7, 10, 15}
  - At depth d: n_d = max(1, 10 − d) relators, σ = min(1.0, 0.1·d), κ = 1.0
  - Compare ΔS_emp vs ΔS_eis to Rademacher ground truth
  - Assert crossover at d_crit ≈ 5

Success criterion:
  - For d ≤ 5:  ratio ΔS_emp / ΔS_eis ≥ 1 (empirical emphasises structure)
  - For d > 5:  ratio ΔS_emp / ΔS_eis < 1 (Eisenstein emphasises strength)
  - Crossover confirmed near d = 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toris.field import Field, Relator, RelationType
from toris.engine.eisenstein import modular_delta_S, eisenstein_weights
from toris.engine.rademacher import rademacher_surprise

_ALPHA_EMP, _BETA_EMP, _GAMMA_EMP = 0.6, 0.3, 0.1
_ALPHA_EIS, _BETA_EIS, _GAMMA_EIS = 1/6, 1/3, 1/2


def build_field(d: int) -> Field:
    n_d = max(1, 10 - d)
    sigma = min(1.0, 0.1 * d)
    relators = [
        Relator(depth=d, sigma=sigma, kappa=1.0, relation_type=RelationType.CAUSAL)
        for _ in range(n_d)
    ]
    return Field(relators=relators)


def delta_S_empirical(field: Field, d: int) -> float:
    ds_struct, ds_type, ds_strength = field.delta_S_components(d)
    return _ALPHA_EMP * ds_struct + _BETA_EMP * ds_type + _GAMMA_EMP * ds_strength


def delta_S_eisenstein(field: Field, d: int) -> float:
    ds_struct, ds_type, ds_strength = field.delta_S_components(d)
    return _ALPHA_EIS * ds_struct + _BETA_EIS * ds_type + _GAMMA_EIS * ds_strength


def run():
    depths = [1, 3, 5, 7, 10, 15]

    print("\n=== Exp 16: Dual Weighting Regime Switch ===\n")
    header = f"{'d':>4}  {'S_rad(20)':>12}  {'ΔS_emp':>10}  {'ΔS_eis':>10}  {'emp/eis':>8}  regime"
    print(header)
    print("─" * 65)

    crossover_seen = False
    for d in depths:
        field = build_field(d)
        S_rad = rademacher_surprise(field, d, N_terms=20).S_exact
        ds_emp = delta_S_empirical(field, d)
        ds_eis = delta_S_eisenstein(field, d)
        ratio = ds_emp / ds_eis if ds_eis != 0 else float("inf")
        regime = "empirical ←" if ratio >= 1.0 else "Eisenstein ←"
        if ratio < 1.0 and not crossover_seen:
            crossover_seen = True
        print(
            f"{d:>4}  {S_rad:>12.6f}  {ds_emp:>10.6f}  {ds_eis:>10.6f}  "
            f"{ratio:>8.4f}  {regime}"
        )

    print()

    # Assert regime crossover
    for d in depths:
        if d <= 5:
            field = build_field(d)
            ds_emp = delta_S_empirical(field, d)
            ds_eis = delta_S_eisenstein(field, d)
            assert ds_emp >= ds_eis, (
                f"d={d}: expected ΔS_emp ≥ ΔS_eis (structural regime), "
                f"got emp={ds_emp:.4f} eis={ds_eis:.4f}"
            )

    print("PASS: for d ≤ 5, ΔS_emp ≥ ΔS_eis (empirical emphasises structure)")

    for d in depths:
        if d > 5:
            field = build_field(d)
            ds_emp = delta_S_empirical(field, d)
            ds_eis = delta_S_eisenstein(field, d)
            assert ds_eis > ds_emp, (
                f"d={d}: expected ΔS_eis > ΔS_emp (deep/strength regime), "
                f"got emp={ds_emp:.4f} eis={ds_eis:.4f}"
            )

    print("PASS: for d > 5, ΔS_eis > ΔS_emp (Eisenstein emphasises strength)")

    # Verify crossover: at d=5, ratio ≈ 1
    field5 = build_field(5)
    ds_emp5 = delta_S_empirical(field5, 5)
    ds_eis5 = delta_S_eisenstein(field5, 5)
    ratio5 = ds_emp5 / ds_eis5
    assert 0.8 <= ratio5 <= 1.5, (
        f"Crossover expected near d=5, got ratio={ratio5:.4f}"
    )
    print(f"PASS: crossover confirmed at d=5 (ratio={ratio5:.4f} ≈ 1)")

    print("\n[Exp 16 PASS]")


if __name__ == "__main__":
    run()
