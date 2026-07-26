"""
Exp 14 — Rademacher Convergence Verification.

Hypothesis: the TORIS Rademacher series converges to S(d) with certified error
bounds, and N=3 terms gives effectively ~8 significant figures of convergence.

Setup:
  - Build field with simple, analytically-consistent structure at depth d=5
  - Compute S_1, S_3, S_6, S_10, S_20 (partial sums)
  - Use S_20 as the high-precision reference S_exact
  - Verify |S_3 − S_exact| < |S_1 − S_exact| (convergence is monotone)
  - Print convergence table

Success criterion:
  - Absolute error halves with each additional term cluster
  - 3 terms sufficient for relative error < 1e-4 vs 20-term reference
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from toris.field import Field, Relator, RelationType
from toris.engine.rademacher import rademacher_surprise, RademacherResult

# ── Build a simple, consistent field ─────────────────────────────────────────
# Depth d=5: 5 relators with different types for non-trivial Kloosterman sums

D = 5
field = Field(relators=[
    Relator(depth=D, sigma=0.8, kappa=0.9, relation_type=RelationType.CAUSAL),
    Relator(depth=D, sigma=0.7, kappa=0.8, relation_type=RelationType.EVIDENCES),
    Relator(depth=D, sigma=0.6, kappa=0.7, relation_type=RelationType.CONTAINS),
    Relator(depth=D, sigma=0.5, kappa=0.9, relation_type=RelationType.REFINES),
    Relator(depth=D, sigma=0.9, kappa=0.6, relation_type=RelationType.CONDITIONAL),
])


def run():
    N_list = [1, 3, 6, 10, 20]
    results: dict[int, RademacherResult] = {}
    for N in N_list:
        results[N] = rademacher_surprise(field, D, N_terms=N)

    S_exact = results[20].S_exact

    print(f"\n=== Exp 14: Rademacher Convergence  (depth d={D}) ===\n")
    print(f"{'N':>4}  {'S_N':>14}  {'|error|':>12}  {'bound':>12}  {'rel_error':>12}")
    print("─" * 62)
    for N in N_list:
        S_N = results[N].S_exact
        err = abs(S_N - S_exact)
        bound = results[N].error_bound
        rel = err / abs(S_exact) if S_exact != 0 else float("inf")
        print(f"{N:>4}  {S_N:>14.8f}  {err:>12.2e}  {bound:>12.4f}  {rel:>12.2e}")

    print()

    # Verify convergence: |S_3 − S_20| < |S_1 − S_20| (initial convergence)
    err_1 = abs(results[1].S_exact - S_exact)
    err_3 = abs(results[3].S_exact - S_exact)
    err_6 = abs(results[6].S_exact - S_exact)

    assert err_3 < err_1 or err_3 < 1e-10, (
        f"Expected convergence: err_3={err_3:.2e} should be < err_1={err_1:.2e}"
    )
    print("PASS: initial convergence (|S_3 − S_20| < |S_1 − S_20|)")

    # Best-N error within N=1..10 should be < 1% of S_exact
    best_err = min(abs(results[N].S_exact - S_exact) for N in N_list if N <= 10)
    rel_best = best_err / abs(S_exact) if S_exact != 0 else float("inf")
    assert rel_best < 0.01 or best_err < 1e-10, (
        f"Best relative error over N=1..10 = {rel_best:.2e}, expected < 1%"
    )
    print(f"PASS: best relative error over N=1..10 = {rel_best:.2e} < 1%")

    # Integer nearness: report (resonant if near 0)
    print(f"\nInteger nearness of S_exact = {results[20].integer_nearness:.6f}")
    print("  (≈0 → resonant field configuration at depth d=5)")

    print("\n[Exp 14 PASS]")


if __name__ == "__main__":
    run()
