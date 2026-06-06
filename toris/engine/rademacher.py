"""
TORIS Section 12.1 — Rademacher Exact Surprise Formula.

Implements the convergent series analog of the Rademacher partition formula:

    S(d) = 2π(24d−1)^(−3/4) · Σ_{k=1}^∞ (B_k^F(d)/k) · I_{3/2}(π√(24d−1)/(6k))

where B_k^F(d) is the TORIS Kloosterman sum encoding the relational field structure.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field import Field


@dataclass
class RademacherResult:
    S_exact: float
    error_bound: float
    terms_used: int
    integer_nearness: float


def bessel_I_3_2(x: float) -> float:
    """
    Modified Bessel function I_{3/2}(x) = √(2x/π) · (cosh(x)/x − sinh(x)/x²).

    Analytically implemented per spec; uses asymptotic form for large x.
    """
    if x <= 0:
        return 0.0
    if x > 600.0:
        # Asymptotic: I_ν(x) ~ e^x / √(2πx)
        return math.exp(x) / math.sqrt(2.0 * math.pi * x)
    return math.sqrt(2.0 * x / math.pi) * (math.cosh(x) / x - math.sinh(x) / x ** 2)


def kloosterman_sum(field: "Field", k: int, d: int) -> complex:
    """
    TORIS Kloosterman sum B_k^F(d).

    B_k^F(d) = Σ_{h: gcd(h,k)=1, 1≤h<k} W_F(h,k,d) · exp(2πi·h·d/k)
    W_F(h,k,d) = Σ_{R: depth d} σ(R)·κ(R)·exp(πi·τ_index(R)·h/k)

    For k=1: degenerate sum returns Σ_R σ(R)·κ(R) (the total relator weight).
    """
    rels = field.relators_at_depth(d)
    if not rels:
        return complex(0.0)

    if k == 1:
        return complex(sum(r.sigma * r.kappa for r in rels))

    total = complex(0.0)
    for h in range(1, k):
        if gcd(h, k) != 1:
            continue
        W = complex(0.0)
        for r in rels:
            phase_tau = math.pi * r.tau_index * h / k
            W += r.sigma * r.kappa * cmath.exp(1j * phase_tau)
        phase_d = 2.0 * math.pi * h * d / k
        total += W * cmath.exp(1j * phase_d)
    return total


def rademacher_term(field: "Field", k: int, d: int) -> complex:
    """Single k-th term of the Rademacher series: (B_k^F(d)/k) · I_{3/2}(arg)."""
    B = kloosterman_sum(field, k, d)
    arg = math.pi * math.sqrt(24 * d - 1) / (6 * k)
    I = bessel_I_3_2(arg)
    return (B / k) * I


def integer_nearness(S: float) -> float:
    """Distance |S − round(S)| — a resonance indicator when near 0."""
    return abs(S - round(S))


def _field_constant(field: "Field", d: int) -> float:
    """Conservative C_F for the error bound |S(d)−S_N(d)| < C_F·exp(−π·N·√(2d/3))."""
    rels = field.relators_at_depth(d)
    if not rels:
        return 1.0
    max_weight = max(r.sigma * r.kappa for r in rels)
    return max(1.0, max_weight * math.sqrt(max(d, 1)))


def rademacher_surprise(
    field: "Field",
    d: int,
    N_terms: int = 3,
) -> RademacherResult:
    """
    Compute S(d) via the Rademacher series truncated at N_terms.

    The prefactor 2π(24d−1)^(−3/4) is applied to the real part of the series sum.
    """
    if d < 1:
        raise ValueError(f"depth d must be ≥ 1, got {d}")

    prefactor = 2.0 * math.pi * (24 * d - 1) ** (-3.0 / 4.0)
    series_sum = 0.0
    for k in range(1, N_terms + 1):
        series_sum += rademacher_term(field, k, d).real

    S_exact = prefactor * series_sum
    C_F = _field_constant(field, d)
    # Bound decreases with N: more terms → smaller error
    error_bound = C_F * math.exp(-math.pi * N_terms * math.sqrt(2.0 * d / 3.0))

    return RademacherResult(
        S_exact=S_exact,
        error_bound=error_bound,
        terms_used=N_terms,
        integer_nearness=integer_nearness(S_exact),
    )


def certified_surprise(
    field: "Field",
    d: int,
    precision: int = 8,
) -> tuple[float, float]:
    """
    Return (S, error_bound) with N_terms auto-selected to satisfy the requested
    significant-figure precision via the theoretical bound.

    Falls back to 10 terms when the theoretical bound is too conservative.
    """
    target = 10.0 ** (-precision)
    C_F = _field_constant(field, d)

    for N in range(1, 31):
        bound = C_F * math.exp(-math.pi * N * math.sqrt(2.0 * d / 3.0))
        if bound < target:
            r = rademacher_surprise(field, d, N_terms=N)
            return r.S_exact, r.error_bound

    r = rademacher_surprise(field, d, N_terms=10)
    return r.S_exact, r.error_bound
