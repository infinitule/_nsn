"""
TORIS Section 12.3 — Harmonic Maass Completion of the TASF.

Productive contradictions are poles of the holomorphic surprise density F^+(κ).
The harmonic Maass form theory (Zwegers 2002) supplies the non-holomorphic
correction F^-(κ, κ̄) — the shadow — via the Eichler integral of the shadow
cusp form g_C(z).

Complete TASF:
    ΔS_complete = ΔS_mock + ΔS_shadow

Shadow correction (approximate formula for real κ):
    |ΔS_shadow| ≈ Σ_C |Res[F^+, κ_C]|² · π / (κ_max − κ_C)
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.integrate import quad

if TYPE_CHECKING:
    from toris.field import Contradiction, Field, Goal


@dataclass
class CompleteResult:
    delta_S_mock: float
    delta_S_shadow: float
    delta_S_complete: float
    shadow_fraction: float  # ΔS_shadow / ΔS_complete


def shadow_cusp_form(contradiction: "Contradiction", z: complex) -> complex:
    """
    Shadow cusp form g_C(z) for contradiction C.

        g_C(z) = σ(R_a) · σ(R_b) · exp(2πi · τ_diff(C) · z)
    """
    scale = contradiction.relator_a.sigma * contradiction.relator_b.sigma
    return scale * cmath.exp(2j * math.pi * contradiction.tau_diff * z)


def eichler_integral(
    contradiction: "Contradiction",
    kappa: float,
    kappa_max: float = 1.0,
) -> complex:
    """
    Numerical Eichler integral E_C(κ) = ∫_{−κ̄}^{κ_max} g_C(z)·(z+κ)^{−2} dz.

    For real κ the lower limit is −κ. The integrand has an order-2 pole at
    z = −κ (the lower endpoint); we displace the lower limit by ε = 1e-6.
    """
    eps = 1e-6
    lower = -kappa + eps
    if lower >= kappa_max:
        return complex(0.0)

    tau_diff = contradiction.tau_diff
    scale = contradiction.relator_a.sigma * contradiction.relator_b.sigma

    def integrand_real(z: float) -> float:
        g = scale * cmath.exp(2j * math.pi * tau_diff * z)
        return (g / (z + kappa) ** 2).real

    def integrand_imag(z: float) -> float:
        g = scale * cmath.exp(2j * math.pi * tau_diff * z)
        return (g / (z + kappa) ** 2).imag

    re, _ = quad(integrand_real, lower, kappa_max, limit=200)
    im, _ = quad(integrand_imag, lower, kappa_max, limit=200)
    return complex(re, im)


def shadow_density(
    contradiction: "Contradiction",
    kappa: float,
    kappa_max: float = 1.0,
) -> complex:
    """
    Shadow density F^-_C(κ, κ̄) = Res[F^+(κ), κ_C] · E_C(κ, κ̄).

    The residue at the contradiction pole is approximated as σ_a · σ_b.
    """
    res = contradiction.relator_a.sigma * contradiction.relator_b.sigma
    E = eichler_integral(contradiction, kappa, kappa_max)
    return res * E


def shadow_correction(field: "Field", goal: "Goal") -> float:
    """
    ΔS_shadow = Σ_C |Res[F^+, κ_C]|² · π / (κ_max − κ_C)

    Approximate closed-form per spec §12.3.4.
    Never returns 0 when productive contradictions exist.
    """
    kappa_max = goal.kappa_max
    total = 0.0
    for c in field.productive_contradictions():
        kappa_C = c.kappa_pole
        res = c.relator_a.sigma * c.relator_b.sigma
        denom = max(kappa_max - kappa_C, 1e-3)
        total += res ** 2 * math.pi / denom
    return total


def _mock_surprise(field: "Field", goal: "Goal") -> float:
    """
    ΔS_mock — holomorphic (mock) part of the TASF.

    Approximated as Σ_R σ(R)·κ(R) summed over all relators, weighted by
    the goal quality factor Q(G).
    """
    base = sum(r.sigma * r.kappa for r in field.relators)
    return base * (1.0 + goal.q)


def complete_tasf(field: "Field", goal: "Goal") -> CompleteResult:
    """
    Return the Maass-completed TASF result splitting ΔS into mock and shadow.
    """
    mock = _mock_surprise(field, goal)
    shadow = shadow_correction(field, goal)
    complete = mock + shadow
    fraction = shadow / complete if complete != 0.0 else 0.0
    return CompleteResult(
        delta_S_mock=mock,
        delta_S_shadow=shadow,
        delta_S_complete=complete,
        shadow_fraction=fraction,
    )
