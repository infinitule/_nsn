"""
TORIS Section 12.2 — Eisenstein Weights for ΔS Components.

The three Ramanujan-Eisenstein series P (weight 2), Q (weight 4), R (weight 6)
ground the dual-weighting theorem: empirical weights (0.6, 0.3, 0.1) dominate
at shallow depth d ≤ d_crit; Eisenstein weights (1/6, 1/3, 1/2) take over at
deep depth d > d_crit (default d_crit = 5).
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field import Field

# Empirical weights (Section 9–11, MATH_SPEC §3.2)
_ALPHA_EMP, _BETA_EMP, _GAMMA_EMP = 0.6, 0.3, 0.1

# Eisenstein weights from modular form weight ratios 2:4:6 = 1/6 : 1/3 : 1/2
_ALPHA_EIS = 2 / 12   # = 1/6
_BETA_EIS  = 4 / 12   # = 1/3
_GAMMA_EIS = 6 / 12   # = 1/2


def P_series(q: float, N_terms: int = 50) -> float:
    """
    Eisenstein series P(q) of weight 2.

        P(q) = 1 − 24 · Σ_{k=1}^{N} k · q^k / (1 − q^k)
    """
    if abs(q) >= 1.0:
        raise ValueError("|q| must be < 1")
    total = 0.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        total += k * qk / (1.0 - qk)
    return 1.0 - 24.0 * total


def Q_series(q: float, N_terms: int = 50) -> float:
    """
    Eisenstein series Q(q) of weight 4.

        Q(q) = 1 + 240 · Σ_{k=1}^{N} k³ · q^k / (1 − q^k)
    """
    if abs(q) >= 1.0:
        raise ValueError("|q| must be < 1")
    total = 0.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        total += k ** 3 * qk / (1.0 - qk)
    return 1.0 + 240.0 * total


def R_series(q: float, N_terms: int = 50) -> float:
    """
    Eisenstein series R(q) of weight 6.

        R(q) = 1 − 504 · Σ_{k=1}^{N} k⁵ · q^k / (1 − q^k)
    """
    if abs(q) >= 1.0:
        raise ValueError("|q| must be < 1")
    total = 0.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        total += k ** 5 * qk / (1.0 - qk)
    return 1.0 - 504.0 * total


def eisenstein_weights(d: int, d_crit: int = 5) -> tuple[float, float, float]:
    """
    Return (alpha, beta, gamma) for ΔS = α·ΔS_struct + β·ΔS_type + γ·ΔS_strength.

    d ≤ d_crit → empirical (0.6, 0.3, 0.1)
    d > d_crit → Eisenstein (1/6, 1/3, 1/2)
    """
    if d <= d_crit:
        return _ALPHA_EMP, _BETA_EMP, _GAMMA_EMP
    return _ALPHA_EIS, _BETA_EIS, _GAMMA_EIS


def modular_delta_S(field: "Field", d: int, d_crit: int = 5) -> float:
    """ΔS(d) with depth-appropriate Eisenstein or empirical weights."""
    alpha, beta, gamma = eisenstein_weights(d, d_crit)
    ds_struct, ds_type, ds_strength = field.delta_S_components(d)
    return alpha * ds_struct + beta * ds_type + gamma * ds_strength


def tau_function(field: "Field", d: int) -> complex:
    """
    TORIS tau function τ_F(d) — weight-12 analog of Ramanujan's τ(n).

        τ_F(d) = Σ_{R: depth d} σ(R)^5 · κ(R)^7 · exp(2πi · τ_index(R) / 12)
    """
    total = complex(0.0)
    for r in field.relators_at_depth(d):
        phase = 2.0 * math.pi * r.tau_index / 12.0
        total += r.sigma ** 5 * r.kappa ** 7 * cmath.exp(1j * phase)
    return total


def tau_congruence_check(field: "Field", d: int, S_d: float) -> bool:
    """
    Check S(d) ≡ 11·τ_F(d) (mod 13) when d ≡ 6 (mod 13).

    Applies the congruence to integer parts (round) of the real quantities.
    Returns True if the congruence holds or is not applicable.
    """
    if d % 13 != 6:
        return True  # congruence only applies at d ≡ 6 (mod 13)
    tau_F = tau_function(field, d)
    lhs = round(S_d) % 13
    rhs = (11 * round(abs(tau_F))) % 13
    return lhs == rhs
