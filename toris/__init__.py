"""
TORIS — Topological Relational Inference System.

Section 12: Exact Surprise, Eisenstein Weights, Maass Completion.
"""

from .field import Contradiction, Field, Goal, RelationType, Relator
from .engine.rademacher import (
    RademacherResult,
    bessel_I_3_2,
    certified_surprise,
    integer_nearness,
    kloosterman_sum,
    rademacher_surprise,
    rademacher_term,
)
from .engine.eisenstein import (
    P_series,
    Q_series,
    R_series,
    eisenstein_weights,
    modular_delta_S,
    tau_congruence_check,
    tau_function,
)
from .engine.maass_completion import (
    CompleteResult,
    complete_tasf,
    eichler_integral,
    shadow_correction,
    shadow_cusp_form,
    shadow_density,
)
from .engine.complete_surprise import UnifiedResult, UnifiedSurprise

__all__ = [
    # Field types
    "Contradiction", "Field", "Goal", "RelationType", "Relator",
    # Rademacher
    "RademacherResult", "bessel_I_3_2", "certified_surprise",
    "integer_nearness", "kloosterman_sum", "rademacher_surprise", "rademacher_term",
    # Eisenstein
    "P_series", "Q_series", "R_series",
    "eisenstein_weights", "modular_delta_S", "tau_congruence_check", "tau_function",
    # Maass
    "CompleteResult", "complete_tasf", "eichler_integral",
    "shadow_correction", "shadow_cusp_form", "shadow_density",
    # Unified
    "UnifiedResult", "UnifiedSurprise",
]
__version__ = "0.1.0"
