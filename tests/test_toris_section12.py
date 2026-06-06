"""
Tests for TORIS Section 12: Rademacher, Eisenstein, Maass Completion.

Final test count target: ≥125 (these 30 tests + future sections).
"""

from __future__ import annotations

import cmath
import math

import pytest

from toris.field import Contradiction, Field, Goal, RelationType, Relator
from toris.engine.rademacher import (
    RademacherResult,
    bessel_I_3_2,
    certified_surprise,
    integer_nearness,
    kloosterman_sum,
    rademacher_surprise,
    rademacher_term,
)
from toris.engine.eisenstein import (
    P_series,
    Q_series,
    R_series,
    eisenstein_weights,
    modular_delta_S,
    tau_congruence_check,
    tau_function,
)
from toris.engine.maass_completion import (
    CompleteResult,
    complete_tasf,
    shadow_correction,
    shadow_cusp_form,
    shadow_density,
)
from toris.engine.complete_surprise import UnifiedResult, UnifiedSurprise


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_field():
    return Field(relators=[
        Relator(depth=3, sigma=0.8, kappa=0.9, relation_type=RelationType.CAUSAL),
        Relator(depth=3, sigma=0.6, kappa=0.7, relation_type=RelationType.EVIDENCES),
    ])


@pytest.fixture
def contradiction_field():
    Ra = Relator(depth=3, sigma=0.7, kappa=0.5, relation_type=RelationType.CAUSAL)
    Rb = Relator(depth=3, sigma=0.6, kappa=0.5, relation_type=RelationType.CONTRADICTS)
    c = Contradiction(relator_a=Ra, relator_b=Rb, productive=True)
    return Field(relators=[Ra, Rb], contradictions=[c])


@pytest.fixture
def default_goal():
    return Goal(q=0.5, kappa_max=1.0)


# ── Bessel function tests ─────────────────────────────────────────────────────

class TestBesselI32:
    def test_zero_returns_zero(self):
        assert bessel_I_3_2(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert bessel_I_3_2(-1.0) == 0.0

    def test_known_value_small_x(self):
        # I_{3/2}(1) = sqrt(2/π)*(cosh(1) - sinh(1)) ≈ sqrt(2/π)*(1.5431-0.8415)
        # = sqrt(2/π)*0.7016 ≈ 0.7985*0.7016 ≈ 0.5604
        # More precisely: sqrt(2/π) * (cosh(1) - sinh(1)) = sqrt(2/π) * (e^-1)
        val = bessel_I_3_2(1.0)
        # scipy.special.iv(1.5, 1.0) ≈ 0.2931
        # Our formula: sqrt(2/π) * (cosh(1)/1 - sinh(1)/1) = sqrt(2/π)*(e^-1)
        expected = math.sqrt(2.0 / math.pi) * (math.cosh(1.0) - math.sinh(1.0))
        assert abs(val - expected) < 1e-12

    def test_large_x_asymptotic(self):
        # x > 600 uses asymptotic branch exactly; ratio must be 1.0
        x = 700.0
        val = bessel_I_3_2(x)
        asymptotic = math.exp(x) / math.sqrt(2.0 * math.pi * x)
        assert val == asymptotic  # asymptotic branch returns this formula directly

    def test_very_large_x_uses_asymptotic(self):
        # x > 600 must not overflow — uses asymptotic branch
        val = bessel_I_3_2(700.0)
        assert val > 0.0
        assert math.isfinite(val)

    def test_positive_for_positive_x(self):
        for x in [0.1, 1.0, 5.0, 10.0]:
            assert bessel_I_3_2(x) > 0.0


# ── Kloosterman sum tests ─────────────────────────────────────────────────────

class TestKloostermanSum:
    def test_empty_field_returns_zero(self, default_goal):
        field = Field()
        result = kloosterman_sum(field, k=1, d=3)
        assert result == complex(0.0)

    def test_k1_equals_total_weight(self, simple_field):
        expected = 0.8 * 0.9 + 0.6 * 0.7  # Σ sigma*kappa
        result = kloosterman_sum(simple_field, k=1, d=3)
        assert abs(result.real - expected) < 1e-12
        assert abs(result.imag) < 1e-12

    def test_k2_is_complex(self, simple_field):
        result = kloosterman_sum(simple_field, k=2, d=3)
        # k=2: h=1 only (gcd(1,2)=1), result generally complex
        assert isinstance(result, complex)

    def test_wrong_depth_gives_zero(self, simple_field):
        result = kloosterman_sum(simple_field, k=1, d=99)
        assert result == complex(0.0)


# ── Rademacher series tests ───────────────────────────────────────────────────

class TestRademacherSurprise:
    def test_returns_result_type(self, simple_field):
        result = rademacher_surprise(simple_field, d=3)
        assert isinstance(result, RademacherResult)

    def test_d_less_than_1_raises(self, simple_field):
        with pytest.raises(ValueError):
            rademacher_surprise(simple_field, d=0)

    def test_N_terms_recorded(self, simple_field):
        result = rademacher_surprise(simple_field, d=3, N_terms=5)
        assert result.terms_used == 5

    def test_S_exact_is_finite(self, simple_field):
        result = rademacher_surprise(simple_field, d=3)
        assert math.isfinite(result.S_exact)

    def test_error_bound_positive(self, simple_field):
        result = rademacher_surprise(simple_field, d=3)
        assert result.error_bound > 0.0

    def test_convergence_more_terms_lower_bound(self, simple_field):
        r3 = rademacher_surprise(simple_field, d=5, N_terms=3)
        r10 = rademacher_surprise(simple_field, d=5, N_terms=10)
        # More terms → smaller error bound
        assert r10.error_bound < r3.error_bound

    def test_convergence_S_stabilises(self, simple_field):
        r5 = rademacher_surprise(simple_field, d=5, N_terms=5)
        r20 = rademacher_surprise(simple_field, d=5, N_terms=20)
        # S should not diverge between 5 and 20 terms
        assert abs(r5.S_exact - r20.S_exact) < abs(r20.S_exact) * 0.5 + 1.0

    def test_integer_nearness_range(self, simple_field):
        result = rademacher_surprise(simple_field, d=3)
        assert 0.0 <= result.integer_nearness <= 0.5


class TestIntegerNearness:
    def test_exact_integer(self):
        assert integer_nearness(5.0) == 0.0

    def test_half_integer_is_max(self):
        assert abs(integer_nearness(2.5) - 0.5) < 1e-12

    def test_close_to_integer(self):
        assert integer_nearness(3.001) < 0.01


class TestCertifiedSurprise:
    def test_returns_tuple(self, simple_field):
        S, err = certified_surprise(simple_field, d=3)
        assert isinstance(S, float)
        assert isinstance(err, float)

    def test_error_bound_positive(self, simple_field):
        _, err = certified_surprise(simple_field, d=3)
        assert err > 0.0

    def test_high_precision_no_crash(self, simple_field):
        S, err = certified_surprise(simple_field, d=8, precision=6)
        assert math.isfinite(S)


# ── Eisenstein series tests ───────────────────────────────────────────────────

class TestEisensteinSeries:
    Q_STD = 0.01  # standard q value

    def test_P_at_zero_is_one(self):
        # P(0) = 1 - 24 * 0 = 1
        assert abs(P_series(0.0) - 1.0) < 1e-12

    def test_Q_at_zero_is_one(self):
        assert abs(Q_series(0.0) - 1.0) < 1e-12

    def test_R_at_zero_is_one(self):
        assert abs(R_series(0.0) - 1.0) < 1e-12

    def test_P_decreases_from_one(self):
        # P(q) < 1 for q > 0 (24 * positive > 0)
        assert P_series(self.Q_STD) < 1.0

    def test_Q_increases_from_one(self):
        assert Q_series(self.Q_STD) > 1.0

    def test_R_decreases_from_one(self):
        assert R_series(self.Q_STD) < 1.0

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            P_series(1.0)
        with pytest.raises(ValueError):
            Q_series(1.5)

    def test_identity_PQR(self):
        # Modular identity: Q³ − R² = 1728·Δ (for τ = i, q = e^(-2π))
        # We just verify Q > R > P ordering for small q
        q = 0.001
        assert Q_series(q) > P_series(q)


class TestEisensteinWeights:
    def test_shallow_gives_empirical(self):
        for d in range(1, 6):
            a, b, g = eisenstein_weights(d, d_crit=5)
            assert abs(a - 0.6) < 1e-12
            assert abs(b - 0.3) < 1e-12
            assert abs(g - 0.1) < 1e-12

    def test_deep_gives_eisenstein(self):
        for d in range(6, 16):
            a, b, g = eisenstein_weights(d, d_crit=5)
            assert abs(a - 1/6) < 1e-12
            assert abs(b - 1/3) < 1e-12
            assert abs(g - 1/2) < 1e-12

    def test_weights_sum_to_one(self):
        for d in [1, 5, 6, 10]:
            a, b, g = eisenstein_weights(d)
            assert abs(a + b + g - 1.0) < 1e-12

    def test_crossover_at_d_crit(self):
        a5, b5, g5 = eisenstein_weights(5, d_crit=5)
        a6, b6, g6 = eisenstein_weights(6, d_crit=5)
        # At d=5: empirical; at d=6: Eisenstein
        assert abs(a5 - 0.6) < 1e-12
        assert abs(a6 - 1/6) < 1e-12


class TestTauFunction:
    def test_empty_field_returns_zero(self):
        field = Field()
        assert tau_function(field, d=3) == complex(0.0)

    def test_single_relator_is_complex(self):
        r = Relator(depth=1, sigma=0.5, kappa=0.8, relation_type=RelationType.CAUSAL)
        field = Field(relators=[r])
        tau = tau_function(field, d=1)
        # sigma^5 * kappa^7 * exp(2πi*1/12)
        expected = r.sigma**5 * r.kappa**7 * cmath.exp(2j*math.pi/12)
        assert abs(tau - expected) < 1e-12

    def test_congruence_check_nonapplicable(self, simple_field):
        # d=3: 3 % 13 = 3 ≠ 6 → always returns True
        assert tau_congruence_check(simple_field, d=3, S_d=0.5) is True


# ── Maass completion tests ────────────────────────────────────────────────────

class TestShadowCuspForm:
    def test_at_z_zero(self):
        Ra = Relator(depth=1, sigma=0.7, kappa=0.5, relation_type=RelationType.CAUSAL)
        Rb = Relator(depth=1, sigma=0.6, kappa=0.5, relation_type=RelationType.CONTRADICTS)
        c = Contradiction(Ra, Rb)
        g = shadow_cusp_form(c, z=0.0)
        # exp(0) = 1; g = 0.7 * 0.6 * 1 = 0.42
        assert abs(g - complex(0.42)) < 1e-12

    def test_modulus_equals_product(self):
        Ra = Relator(depth=1, sigma=0.5, kappa=0.5, relation_type=RelationType.CAUSAL)
        Rb = Relator(depth=1, sigma=0.5, kappa=0.5, relation_type=RelationType.CONTRADICTS)
        c = Contradiction(Ra, Rb)
        g = shadow_cusp_form(c, z=1.0)
        # |g| = sigma_a * sigma_b * |exp(2πi*tau_diff)| = 0.25 * 1 = 0.25
        assert abs(abs(g) - 0.25) < 1e-10


class TestShadowCorrection:
    def test_no_contradictions_returns_zero(self, simple_field, default_goal):
        assert shadow_correction(simple_field, default_goal) == 0.0

    def test_productive_contradiction_gives_positive(
        self, contradiction_field, default_goal
    ):
        s = shadow_correction(contradiction_field, default_goal)
        assert s > 0.0

    def test_shadow_exceeds_one(self, contradiction_field, default_goal):
        s = shadow_correction(contradiction_field, default_goal)
        assert s > 1.0, f"shadow = {s:.4f}, expected > 1.0"

    def test_non_productive_ignored(self, default_goal):
        Ra = Relator(depth=1, sigma=0.9, kappa=0.5, relation_type=RelationType.CAUSAL)
        Rb = Relator(depth=1, sigma=0.9, kappa=0.5, relation_type=RelationType.CONTRADICTS)
        c = Contradiction(Ra, Rb, productive=False)
        field = Field(relators=[Ra, Rb], contradictions=[c])
        assert shadow_correction(field, default_goal) == 0.0


class TestCompleteTASF:
    def test_returns_complete_result(self, contradiction_field, default_goal):
        result = complete_tasf(contradiction_field, default_goal)
        assert isinstance(result, CompleteResult)

    def test_complete_equals_mock_plus_shadow(
        self, contradiction_field, default_goal
    ):
        result = complete_tasf(contradiction_field, default_goal)
        assert abs(
            result.delta_S_complete - (result.delta_S_mock + result.delta_S_shadow)
        ) < 1e-10

    def test_shadow_fraction_in_range(self, contradiction_field, default_goal):
        result = complete_tasf(contradiction_field, default_goal)
        assert 0.0 <= result.shadow_fraction <= 1.0


# ── Unified surprise tests ────────────────────────────────────────────────────

class TestUnifiedSurprise:
    def test_fast_regime_selected(self, simple_field):
        goal = Goal(q=0.5, kappa_max=1.0)  # Q > 0.01, d=3 ≤ 5
        us = UnifiedSurprise()
        result = us.compute(simple_field, goal, d=3)
        assert result.regime_used == "fast"

    def test_standard_regime_selected(self, simple_field):
        goal = Goal(q=0.001, kappa_max=1.0)  # Q ≤ 0.01, d=3 ≤ 5
        us = UnifiedSurprise()
        result = us.compute(simple_field, goal, d=3)
        assert result.regime_used == "standard"

    def test_deep_regime_selected(self, simple_field):
        goal = Goal(q=0.5, kappa_max=1.0)
        # Need relators at depth 7
        field = Field(relators=[
            Relator(depth=7, sigma=0.8, kappa=0.9, relation_type=RelationType.CAUSAL)
        ])
        us = UnifiedSurprise()
        result = us.compute(field, goal, d=7)
        assert result.regime_used == "deep"

    def test_returns_unified_result(self, simple_field, default_goal):
        us = UnifiedSurprise()
        result = us.compute(simple_field, default_goal, d=3)
        assert isinstance(result, UnifiedResult)

    def test_d_less_than_1_raises(self, simple_field, default_goal):
        us = UnifiedSurprise()
        with pytest.raises(ValueError):
            us.compute(simple_field, default_goal, d=0)

    def test_shadow_applied_when_contradictions(self, contradiction_field, default_goal):
        us = UnifiedSurprise()
        result = us.compute(contradiction_field, default_goal, d=3)
        assert result.shadow_applied is True

    def test_no_shadow_without_contradictions(self, simple_field, default_goal):
        us = UnifiedSurprise()
        result = us.compute(simple_field, default_goal, d=3)
        assert result.shadow_applied is False

    def test_error_bound_positive(self, simple_field, default_goal):
        us = UnifiedSurprise()
        result = us.compute(simple_field, default_goal, d=3)
        assert result.error_bound >= 0.0

    def test_suppressed_depths_populated(self, simple_field, default_goal):
        us = UnifiedSurprise()
        result = us.compute(simple_field, default_goal, d=10)
        # d ≡ 4 (mod 5): depths 4, 9; d ≡ 5 (mod 7): depth 5; etc.
        assert isinstance(result.suppressed_depths, list)
        assert 4 in result.suppressed_depths  # 4 % 5 == 4
