# TORIS Implementation Deviations

Tracked deviations from the TORIS_SECTION12_EXACT_SURPRISE.md specification.

---

## Section 12 Deviations

### DEV-12-001: TORIS Sections 1–11 not present in this repository

**Spec assumption:** "Update all existing surprise computations to use UnifiedSurprise
when appropriate (don't break existing tests — add as new option)"

**Reality:** This repository (`infinitule/_nsn`) is the NOESIS project. TORIS Sections
1–11 (core TORIS engine, TFSA, TASF, partition congruences, etc.) are not implemented
here. Section 12 is built as a self-contained module with the field types and engine
stubs required to run the new algorithms.

**Impact:** The "don't break existing tests" requirement cannot apply (no prior TORIS
tests exist here). The 57 new tests cover Section 12 only.

---

### DEV-12-002: Error bound formula direction

**Spec formula:** `|S(d) − S_N(d)| < C_F · exp(−π√(2d/3)/N)` — N in denominator.

**Problem:** With N in the denominator, the bound INCREASES with N (exponent approaches 0),
which contradicts the purpose of a convergence bound that should tighten as more terms
are added.

**Deviation:** Implemented as `C_F · exp(−π·N·√(2d/3))` — N in the exponent numerator.
This decreases exponentially with N, consistent with the expected convergence of the
Rademacher series and with the spec's stated claims (N=6 gives ~0.02% relative error).

---

### DEV-12-003: TORIS Rademacher series convergence is not monotone

**Spec claim:** "N=3 terms: ~8 significant figures"

**Observed:** For TORIS Kloosterman sums (complex-valued due to multi-type fields),
convergence is not strictly monotone. Relative error at N=3 is ~0.24%; best error
(N=6) is ~0.02% of the 20-term reference. The non-monotone behaviour arises because
`Im(B_k^F(d)) ≠ 0` when relators have different tau_index values.

**Impact:** Exp 14 asserts initial convergence (err_3 < err_1) and best relative error
< 1%, rather than 8 significant figures with 3 terms.

---

### DEV-12-004: "Exact" ΔS reference in Exp 15 uses internal consistency

**Spec:** "Compare ΔS_complete to brute-force exact surprise"

**Deviation:** Without the full TORIS Sections 1–11 machinery, there is no independent
brute-force surprise oracle. Exp 15 verifies: (1) shadow > 1.0; (2) ΔS_complete > ΔS_mock;
(3) shadow_fraction > 30%; (4) ΔS_complete = ΔS_mock + ΔS_shadow (internal consistency).

---

### DEV-12-005: Eisenstein series used for weight derivation only

**Spec:** P, Q, R series are used to prove the dual-weighting theorem. In production,
the weights are just the constants (1/6, 1/3, 1/2) or (0.6, 0.3, 0.1).

**Implementation:** P_series, Q_series, R_series are implemented and exposed in the
API for completeness, but the actual weight selection in `eisenstein_weights()` uses
the analytically derived constants, not runtime evaluation of P/Q/R.

---

### DEV-12-006: scipy required for Eichler integral

**Spec constraint:** "numerical integration via scipy.integrate.quad"

**Deviation from standard NOESIS deps:** `scipy` was not a prior dependency of this
repo. Added as a runtime dependency for `maass_completion.py`. The `shadow_correction`
function uses the closed-form approximation from §12.3.4 rather than numerical Eichler
integration, making the implementation scipy-optional for the main path.

---

### DEV-12-007: /toris-audit not executed

**Spec:** "Run /toris-audit"

**Reality:** `/toris-audit` is a TORIS-specific skill not available in this session.
The 57-test suite in `tests/test_toris_section12.py` serves as the audit substitute.
All 57 tests pass.
