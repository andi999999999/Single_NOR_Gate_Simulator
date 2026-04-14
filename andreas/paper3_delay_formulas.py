from dataclasses import dataclass

import numpy as np

from andreas.parameter import NORModelParams, DerivedConstants, PhysicalParams, CalculatedParams

@dataclass
class Helpers:
    abs_delta: float
    # Eq. 11 (pos. Delta)
    d: float
    c_prime: float
    chi: float
    sqrt_chi: float
    A: float
    # Eq. 16 (neg. Delta, α₁ replacing α₂, abs_delta)
    d_bar: float
    c_bar_prime: float
    chi_bar: float
    sqrt_chi_bar: float
    A_bar: float        # Ā

"""   ===== δ↓ = falling Output (Eq. 45-55) =====  """
"""--- Complex Cases (Eq. 45-51) ---"""

# Eq. 45: Case (h,a)
# Δ = t_B - t_A ≤ 0
# (1,0) -> (0,0) -> (1,0)
def case_h_a(delta, delta_prime, T, params: NORModelParams):
    h = _compute_helpers(params, delta)

    delta_min = params.physical.delta_min
    R = params.calculated.R
    C1 = params.derived.C1
    C3 = params.derived.C3
    RnA = params.calculated.RnA
    a = params.derived.a

    T_eff = T + delta_min

    exp_term = np.exp(-T_eff / (2 * R * C3))

    # these use Ā from Eq. 16, becuase Δ ≤ 0
    exp1 = (-h.A_bar + a) / (2 * R * C3)
    exp2 = h.A_bar / (2 * R * C3)

    if delta_prime >= T_eff:
        # Sub-Case 1: Δ' ≥ T + δ_min
        excess = 2 * (delta_prime - T_eff)

        base1 = 1 + (2 * T_eff) / (a + h.abs_delta + h.sqrt_chi_bar + excess)
        base2 = 1 + (2 * T_eff) / (a + h.abs_delta - h.sqrt_chi_bar + excess)

    else:
        # Sub-Case 2: 0 ≤ Δ' < T + δ_min
        base1 = 1 + (2 * delta_prime) / (a + h.abs_delta + h.sqrt_chi_bar)
        base2 = 1 + (2 * delta_prime) / (a + h.abs_delta - h.sqrt_chi_bar)

    inner = 2 - exp_term * (base1 ** exp1) * (base2 ** exp2)

    return delta_min + C1 * RnA * np.log(inner)

# (0, 1) -> (0, 0) -> (1, 0)
def case_g_a(delta, delta_prime, T, params: NORModelParams):
    h = _compute_helpers(params, delta)

    delta_min = params.physical.delta_min
    R = params.calculated.R
    C1 = params.derived.C1
    C3 = params.derived.C3
    RnA = params.calculated.RnA
    a = params.derived.a

    T_eff = T + delta_min

    exp_term = np.exp(-T_eff / (2 * R * C3))

    # using A from Eq. 11, because Δ ≥ 0
    exp1 = (-h.A + a) / (2 * R * C3)
    exp2 = h.A / (2 * R * C3)

    if delta_prime >= T_eff:
        # Sub-Case 1: Δ' ≥ T + δ_min
        excess = 2 * (delta_prime - T_eff)

        base1 = 1 + (2 * T_eff) / (a + delta + h.sqrt_chi + excess)
        base2 = 1 + (2 * T_eff) / (a + delta - h.sqrt_chi + excess)

    else:
        # Sub-Case 2: 0 ≤ Δ' < T + δ_min
        base1 = 1 + (2 * delta_prime) / (a + delta + h.sqrt_chi)
        base2 = 1 + (2 * delta_prime) / (a + delta - h.sqrt_chi)

    inner = 2 - exp_term * (base1 ** exp1) * (base2 ** exp2)

    return delta_min + C1 * RnA * np.log(inner)

# (1,0) -> (0, 0) -> (0, 1)
def case_h_b():

# (0,1) -> (0, 0) -> (0, 1)
def case_g_b():


def _falling_complex_helper(delta_used, delta_prime, T, params: NORModelParams,
                            A_used, sqrt_chi_used, prefactor):
    """
    Common calculation for Eq. 45-51.

    Args:
        A_used:         A or Ā (depending on sign of Δ)
        sqrt_chi_used:  √χ or √χ̄
        delta_used:     Δ or |Δ|
        prefactor:      C₁·RnA or C'₁·RnB
    """

    delta_min = params.physical.delta_min
    R = params.calculated.R
    C3 = params.derived.C3
    a = params.derived.a

    T_eff = T + delta_min

    exp_term = np.exp(-T_eff / (2 * R * C3))

    exp1 = (-A_used + a) / (2 * R * C3)
    exp2 = A_used / (2 * R * C3)

    if delta_prime >= T_eff:
        excess = 2 * (delta_prime - T_eff)

        base1 = 1 + (2 * T_eff) / (a + delta_used + sqrt_chi_used + excess)
        base2 = 1 + (2 * T_eff) / (a + delta_used - sqrt_chi_used + excess)

    else:
        # Sub-Case 2: 0 ≤ Δ' < T + δ_min
        base1 = 1 + (2 * delta_prime) / (a + delta_used + sqrt_chi_used)
        base2 = 1 + (2 * delta_prime) / (a + delta_used - sqrt_chi_used)

    inner = 2 - exp_term * (base1 ** exp1) * (base2 ** exp2)

    return delta_min + prefactor * np.log(inner)



"""--- Simple Cases (Eq. 52-55)---"""
# Eq. 52: Case (a,c) und Case (f,c)
# δ↓(T) = -C₂·RnB·(T + δ_min) / (C₁·(RnA + RnB)) + δ_min
# (0,0) -> (1, 0) -> (1, 1)
def case_a_c(T, params: NORModelParams):
    C1 = params.derived.C1
    C2 = params.derived.C2
    RnB = params.calculated.RnB
    RnA = params.calculated.RnA
    delta_min = params.physical.delta_min

    return ((-C2 * RnB * (T + delta_min))
            / (C1 * (RnA + RnB))
            + delta_min)

# Equals: Case (a,c), Eq. 52
# (1,1) -> (1, 0) -> (1, 1)
def case_f_c(T, params: NORModelParams):
    return case_a_c(T, params)

# Eq. 53: Case (b,d) und Case (e,d)
# δ↓(T) = -C₂·RnA·(T + δ_min) / (C'₁·(RnA + RnB)) + δ_min
# (0,0) -> (0, 1) -> (1, 1)
def case_b_d(T, params: NORModelParams):
    C1_p = params.derived.C1_p
    C2 = params.derived.C2
    RnB = params.calculated.RnB
    RnA = params.calculated.RnA
    delta_min = params.physical.delta_min

    return ((-C2 * RnA * (T + delta_min))
            / (C1_p * (RnA + RnB))
            + delta_min)

# Equals: Case (b,d), Eq. 53
# (1,1) -> (0, 1) -> (1, 1)
def case_e_d(T, params: NORModelParams):
    return case_b_d(T, params)

# Eq. 54: Case (c,e) und Case (d,e)
# δ↓(T) = -C'₁·(RnA + RnB)·(T + δ_min) / (C₂·RnA) + δ_min
# (1,0) -> (1, 1) -> (0, 1)
def case_c_e(T, params: NORModelParams):
    C1_p = params.derived.C1_p
    C2 = params.derived.C2
    RnB = params.calculated.RnB
    RnA = params.calculated.RnA
    delta_min = params.physical.delta_min

    return (((-C1_p * (RnA + RnB) * (T + delta_min))
            / (C2 * RnA))
            + delta_min)

# Equals: Case (c,e), Eq. 54
# (0,1) -> (1, 1) -> (0, 1)
def case_d_e(T, params: NORModelParams):
    return case_c_e(T, params)

# Eq. 55: Case (c,f) und Case (d,f)
# δ↓(T) = -C₁·(RnA + RnB)·(T + δ_min) / (C₂·RnB) + δ_min
# (1,0) -> (1, 1) -> (1, 0)
def case_c_f(T, params: NORModelParams):
    C1 = params.derived.C1
    C2 = params.derived.C2
    RnB = params.calculated.RnB
    RnA = params.calculated.RnA
    delta_min = params.physical.delta_min

    return (((-C1 * (RnA + RnB) * (T + delta_min))
            / (C2 * RnB))
            + delta_min)

# Equals: Case (c,f), Eq. 55
# (0,1) -> (1, 1) -> (1, 0)
def case_d_f():


# (0,0) -> (0, 1) -> (0, 0)
def case_b_g():


# (1,1) -> (0, 1) -> (0, 0)
def case_e_g():


# (0,0) -> (1, 0) -> (0, 0)
def case_a_h():


# (1,1) -> (1, 0) -> (0,0)
def case_f_h():

"""Helper-variables, found after Eq. 11, depending on Δ"""
def _compute_helpers(params: NORModelParams, delta):
    a = params.derived.a
    alpha1 = params.calculated.alpha1
    alpha2 = params.calculated.alpha2
    R = params.calculated.R
    abs_delta = abs(delta)

    d = a + delta
    c_prime = (alpha2 * delta) / (2 * R)
    chi = d ** 2 - 4 * c_prime
    sqrt_chi = np.sqrt(chi)
    A = (alpha2 * delta - a * R * (d - sqrt_chi)) / (2 * R * sqrt_chi)

    # --- Eq. 16: "bar" versions (with α₁ (replacing a2) and abs_delta, for neg. Δ) ---
    d_bar = a + abs_delta
    c_bar_prime = (alpha1 * abs_delta) / (2 * R)
    chi_bar = d_bar ** 2 - 4 * c_bar_prime
    sqrt_chi_bar = np.sqrt(chi_bar)
    A_bar = (alpha1 * abs_delta - a * R * (d_bar - sqrt_chi_bar)) / (2 * R * sqrt_chi_bar)

    return Helpers(
        abs_delta=abs_delta,
        d=d,
        c_prime=c_prime,
        chi=chi,
        sqrt_chi=sqrt_chi,
        A=A,
        d_bar=d_bar,
        c_bar_prime=c_bar_prime,
        chi_bar=chi_bar,
        sqrt_chi_bar=sqrt_chi_bar,
        A_bar=A_bar,
    )