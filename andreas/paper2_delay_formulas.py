from dataclasses import dataclass

import numpy as np
from optype import do_le
from scipy.special import lambertw

from parameter import NORModelParams, DerivedConstants, PhysicalParams, CalculatedParams, basic_sanity_test as parameter_basic_sanity_test


"""Helper-variables, these are dynamically calculated, found after Eq. 11/15, depending on Δ"""
@dataclass
class Helpers:
    abs_delta: float
    # Eq. 11 (pos. Delta)
    d: float
    #c_prime: float
    chi: float
    sqrt_chi: float
    A: float
    # Eq. 16 (neg. Delta, α₁ replacing α₂, abs_delta)
    d_bar: float
    #c_bar_prime: float
    chi_bar: float
    sqrt_chi_bar: float
    A_bar: float        # Ā


def _compute_helpers(params: NORModelParams, delta):
    # extract constants from parameters
    a = params.derived.a
    alpha1 = params.calculated.alpha1
    alpha2 = params.calculated.alpha2
    R = params.calculated.R

    abs_delta = abs(delta)

    # calculate non-bar versions of helpers, Eq. 11
    d = a + delta
    c_prime = (alpha2 * delta) / (2 * R)
    chi = d ** 2 - 4 * c_prime
    sqrt_chi = np.sqrt(chi)
    A = (alpha2 * delta - a * R * (d - sqrt_chi)) / (2 * R * sqrt_chi)

    # calculate non-bar versions of helpers (used when negative delta), Eq. 15
    # --- Eq. 15: "bar" versions (with α₁ (replacing a2) and abs_delta, for neg. Δ) ---
    d_bar = a + abs_delta
    c_bar_prime = (alpha1 * abs_delta) / (2 * R)
    chi_bar = d_bar ** 2 - 4 * c_bar_prime
    sqrt_chi_bar = np.sqrt(chi_bar)
    A_bar = (alpha1 * abs_delta - a * R * (d_bar - sqrt_chi_bar)) / (2 * R * sqrt_chi_bar)

    return Helpers(
        abs_delta=abs_delta,
        d=d,
        #c_prime=c_prime,
        chi=chi,
        sqrt_chi=sqrt_chi,
        A=A,
        d_bar=d_bar,
        #c_bar_prime=c_bar_prime,
        chi_bar=chi_bar,
        sqrt_chi_bar=sqrt_chi_bar,
        A_bar=A_bar,
    )



"""   ===== δ↓ = falling Output (Eq. 32-37) =====  """

""" Case (a) and Case (f) """
# Eq. 32
# a: (0, 0) -> (1, 0) / f: (1, 1) -> (1, 0)
def δ_case_a_f(Vint, params: NORModelParams):
    tau = _tau_case_a_f(params)

    return _δ_falling_helper(Vint, tau, params)

# Eq. 33
def Vout_case_a_f(t, params: NORModelParams):
    tau = _tau_case_a_f(params)

    return _Vout_falling_helper(t, tau, params)

# R*C time constant
def _tau_case_a_f(params):
    return params.derived.C1 * params.calculated.RnA

""" Case (b) and Case (e) """
# Eq. 34
# b: (0, 0) -> (0, 1) / e: (1, 1) -> (0, 1)
def δ_case_b_e(Vint, params: NORModelParams):
    tau = _tau_case_b_e(params)

    return _δ_falling_helper(Vint, tau, params)

# Eq. 35
def Vout_case_b_e(t, params: NORModelParams):
    tau = _tau_case_b_e(params)

    return _Vout_falling_helper(t, tau, params)

# R*C time constant
def _tau_case_b_e(params):
    return params.derived.C1_p * params.calculated.RnB

""" Case (c) and Case (d) """
# Eq. 36
# c: (1, 0) -> (1, 1) / d: (0, 1) -> (1, 1)
def δ_case_c_d(Vint, params: NORModelParams):
    tau = _tau_case_c_d(params)

    return _δ_falling_helper(Vint, tau, params)

# Eq. 37
def Vout_case_c_d(t, params: NORModelParams):
    tau = _tau_case_c_d(params)

    return _Vout_falling_helper(t, tau, params)

# R*C time constant
def _tau_case_c_d(params):
    C2, RnA, RnB = params.derived.C2, params.calculated.RnA, params.calculated.RnB
    return C2 * RnA * RnB / (RnA + RnB)


"""   --- helper methods for: δ↓ = falling Output (Eq. 32-37) --- """
# helper method for δ↓ formulas 32, 34, 36, as they are similar in structure
def _δ_falling_helper(Vint, tau, params: NORModelParams):
    VDD = params.physical.VDD
    delta_min = params.physical.delta_min

    return tau * np.log(2 * Vint / VDD) + delta_min

# helper method for δ↓ formulas 33, 35, 37, as they are similar in structure
def _Vout_falling_helper(t, tau, params: NORModelParams):
    VDD = params.physical.VDD

    return (VDD / 2) * np.exp(-t / tau)


"""   ===== δ↑ = rising Output (Eq. 38-41) =====  """

""" Case (g) """
# Eq. 38
# (0, 1) -> (0, 0)
def δ_case_g(delta, Vint, params: NORModelParams):
    assert delta >= 0, f"Case (g) requires Δ ≥ 0, got {delta}"

    return _δ_rising_helper(
        delta_used=delta,
        Vint=Vint,
        alpha_used=params.calculated.alpha1,
        delta_sat_func=δVint_inf,
        params=params
    )


def Vout_case_g(t, delta, Vint, params, delay_g):
    helpers = _compute_helpers(params, delta)

    return _Vout_rising_helper(
        t,
        delta_used=delta,
        Vint=Vint,
        sqrt_chi_used=helpers.sqrt_chi,
        A_used=helpers.A,
        delay=delay_g,
        params=params
    )



""" Case (h) """
# Eq. 40
# (1, 0) -> (0, 0)
def δ_case_h(delta, Vint, params: NORModelParams):
    assert delta <= 0, f"Case (h) requires Δ ≤ 0, got {delta}"

    return _δ_rising_helper(
        delta_used=abs(delta),
        Vint=Vint,
        alpha_used=params.calculated.alpha2,
        delta_sat_func=δVint_neg_inf,
        params=params
    )

def Vout_case_h(t, delta, Vint, params, delay_h):
    helpers = _compute_helpers(params, delta)

    return _Vout_rising_helper(
        t,
        delta_used=helpers.abs_delta,
        Vint=Vint,
        sqrt_chi_used=helpers.sqrt_chi_bar,
        A_used=helpers.A_bar,
        delay=delay_h,
        params=params
    )

"""   --- helper methods for: δ↑ = rising Output (Case g&h, Eq. 38-41) --- """
# helper method for δ↑ formulas 38 & 40 as they are similar in structure
def _δ_rising_helper(delta_used, Vint, alpha_used, delta_sat_func, params: NORModelParams):
    VDD = params.physical.VDD
    R = params.calculated.R
    C3 = params.derived.C3
    delta_min = params.physical.delta_min

    if Vint > VDD / 2:
        return -2 * R * C3 * np.log(VDD / (2 * (VDD - Vint))) + delta_min
    else:
        delta_0 = δVint_0(Vint, params)
        delta_inf = delta_sat_func(Vint, params)
        alpha1 = params.calculated.alpha1
        alpha2 = params.calculated.alpha2

        delta_crit = (alpha1 + alpha2) * (delta_0 - delta_inf) / alpha_used

        if delta_used < delta_crit:
            return delta_0 - (alpha_used / (alpha1 + alpha2)) * delta_used + delta_min
        else:
            return delta_inf + delta_min

# helper method for δ↑ formulas 39 & 41 as they are similar in structure
def _Vout_rising_helper(t, delta_used, Vint, sqrt_chi_used, A_used, delay, params):
    VDD = params.physical.VDD
    delta_min = params.physical.delta_min
    tau3 = params.derived.tau3

    a = params.derived.a

    e_factor = np.exp(-t / tau3)
    exp_A_neg = (-A_used + a) / tau3  # exponent of first term
    exp_A_pos = A_used / tau3  # exponent of second term

    denom_plus = a + delta_used + sqrt_chi_used
    denom_minus = a + delta_used - sqrt_chi_used

    if Vint > VDD / 2:
        return VDD + (Vint - VDD) * e_factor \
            * _power_term(t, denom_plus, exp_A_neg) \
            * _power_term(t, denom_minus, exp_A_pos)

    else:
        shift = 2 * (delay - delta_min)

        denom_plus_shifted = denom_plus + shift
        denom_minus_shifted = denom_minus + shift

        return VDD * (1 - (e_factor / 2)
                      * _power_term(t, denom_plus_shifted, exp_A_neg)
                      * _power_term(t, denom_minus_shifted, exp_A_pos))

# (DE: Potenzterm)
def _power_term(t, denom, exponent):
    """(1 + 2t/denom)^exponent"""
    return (1 + 2 * t / denom) ** exponent



"""    --- δ(Vint) calculation used for: δ↑ = rising Output (Eq. 38-41) --- """
# Eq. 42
def δVint_0(Vint, params: NORModelParams):
    alpha_eff = params.calculated.alpha1 + params.calculated.alpha2

    return _δVint_rising_helper(Vint, alpha_eff, params)

# Eq. 43
def δVint_inf(Vint, params: NORModelParams):
    return  _δVint_rising_helper(Vint, params.calculated.alpha2, params)

# Eq. 44
def δVint_neg_inf(Vint, params: NORModelParams):
    return  _δVint_rising_helper(Vint, params.calculated.alpha1, params)

# common structure among calculations of δ(Vint)
def _δVint_rising_helper(Vint, alpha_eff, params: NORModelParams):
    R = params.calculated.R
    C3 = params.derived.C3
    VDD = params.physical.VDD

    prefactor = -alpha_eff / (2*R)
    exponent =  (4 * R**2 * C3) / alpha_eff
    base = ((2 * (VDD - Vint)) / VDD) ** exponent
    w_arg = -1 / (np.e * base)

    return prefactor * (1 + lambertw(w_arg, k=-1).real)







def basic_sanity_check():
    params, delays, physical = parameter_basic_sanity_test()
    VDD = params.physical.VDD

    print("\n\n=============== Delay Formulas sanity check ===============")
    print("----- SPICE values - for comparison: -----")
    print(f"  δ↓_S(-∞) = {delays.S_fall_neg * 1e12:.4f} ps")
    print(f"  δ↓_S(0)  = {delays.S_fall_0 * 1e12:.4f} ps")
    print(f"  δ↓_S(+∞) = {delays.S_fall_pos * 1e12:.4f} ps")
    print(f"  δ↑_S(-∞) = {delays.S_rise_neg * 1e12:.4f} ps")
    print(f"  δ↑_S(0)  = {delays.S_rise_0 * 1e12:.4f} ps")
    print(f"  δ↑_S(+∞) = {delays.S_rise_pos * 1e12:.4f} ps")

    print("----- Model calculated values: -----")
    print(f"  δ↓_M(-∞) = {δ_case_b_e(VDD, params) * 1e12:.4f} ps")
    print(f"  δ↓_M(0)  = {δ_case_c_d(VDD, params) * 1e12:.4f} ps")
    print(f"  δ↓_M(+∞) = {δ_case_a_f(VDD, params) * 1e12:.4f} ps")
    print(f"  δ↑_M(-∞) = {δ_case_h(-1e6, 0, params) * 1e12:.4f} ps")
    print(f"  δ↑_M(0) (Case g) = {δ_case_g(0, 0, params) * 1e12:.4f} ps")
    print(f"  δ↑_M(0) (Case h) = {δ_case_h(0, 0, params) * 1e12:.4f} ps")
    print(f"  δ↑_M(+∞) = {δ_case_g(1e6, 0, params) * 1e12:.4f} ps")



if __name__ == "__main__":
    basic_sanity_check()