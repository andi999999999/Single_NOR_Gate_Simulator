"""Gate-Parameter-Model

Core: this is the NOR-Gate model parameterization taken from the paper (prop- 4.3, Eq. 24-31)
'parameterize_nor' uses the SPICE measured NOR delays to calculate the Parameters (Calculated & Derived)

NAND ('parameterize_nand') is just a thin wrapper. It uses the exact same parameterization
as NOR via De-Morgan transformation - we switch rising/falling (reflection around VDD/2)
then call the NOR parameterization. All internal helper methods implement the NOR equations
described in the Paper and are 'shared' between both (by transformation).

Note: For the NAND algorithm there is a thin wrapper around the NOR algorithm
again, utilizing De-Morgan transformations.
"""

import argparse
import tomllib
import numpy as np
from dataclasses import dataclass
from scipy.special import lambertw
from scipy.optimize import brentq

@dataclass
class MeasuredDelays:
    """SPICE measured delay values, for initialization"""
    # falling output (nMOS)
    S_fall_neg: float   # δ↓_S(−∞)
    S_fall_0:   float   # δ↓_S(0)
    S_fall_pos: float   # δ↓_S(+∞)
    # rising output (pMOS)
    S_rise_neg: float   # δ↑_S(−∞)
    S_rise_0:   float   # δ↑_S(0)
    S_rise_pos: float   # δ↑_S(+∞)

@dataclass
class PhysicalParams:
    """Input variables: physically motivated, but freely selectable/robust.
    C: Load capacity (scaling parameter) [F]
    delta_min: Pure delay (measured, but the model is robust) [s]
    """
    delta_min: float
    C: float
    VDD: float

@dataclass
class DerivedConstants:
    """Eq. 12–15: depend on C AND the calculated resistances"""
    C1: float     # C(R5 + RnA) / RnA
    C1_p: float   # C(R5 + RnB) / RnB
    C2: float     # C(R5(RnA+RnB) + RnA·RnB) / (RnA·RnB)
    C3: float     # C(R5 + 2R) / (2R)
    a: float      # (α₁ + α₂) / (2R)
    tau3: float   # = 2R·C3, used in Cases (g)/(h) and Algorithm 1

@dataclass
class CalculatedParams:
    """Proposition 4.3: Calculated Modellparameters"""
    R5: float
    RnA: float
    RnB: float
    R: float  # solved numerically
    alpha1: float
    alpha2: float

@dataclass
class NORModelParams:
    """Wrapper Class, everything describing the Gate-Modell"""
    physical: PhysicalParams
    calculated: CalculatedParams
    derived: DerivedConstants


def load_config(path: str) -> tuple[MeasuredDelays, PhysicalParams]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    d = raw["measured_delays"]
    delays = MeasuredDelays(
        S_fall_neg = d["delta_S_fall_neg_inf"],
        S_fall_0   = d["delta_S_fall_zero"],
        S_fall_pos = d["delta_S_fall_pos_inf"],
        S_rise_neg = d["delta_S_rise_neg_inf"],
        S_rise_0   = d["delta_S_rise_zero"],
        S_rise_pos = d["delta_S_rise_pos_inf"],
    )

    m = raw["model"]
    physical = PhysicalParams(
        delta_min=m["delta_min"],
        C=m["C"],
        VDD=m["VDD"]
    )

    return delays, physical

# Wrapper to calulate all Model parameters
def parameterize_nor(delays: MeasuredDelays, physical: PhysicalParams) -> NORModelParams:
    calculated = _calculate_params(delays, physical)
    derived = _compute_derived_constants(physical, calculated)

    return NORModelParams(physical, calculated, derived)

def parameterize_nand(delays: MeasuredDelays, physical: PhysicalParams) -> NORModelParams:
    """Using De-Morgan we can convert NAND to its  NOR equivalent: the reflection on VDD/2 switches rising/falling.
    Therefore we simply exchange rising/falling and pass it on to the existing NOR-parametarization
    The NAND simulator itself will use these NOR gate parameters accoringly to generate NAND outputs using De-Morgan rules
    """
    nor_space = MeasuredDelays(
        S_fall_neg=delays.S_rise_neg, S_fall_0=delays.S_rise_0, S_fall_pos=delays.S_rise_pos,
        S_rise_neg=delays.S_fall_neg, S_rise_0=delays.S_fall_0, S_rise_pos=delays.S_fall_pos,
    )
    return parameterize_nor(nor_space, physical)



# Computing constants C1 - C3 (Eq. 12 - 15)
def _compute_derived_constants(physical: PhysicalParams, calculated: CalculatedParams) -> DerivedConstants:
    C = physical.C
    R5, RnA, RnB, R, alpha1, alpha2 = calculated.R5, calculated.RnA, calculated.RnB, calculated.R, calculated.alpha1, calculated.alpha2

    C3 = (C * (R5 + 2 * R)) / (2 * R)

    return DerivedConstants(
        C1= (C * (R5 + RnA)) / RnA,
        C1_p= (C * (R5 + RnB)) / RnB,
        C2= (C * (R5 * (RnA + RnB) + RnA * RnB)) / (RnA * RnB),
        C3= C3,
        a = (alpha1 + alpha2 ) / (2*R),
        tau3=2 * R * C3
    )

# Wrapper for calculated params (Eq. 24 - 31)
def _calculate_params(delays: MeasuredDelays, physical: PhysicalParams) -> CalculatedParams:
    R5, RnA, RnB = _compute_nmos_params(delays, physical)
    R, alpha1, alpha2 = _compute_pmos_params(delays, physical, R5)

    return CalculatedParams(R5, RnA, RnB, R, alpha1, alpha2)

# The following are help formulas for parameterize
# Eq. 24, 25, 26, 27
def _compute_nmos_params(delays: MeasuredDelays, physical: PhysicalParams):
    # Eq. 27
    epsilon = np.sqrt((delays.S_fall_pos - delays.S_fall_0) * (delays.S_fall_neg - delays.S_fall_0))

    # Eq. 24
    R5 = (delays.S_fall_0 - physical.delta_min - epsilon) / (np.log(2) * physical.C)
    #Eq. 25
    RnA = (delays.S_fall_pos - delays.S_fall_0 + epsilon) / (np.log(2) * physical.C)
    #Eq. 26
    RnB = (delays.S_fall_neg - delays.S_fall_0 + epsilon) / (np.log(2) * physical.C)

    return R5, RnA, RnB

# Eq. 28, 29, 30, 31
def _compute_pmos_params(delays: MeasuredDelays, physical: PhysicalParams, R5):
    R = _solve_R(delays.S_rise_0, delays.S_rise_pos, delays.S_rise_neg, R5, physical.C, physical.delta_min)

    alpha1 = _A_eq28(delays.S_rise_neg - physical.delta_min, R, R5, physical.C)
    alpha2 = _A_eq28(delays.S_rise_pos - physical.delta_min, R, R5, physical.C)

    return R, alpha1, alpha2

# Eq. 29 - solving for R
def _solve_R(S_rise_0, S_rise_pos, S_rise_neg, R5, C, delta_min):

    def objective(R):
        return _eq29(S_rise_0, S_rise_pos, S_rise_neg, R, R5, C, delta_min)

    R_lo = 1e-6  # close to 0 but positive (Brentq requires borders with different signs)
    R_hi = 100 * R5  # upper border high to allow a high R, but not unrealistic

    R = brentq(objective, R_lo, R_hi, xtol=1e-15, rtol=1e-12)
    return R

# Eq. 29
def _eq29(S_rise_0, S_rise_pos, S_rise_neg, R, R5, C, delta_min):
    return (
            _A_eq28(S_rise_0 - delta_min, R, R5, C)
            - _A_eq28(S_rise_pos - delta_min, R, R5, C)
            - _A_eq28(S_rise_neg - delta_min, R, R5, C)
    )

# Eq. 28
def _A_eq28(t, R, R5, C):
    K = (C * (R5 + 2*R) * np.log(2)) / t # repeating term

    w_arg = (K-1) * np.exp(K - 1)
    w_val = lambertw(w_arg, k=-1).real # W-1 (Lambert W with branch k = -1)

    numerator = -2 * R * t * (1 - K)
    denominator = w_val + 1 - K

    return numerator/denominator





def print_params_report():
    parser = argparse.ArgumentParser(description="Params Report")
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to Gate-Params-TOML")
    parser.add_argument("--gate", choices=["nor", "nand"], default="nor",
                        help="Which gate Model? (Default: nor)")
    args = parser.parse_args()

    config = args.config or (
        "nand_gate_params.toml" if args.gate == "nand" else "nor_gate_params.toml"
    )
    delays, physical = load_config(config)

    parameterize = parameterize_nand if args.gate == "nand" else parameterize_nor
    params = parameterize(delays, physical)

    print("=== Loaded Delays ===")
    print(f"  δ↓_S(-∞) = {delays.S_fall_neg*1e12:.4f} ps")
    print(f"  δ↓_S(0)  = {delays.S_fall_0  *1e12:.4f} ps")
    print(f"  δ↓_S(+∞) = {delays.S_fall_pos*1e12:.4f} ps")
    print(f"  δ↑_S(-∞) = {delays.S_rise_neg*1e12:.4f} ps")
    print(f"  δ↑_S(0)  = {delays.S_rise_0  *1e12:.4f} ps")
    print(f"  δ↑_S(+∞) = {delays.S_rise_pos*1e12:.4f} ps")

    print("\n=== Physical Parameters ===")
    print(f"  δ_min = {physical.delta_min * 1e15:.4f} fs")
    print(f"  C     = {physical.C * 1e15:.4f} fF")
    print(f"  VDD     = {physical.VDD   :.4f} V")

    print("\n=== Computed Parameters ===")
    print(f"  R5     = {params.calculated.R5    :.6f} Ω")
    print(f"  RnA    = {params.calculated.RnA   :.6f} Ω")
    print(f"  RnB    = {params.calculated.RnB   :.6f} Ω")
    print(f"  R      = {params.calculated.R     :.6f} Ω")
    print(f"  α₁     = {params.calculated.alpha1*1e12:.6f} ps")
    print(f"  α₂     = {params.calculated.alpha2*1e12:.6f} ps")

    print("\n=== Derived Constants ===")
    print(f"  C1     = {params.derived.C1  *1e15:.4f}")
    print(f"  C1_p   = {params.derived.C1_p*1e15:.4f}")
    print(f"  C2     = {params.derived.C2  *1e15:.4f}")
    print(f"  C3     = {params.derived.C3  *1e15:.4f}")
    print(f"  a      = {params.derived.a   *1e12:.6f} ps/Ω")
    print(f"  τ₃     = {params.derived.tau3}")

    return params, delays, physical

if __name__ == "__main__":
    print_params_report()
