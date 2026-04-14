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
def parameterize(delays: MeasuredDelays, physical: PhysicalParams) -> NORModelParams:
    calculated = calculate_params(delays, physical)
    derived = compute_derived_constants(physical, calculated)

    return NORModelParams(physical, calculated, derived)



# Computing constants C1 - C3 (Eq. 12 - 15)
def compute_derived_constants(physical: PhysicalParams, calculated: CalculatedParams) -> DerivedConstants:
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
def calculate_params(delays: MeasuredDelays, physical: PhysicalParams) -> CalculatedParams:
    R5, RnA, RnB = compute_nmos_params(delays, physical)
    R, alpha1, alpha2 = compute_pmos_params(delays, physical, R5)

    return CalculatedParams(R5, RnA, RnB, R, alpha1, alpha2)

# The following are help formulas for parameterize
# Eq. 24, 25, 26, 27
def compute_nmos_params(delays: MeasuredDelays, physical: PhysicalParams):
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
def compute_pmos_params(delays: MeasuredDelays, physical: PhysicalParams, R5):
    R = solve_R(delays.S_rise_0, delays.S_rise_pos, delays.S_rise_neg, R5, physical.C, physical.delta_min)

    alpha1 = A_eq28(delays.S_rise_neg - physical.delta_min, R, R5, physical.C)
    alpha2 = A_eq28(delays.S_rise_pos - physical.delta_min, R, R5, physical.C)

    return R, alpha1, alpha2

# Eq. 29 - solving for R
def solve_R(S_rise_0, S_rise_pos, S_rise_neg, R5, C, delta_min):

    def objective(R):
        return eq29(S_rise_0, S_rise_pos, S_rise_neg, R, R5, C, delta_min)

    R_lo = 1e-6  # close to 0 but positive (Brentq requires borders with different signs)
    R_hi = 100 * R5  # upper border high to allow a high R, but not unrealistic

    R = brentq(objective, R_lo, R_hi, xtol=1e-15, rtol=1e-12)
    return R

# Eq. 29
def eq29(S_rise_0, S_rise_pos, S_rise_neg, R, R5, C, delta_min):
    return (
            A_eq28(S_rise_0 - delta_min, R, R5, C)
            - A_eq28(S_rise_pos - delta_min, R, R5, C)
            - A_eq28(S_rise_neg - delta_min, R, R5, C)
    )

# Eq. 28
def A_eq28(t, R, R5, C):
    K = (C * (R5 + 2*R) * np.log(2)) / t # repeating term

    w_arg = (K-1) * np.exp(K - 1)
    w_val = lambertw(w_arg, k=-1).real # W-1 (Lambert W with branch k = -1

    numerator = -2 * R * t * (1 - K)
    denominator = w_val + 1 - K

    return numerator/denominator


    

if __name__ == "__main__":
    delays, physical = load_config("gate_params.toml")

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
    print(f"  VDD     = {physical.VDD   :.4f} fF")

    params = parameterize(delays, physical)

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
