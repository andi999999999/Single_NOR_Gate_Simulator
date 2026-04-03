import tomllib
import numpy as np
from dataclasses import dataclass
from scipy.special import lambertw
from scipy.optimize import brentq

@dataclass
class MeasuredDelays:
    # falling output (nMOS)
    S_fall_neg: float   # δ↓_S(−∞)
    S_fall_0:   float   # δ↓_S(0)
    S_fall_pos: float   # δ↓_S(+∞)
    # rising output (pMOS)
    S_rise_neg: float   # δ↑_S(−∞)
    S_rise_0:   float   # δ↑_S(0)
    S_rise_pos: float   # δ↑_S(+∞)

@dataclass
class ModelConfig:
    delta_min: float
    C: float

@dataclass
class NORModelParams:
    """Berechnete Modellparameter (Output der Parametrierung)"""
    R5:   float
    RnA:  float
    RnB:  float
    R:    float    # numerisch gelöst
    alpha1: float  # γ₁ im Paper
    alpha2: float  # γ₂ im Paper


def load_config(path: str) -> tuple[MeasuredDelays, ModelConfig]:
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
    config = ModelConfig(delta_min=m["delta_min"], C=m["C"])

    return delays, config

def parameterize(delays: MeasuredDelays, config: ModelConfig):
    R5, RnA, RnB = compute_nmos_params(delays, config)
    R, alpha1, alpha2 = compute_pmos_params(delays, config, R5)

    return NORModelParams(R5, RnA, RnB, R, alpha1, alpha2)

# The following are help formulas for parameterize
# Eq. 24, 25, 26, 27
def compute_nmos_params(delays: MeasuredDelays, config: ModelConfig):
    # Eq. 27
    epsilon = np.sqrt((delays.S_fall_pos - delays.S_fall_0) * (delays.S_fall_neg - delays.S_fall_0))

    # Eq. 24
    R5 = (delays.S_fall_0 - config.delta_min - epsilon) / (np.log(2) * config.C)
    #Eq. 25
    RnA = (delays.S_fall_pos - delays.S_fall_0 + epsilon) / (np.log(2) * config.C)
    #Eq. 26
    RnB = (delays.S_fall_neg - delays.S_fall_0 + epsilon) / (np.log(2) * config.C)

    return R5, RnA, RnB

# Eq. 28, 29, 30, 31
def compute_pmos_params(delays: MeasuredDelays, config: ModelConfig, R5):
    R = solve_R(delays.S_rise_0, delays.S_rise_pos, delays.S_rise_neg, R5, config.C, config.delta_min)

    alpha1 = A(delays.S_rise_neg - config.delta_min, R, R5, config.C)
    alpha2 = A(delays.S_rise_pos - config.delta_min, R, R5, config.C)

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
            A(S_rise_0 - delta_min, R, R5, C)
            - A(S_rise_pos - delta_min, R, R5, C)
            - A(S_rise_neg - delta_min, R, R5, C)
    )

# Eq. 28
def A(t, R, R5, C):
    K = (C * (R5 + 2*R) * np.log(2)) / t # repeating term

    w_arg = (K-1) * np.exp(K - 1)
    w_val = lambertw(w_arg, k=-1).real # W-1 (Lambert W with branch k = -1

    numerator = -2 * R * t * (1 - K)
    denominator = w_val + 1 - K

    return numerator/denominator


    

if __name__ == "__main__":
    delays, config = load_config("gate_params.toml")

    print("=== Loaded Delays ===")
    print(f"  δ↓_S(-∞) = {delays.S_fall_neg*1e12:.4f} ps")
    print(f"  δ↓_S(0)  = {delays.S_fall_0  *1e12:.4f} ps")
    print(f"  δ↓_S(+∞) = {delays.S_fall_pos*1e12:.4f} ps")
    print(f"  δ↑_S(-∞) = {delays.S_rise_neg*1e12:.4f} ps")
    print(f"  δ↑_S(0)  = {delays.S_rise_0  *1e12:.4f} ps")
    print(f"  δ↑_S(+∞) = {delays.S_rise_pos*1e12:.4f} ps")

    print("\n=== Model Config ===")
    print(f"  δ_min = {config.delta_min*1e15:.4f} fs")
    print(f"  C     = {config.C        *1e15:.4f} fF")

    params = parameterize(delays, config)

    print("\n=== Computed Parameters ===")
    print(f"  R5     = {params.R5    :.6f} Ω")
    print(f"  RnA    = {params.RnA   :.6f} Ω")
    print(f"  RnB    = {params.RnB   :.6f} Ω")
    print(f"  R      = {params.R     :.6f} Ω")
    print(f"  α₁     = {params.alpha1*1e12:.6f} ps")
    print(f"  α₂     = {params.alpha2*1e12:.6f} ps")