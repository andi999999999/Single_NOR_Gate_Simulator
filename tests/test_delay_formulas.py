from pathlib import Path
import pytest
from nor_nand_simulator.model.params import load_config, parameterize_nor
from nor_nand_simulator.model.delay_formulas import (
    δ_case_a_f, δ_case_b_e, δ_case_c_d, δ_case_g, δ_case_h,
)

CONFIG = Path(__file__).resolve().parent.parent / "nor_gate_params.toml"

@pytest.fixture(scope="module")
def model():
    delays, physical = load_config(str(CONFIG))
    params = parameterize_nor(delays, physical)
    return params, delays

# round-trip tests: the parameters were calculated from these delays, out formulas have to reproduce them
def test_roundtrip_falling(model):
    params, delays = model
    VDD = params.physical.VDD
    assert δ_case_b_e(VDD, params) == pytest.approx(delays.S_fall_neg, rel=1e-9)
    assert δ_case_c_d(VDD, params) == pytest.approx(delays.S_fall_0,   rel=1e-9)
    assert δ_case_a_f(VDD, params) == pytest.approx(delays.S_fall_pos, rel=1e-9)

def test_roundtrip_rising(model):
    params, delays = model
    assert δ_case_h(-1e6, 0, params) == pytest.approx(delays.S_rise_neg, rel=1e-6)
    assert δ_case_g( 1e6, 0, params) == pytest.approx(delays.S_rise_pos, rel=1e-6)
    # δ↑_S(0): case g and case h at Δ=0 both have to equal S_rise_0
    assert δ_case_g(0, 0, params) == pytest.approx(delays.S_rise_0, rel=1e-6)
    assert δ_case_h(0, 0, params) == pytest.approx(delays.S_rise_0, rel=1e-6)