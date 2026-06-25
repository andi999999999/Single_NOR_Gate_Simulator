from pathlib import Path
import pytest
from nor_simulator.model.params import load_config, parameterize, eq29

CONFIG = Path(__file__).resolve().parent.parent / "gate_params.toml"

@pytest.fixture(scope="module")
def model():
    delays, physical = load_config(str(CONFIG))
    return parameterize(delays, physical), delays, physical

def test_resistances_positive(model):
    params, _, _ = model
    c = params.calculated
    assert c.R5 > 0 and c.RnA > 0 and c.RnB > 0 and c.R > 0
    assert c.alpha1 > 0 and c.alpha2 > 0

def test_derived_constants_positive(model):
    d = model[0].derived
    assert d.C1 > 0 and d.C1_p > 0 and d.C2 > 0 and d.C3 > 0
    assert d.tau3 > 0

def test_solve_R_satisfies_eq29(model):
    """R is a zero point of eq29 -> resudual has to be ~0"""
    params, delays, physical = model
    residual = eq29(
        delays.S_rise_0, delays.S_rise_pos, delays.S_rise_neg,
        params.calculated.R, params.calculated.R5,
        physical.C, physical.delta_min,
    )
    assert residual == pytest.approx(0.0, abs=1e-15)