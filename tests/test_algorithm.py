from pathlib import Path
import pytest
from nor_simulator.transitions import InputState, InputTransition
from nor_simulator.model.params import load_config, parameterize
from nor_simulator.algorithm import simulate_nor, determine_case, Case

CONFIG = Path(__file__).resolve().parent.parent / "gate_params.toml"
R, F, L, H = InputState.RISING, InputState.FALLING, InputState.LOW, InputState.HIGH
NINF = float("-inf")
DUMMY = InputTransition(x=L, y=L, t=1e-6)
GAP = 1e-9

@pytest.fixture(scope="module")
def model():
    delays, physical = load_config(str(CONFIG))
    return parameterize(delays, physical), delays

def _run(params, *transitions):
    """runs algorithm, returns real outputs (removes -inf initial state)."""
    nor_output_transitions, _ = simulate_nor(list(transitions), params)
    return [o for o in nor_output_transitions if o.t_p != NINF]

""" -----round-trip test through the algorithm, each case testing----- """

# δ↓_S(+∞): only A rises (Case a)
def test_fall_pos(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=L, y=L, t=NINF),
        InputTransition(x=R, y=L, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_fall_pos, rel=1e-9)

# δ↓_S(-∞): only B rises (Case b)
def test_fall_neg(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=L, y=L, t=NINF),
        InputTransition(x=L, y=R, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_fall_neg, rel=1e-9)

# δ↓_S(0): both rise at the same time
def test_fall_zero(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=L, y=L, t=NINF),
        InputTransition(x=R, y=L, t=0.0),
        InputTransition(x=H, y=R, t=0.0), DUMMY)
    assert real[-1].t_p == pytest.approx(delays.S_fall_0, rel=1e-9)

# δ↑_S(0): both falling at the same time
def test_rise_zero(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=H, y=H, t=NINF),
        InputTransition(x=F, y=H, t=0.0),
        InputTransition(x=L, y=F, t=0.0), DUMMY)
    assert real[-1].t_p == pytest.approx(delays.S_rise_0, rel=1e-6)

# δ↑_S(+∞): A falls a long time before B
def test_rise_pos(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=H, y=H, t=NINF),
        InputTransition(x=F, y=H, t=0.0),
        InputTransition(x=L, y=F, t=GAP), DUMMY)
    assert real[-1].t_p - GAP == pytest.approx(delays.S_rise_pos, rel=1e-6)

# δ↑_S(-∞): B falls long before A
def test_rise_neg(model):
    params, delays = model
    real = _run(params,
        InputTransition(x=H, y=H, t=NINF),
        InputTransition(x=H, y=F, t=0.0),
        InputTransition(x=F, y=L, t=GAP), DUMMY)
    assert real[-1].t_p - GAP == pytest.approx(delays.S_rise_neg, rel=1e-6)

""" ------ Test for case mappings ---------- """

@pytest.mark.parametrize("x, y, expected", [
    (R, L, Case.A), (L, R, Case.B), (H, R, Case.C), (R, H, Case.D),
    (F, H, Case.E), (H, F, Case.F), (L, F, Case.G), (F, L, Case.H),
])
def test_determine_case(x, y, expected):
    assert determine_case(x, y) is expected

"""---- initial states (Line 3–6) ----"""

def test_initial_state_low_low(model):
    params, _ = model
    nor_output_transitions, _ = simulate_nor([InputTransition(x=L, y=L, t=NINF),
                                              InputTransition(x=R, y=L, t=0.0), DUMMY], params)
    assert nor_output_transitions[0].o == 1 and nor_output_transitions[0].t_p == NINF # LL → Output starts at 1 (Vint=VDD)

def test_initial_state_not_low_low(model):
    params, _ = model
    nor_output_transitions, _ = simulate_nor([InputTransition(x=H, y=H, t=NINF),
                                              InputTransition(x=F, y=H, t=0.0), DUMMY], params)
    assert nor_output_transitions[0].o == 0 and nor_output_transitions[0].t_p == NINF  # other cases → Output starts at0