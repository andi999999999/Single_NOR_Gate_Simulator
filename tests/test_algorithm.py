import pytest

from nor_nand_simulator.paths import DEFAULT_NOR_CONFIG, DEFAULT_NAND_CONFIG
from nor_nand_simulator.transitions import InputState, InputTransition
from nor_nand_simulator.model.params import load_config, parameterize_nor, parameterize_nand
from nor_nand_simulator.algorithm import simulate_nor, determine_case, Case, simulate_nand, _NEGATE

R, F, L, H = InputState.RISING, InputState.FALLING, InputState.LOW, InputState.HIGH
NINF = float("-inf")
DUMMY = InputTransition(x=L, y=L, t=1e-6)
GAP = 1e-9

""" ===== NOR Simulator Tests ===== """
@pytest.fixture(scope="module")
def nor_model():
    delays, physical = load_config(str(DEFAULT_NOR_CONFIG))
    return parameterize_nor(delays, physical), delays

def _run_nor(params, *transitions):
    """runs algorithm, returns real outputs (removes -inf initial state)."""
    nor_output_transitions, _ = simulate_nor(list(transitions), params)
    return [o for o in nor_output_transitions if o.t_p != NINF]

""" -----round-trip test through the algorithm, each case testing----- """

# δ↓_S(+∞): only A rises (Case a)
def test_fall_pos_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=R, y=L, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_fall_pos, rel=1e-9)

# δ↓_S(-∞): only B rises (Case b)
def test_fall_neg_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=L, y=R, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_fall_neg, rel=1e-9)

# δ↓_S(0): both rise at the same time (Case 1. a & 2. c)
def test_fall_zero_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=R, y=L, t=0.0), # Case a
                    InputTransition(x=H, y=R, t=0.0), DUMMY) # Case c
    assert real[-1].t_p == pytest.approx(delays.S_fall_0, rel=1e-9)

# δ↑_S(0): both falling at the same time (Case 1. e & 2. g)
def test_rise_zero_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=F, y=H, t=0.0), # Case e
                    InputTransition(x=L, y=F, t=0.0), DUMMY) # Case g
    assert real[-1].t_p == pytest.approx(delays.S_rise_0, rel=1e-6)

# δ↑_S(+∞): A falls a long time before B (Case e → g)
def test_rise_pos_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=F, y=H, t=0.0), # Case e
                    InputTransition(x=L, y=F, t=GAP), DUMMY) # Case g
    assert real[-1].t_p - GAP == pytest.approx(delays.S_rise_pos, rel=1e-6)

# δ↑_S(-∞): B falls long before A (Case f → h)
def test_rise_neg_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=H, y=F, t=0.0), # Case f
                    InputTransition(x=F, y=L, t=GAP), DUMMY) # Case h
    assert real[-1].t_p - GAP == pytest.approx(delays.S_rise_neg, rel=1e-6)

# one extra for case d:
# δ↓_S(0) via Case b → d
def test_fall_zero_case_d_nor(nor_model):
    params, delays = nor_model
    real = _run_nor(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=L, y=R, t=0.0),       # Case b
                    InputTransition(x=R, y=H, t=0.0), DUMMY)  # Case d
    assert real[-1].t_p == pytest.approx(delays.S_fall_0, rel=1e-9)

""" ------ Test for case mappings ---------- """

@pytest.mark.parametrize("x, y, expected", [
    (R, L, Case.A), (L, R, Case.B), (H, R, Case.C), (R, H, Case.D),
    (F, H, Case.E), (H, F, Case.F), (L, F, Case.G), (F, L, Case.H),
])
def test_determine_case_nor(x, y, expected):
    assert determine_case(x, y) is expected

"""---- initial states (Line 3–6) ----"""

def test_initial_state_low_low_nor(nor_model):
    params, _ = nor_model
    nor_output_transitions, _ = simulate_nor([InputTransition(x=L, y=L, t=NINF),
                                              InputTransition(x=R, y=L, t=0.0), DUMMY], params)
    assert nor_output_transitions[0].o == 1 and nor_output_transitions[0].t_p == NINF # LL → Output starts at 1 (Vint=VDD)

def test_initial_state_not_low_low_nor(nor_model):
    params, _ = nor_model
    nor_output_transitions, _ = simulate_nor([InputTransition(x=H, y=H, t=NINF),
                                              InputTransition(x=F, y=H, t=0.0), DUMMY], params)
    assert nor_output_transitions[0].o == 0 and nor_output_transitions[0].t_p == NINF  # other cases → Output starts at0



""" ===== NAND Simulator Tests ===== """
@pytest.fixture(scope="module")
def nand_model():
    delays, physical = load_config(str(DEFAULT_NAND_CONFIG))
    return parameterize_nand(delays, physical), delays

def _run_nand(params, *transitions):
    """runs algorithm, returns real outputs (removes -inf initial state)."""
    nand_output_transitions, _ = simulate_nand(list(transitions), params)
    return [o for o in nand_output_transitions if o.t_p != NINF]

""" -----round-trip test through the algorithm, each case testing----- """

# δ↑_S(+∞): only A falls (Case e)
def test_rise_pos_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=F, y=H, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_rise_pos, rel=1e-9)

# δ↑_S(-∞): only B falls (Case f)
def test_rise_neg_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=H, y=F, t=0.0), DUMMY)
    assert real[0].t_p == pytest.approx(delays.S_rise_neg, rel=1e-9)

# δ↓_S(0): both rise at the same time (Case 1. a & 2. c)
def test_fall_zero_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=R, y=L, t=0.0), # Case a
                    InputTransition(x=H, y=R, t=0.0), DUMMY) # Case c
    assert real[-1].t_p == pytest.approx(delays.S_fall_0, rel=1e-9)

# δ↑_S(0): both falling at the same time (Case 1. e & 2. g)
def test_rise_zero_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=F, y=H, t=0.0), # Case e
                    InputTransition(x=L, y=F, t=0.0), DUMMY) # Case g
    assert real[-1].t_p == pytest.approx(delays.S_rise_0, rel=1e-6)

# δ↓_S(+∞): A rises a long time before B (Case a → c)
def test_fall_pos_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=R, y=L, t=0.0), # Case a
                    InputTransition(x=H, y=R, t=GAP), DUMMY) # Case c
    assert real[-1].t_p - GAP == pytest.approx(delays.S_fall_pos, rel=1e-6)

# δ↓_S(-∞): B rises long before A (Case b → d)
def test_fall_neg_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=L, y=L, t=NINF),
                    InputTransition(x=L, y=R, t=0.0), # Case b
                    InputTransition(x=R, y=H, t=GAP), DUMMY) # Case d
    assert real[-1].t_p - GAP == pytest.approx(delays.S_fall_neg, rel=1e-6)

# one extra for case h:
# δ↑_S(0) via Case f → h
def test_rise_zero_case_h_nand(nand_model):
    params, delays = nand_model
    real = _run_nand(params,
                    InputTransition(x=H, y=H, t=NINF),
                    InputTransition(x=H, y=F, t=0.0),  # Case f
                    InputTransition(x=F, y=L, t=0.0), DUMMY)  # Case h
    assert real[-1].t_p == pytest.approx(delays.S_rise_0, rel=1e-9)


def test_nand_is_negated_nor(nand_model):
    params, _ = nand_model
    scenario = [
        InputTransition(x=H, y=H, t=NINF),
        InputTransition(x=F, y=H, t=0.0),
        InputTransition(x=L, y=F, t=GAP),
        DUMMY,
    ]
    negated = [InputTransition(x=_NEGATE[t.x], y=_NEGATE[t.y], t=t.t) for t in scenario]

    nand_out, _ = simulate_nand(scenario, params)
    nor_out, _  = simulate_nor(negated, params)

    assert len(nand_out) == len(nor_out)
    for n, r in zip(nand_out, nor_out):
        assert n.t_p == r.t_p
        assert n.o == 1 - r.o