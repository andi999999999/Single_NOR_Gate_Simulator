"""
Note to self:

Meaning of variables:
- T: t_current-input - t_previous-output - Distance Previous Output to current Input
- t from V_out(t): time from begin of current trajectory
- Δ: = 𝑡_𝐵 − 𝑡_𝐴 - distance between last two *falling* input transitions
- t^current & t^next: absolute time when the current/next input transition happens in global time
- t^0: = t^current + 𝛿(V_int) absolute point in time when output transition happens
- I: sequence of input transitions

- x_i, y_i, t_i: x: input A, y:input B, t_i: global point in time of transistion.
- o_i and t'_i: o_i∈{0,1} - digital value of output transition, t'_i global point in time

Why N >= 2? first entry (x0, y0, t0) with t0 = -inf is initial state, only N = 2 is the first real transition.

What means
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from paper2_delay_formulas import (
    δ_case_a_f, δ_case_b_e, δ_case_c_d, δ_case_g, δ_case_h,
    Vout_case_a_f, Vout_case_b_e, Vout_case_c_d, Vout_case_g, Vout_case_h
)
from parameter import NORModelParams, DerivedConstants, PhysicalParams, CalculatedParams, basic_sanity_test as parameter_basic_sanity_test


class InputState(Enum):
    """Input signal states, syntax similar to paper"""
    RISING  = "↑"
    FALLING = "↓"
    LOW     = "0-"  # stable at 0
    HIGH    = "1-"  # stable at 1

@dataclass
class InputTransition:
    """One entry in the input sequence I = ((x_i, y_i, t_i))."""
    x: InputState   # input A
    y: InputState   # input B
    t: float        # transition time

@dataclass
class OutputTransition:
    o: int      # 0 or 1
    t_p: float    # Output transition time


class Case(Enum):
    A = ("a", δ_case_a_f, Vout_case_a_f)    # (↑, 0-, t)  →  (0,0) → (1,0)
    B = ("b", δ_case_b_e, Vout_case_b_e)    # (0-, ↑, t)  →  (0,0) → (0,1)
    C = ("c", δ_case_c_d, Vout_case_c_d)    # (1-, ↑, t)  →  (1,0) → (1,1)
    D = ("d", δ_case_c_d, Vout_case_c_d)    # (↑, 1-, t)  →  (0,1) → (1,1)
    E = ("e", δ_case_b_e, Vout_case_b_e)    # (↓, 1-, t)  →  (1,1) → (0,1)
    F = ("f", δ_case_a_f, Vout_case_a_f)    # (1-, ↓, t)  →  (1,1) → (1,0)
    G = ("g", δ_case_g, Vout_case_g)        # (0-, ↓, t)  →  (0,1) → (0,0)
    H = ("h", δ_case_h, Vout_case_h)        # (↓, 0-, t)  →  (1,0) → (0,0)

    def __init__(self, label, delay_func, Vout_func):
        self.label = label
        self.delay_func = delay_func
        self.Vout_func = Vout_func


def determine_case(x: InputState, y: InputState) -> Case:
    R, F, L, H = InputState.RISING, InputState.FALLING, InputState.LOW, InputState.HIGH
    case_map = {
        (R, L): Case.A,
        (L, R): Case.B,
        (H, R): Case.C,
        (R, H): Case.D,
        (F, H): Case.E,
        (H, F): Case.F,
        (L, F): Case.G,
        (F, L): Case.H,
    }
    return case_map[(x, y)]


def case_g_h_Vout_helper(params, Vint):
    tau3 = params.derived.tau3
    VDD = params.physical.VDD

    return tau3 * np.log(VDD / (2 * (VDD - Vint)))


# this code could be improved, basically represents algorithm1 from paper, but could use some optimization trough refactoring
def algorithm1(input_transitions: list[InputTransition], params: NORModelParams, debug=False):
    O: list[OutputTransition] = []
    debug_infos: list[dict] = []

    # Line 1: initializing variables
    T = float('inf')                # ∞
    delta = float('inf')             # ∞
    delta_e_temp = float('-inf')   # -∞
    delta_f_temp = float('-inf')   # -∞


    Vint: float

    VDD = params.physical.VDD
    delta_min = params.physical.delta_min

    # Line 2: current state
    current_transition = input_transitions[0]    # initial state, at t_0 = -∞ TODO: should I check that? if first trnasition meets that?
    assert current_transition.t == float('-inf'), "First transition must be at t = -∞"

    # Line 3 -6
    if (current_transition.x == InputState.LOW and current_transition.y == InputState.LOW):
        O.append(OutputTransition(o=1, t_p=float('-inf')))   # -∞
        Vint = VDD
    else:
        O.append(OutputTransition(o=0, t_p=float('-inf')))  # -∞
        Vint = 0.0

    # Line 7
    index_input = 1
    t_next = input_transitions[index_input].t # needed so i dont get a warning for while conditon...
    # Line 8 TODO: all if else are very similar, could outsource to another method, but not completely trivial as I need to update variables
    while t_next < input_transitions[-1].t:
        current_transition = input_transitions[index_input]
        t_next = input_transitions[index_input + 1].t

        case = determine_case(current_transition.x, current_transition.y)

        if case == Case.A or case == Case.B or case == Case.C or case == Case.D:
            delay = case.delay_func(Vint, params) # a, b, c, d requires these 2 arguments, this could be solved more estetically i am aware, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=0, t_p=t_o))

            T = t_next - t_o
            Vint = case.Vout_func(T + delta_min, params)

        elif case == Case.E or case == Case.F:
            if case == Case.E:
                delta_e_temp = current_transition.t
            elif case == Case.F:
                delta_f_temp = current_transition.t

            delay = case.delay_func(Vint, params) # e, f requires these 2 arguments, this could be solved more estetically i am aware, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=0, t_p=t_o))

            T = t_next - t_o
            Vint = case.Vout_func(T + delta_min, params)

        elif case == Case.G:
            if delta_e_temp == float('-inf'):
                delta = 1e6     # saturated: T₁ has always been open, This is necessary, because otherwise delay formula helpers chalculations (chi) would crash, this only executes if case G/H is the first case
            else:
                delta = current_transition.t - delta_e_temp
            delta_f_temp = current_transition.t # for computing delta in case h

            delay = case.delay_func(delta, Vint, params)  # g requires these 3 arguments, this could be solved more estetically i am aware, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=1, t_p=t_o))

            T = t_next - t_o
            # def Vout_case_g(t, delta, Vint, params, delay_g):
            if Vint <= VDD/2:
                Vint = case.Vout_func(T + delta_min, delta, Vint, params, delay)
            else:
                Vint = case.Vout_func(T - case_g_h_Vout_helper(params, Vint) + delta_min, delta, Vint, params, delay)
        elif case == Case.H:
            if delta_f_temp == float('-inf'):
                delta = -1e6    # saturated: T₁ has always been open, This is necessary, because otherwise delay formula helpers chalculations (chi) would crash, this only executes if case G/H is the first case
            else:
                delta = delta_f_temp - current_transition.t
            delta_e_temp = current_transition.t  # for computing delta in case g

            delay = case.delay_func(delta, Vint, params)  # g requires these 3 arguments, this could be solved more estetically i am aware, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=1, t_p=t_o))

            T = t_next - t_o
            # def Vout_case_g(t, delta, Vint, params, delay_g):
            if Vint <= VDD / 2:
                Vint = case.Vout_func(T + delta_min, delta, Vint, params, delay)
            else:
                Vint = case.Vout_func(T - case_g_h_Vout_helper(params, Vint) + delta_min, delta, Vint, params, delay)

        if debug:
            debug_infos.append({
                "case": case.label,
                "t_o": t_o,
                "delay": delay,
                "Vint": Vint,
                "cancelled": is_cancelled,
                "input_t": current_transition.t,
            })

        index_input += 1

    if debug:
        return O, debug_infos
    return O


"""        
        (R, L): Case.A,
        (L, R): Case.B,
        (H, R): Case.C,
        (R, H): Case.D,
        (F, H): Case.E,
        (H, F): Case.F,
        (L, F): Case.G,
        (F, L): Case.H,
        
"""



if __name__ == "__main__":
    basic_sanity_check()

