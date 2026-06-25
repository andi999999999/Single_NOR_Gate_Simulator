"""
"Line" in the code referes to the line of the Algorithm in the Paper

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
"""
import argparse
from enum import Enum

import numpy as np

from nor_simulator.transitions import InputState, InputTransition, OutputTransition
from nor_simulator.model.delay_formulas import (
    δ_case_a_f, δ_case_b_e, δ_case_c_d, δ_case_g, δ_case_h,
    Vout_case_a_f, Vout_case_b_e, Vout_case_c_d, Vout_case_g, Vout_case_h, rising_trajectory_time_offset
)
from nor_simulator.model.params import NORModelParams, load_config, parameterize


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

def sample_segment(vout_local, t_lo, t_hi, origin, n=50):
    if t_hi <= t_lo:                 # nichts Sinnvolles zu zeichnen (z.B. überholte Segmente)
        return [], []
    tl = np.linspace(t_lo, t_hi, n)
    vs = vout_local(tl)                       # vectorising
    return (tl + origin).tolist(), np.asarray(vs).tolist()

# TODO: this code could be improved, basically represents algorithm1 from paper, but could use some optimization trough refactoring
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
    current_transition = input_transitions[0]    # initial state, at t_0 = -∞
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
            delay = case.delay_func(Vint, params) # a, b, c, d requires these 2 arguments, this could be solved more estetically, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=0, t_p=t_o))

            T = t_next - t_o
            Vint = case.Vout_func(T + delta_min, params)


            # for generating V_out traces
            seg_t, seg_v = [], []
            if debug:
                vout_local = lambda tl: case.Vout_func(tl, params)
                seg_t, seg_v = sample_segment(
                    vout_local, t_lo=delta_min - delay, t_hi=T + delta_min,
                    origin=t_o - delta_min,
                )

        elif case == Case.E or case == Case.F:
            if case == Case.E:
                delta_e_temp = current_transition.t
            elif case == Case.F:
                delta_f_temp = current_transition.t

            delay = case.delay_func(Vint, params) # e, f requires these 2 arguments, this could be solved more estetically, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=0, t_p=t_o))

            T = t_next - t_o
            Vint = case.Vout_func(T + delta_min, params)


            # for generating V_out traces
            seg_t, seg_v = [], []
            if debug:
                vout_local = lambda tl: case.Vout_func(tl, params)
                seg_t, seg_v = sample_segment(
                    vout_local, t_lo=delta_min - delay, t_hi=T + delta_min,
                    origin=t_o - delta_min,
                )

        elif case == Case.G:
            if delta_e_temp == float('-inf'):
                delta = 1e6     # saturated: T₁ has always been open, This is necessary, because otherwise delay formula helpers chalculations (chi) would crash, this only executes if case G/H is the first case
            else:
                delta = current_transition.t - delta_e_temp
            delta_f_temp = current_transition.t # for computing delta in case h

            delay = case.delay_func(delta, Vint, params)  # g requires these 3 arguments, this could be solved more estetically, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=1, t_p=t_o))

            T = t_next - t_o
            # def Vout_case_g(t, delta, Vint, params, delay_g):
            vint_start = Vint
            offset = 0.0 if vint_start <= VDD / 2 else rising_trajectory_time_offset(params, vint_start)
            arg_next = T - offset + delta_min
            Vint = case.Vout_func(arg_next, delta, vint_start, params, delay)



            # for generating v_out traces
            seg_t, seg_v = [], []
            if debug:
                vout_local = lambda tl: case.Vout_func(tl, delta, vint_start, params, delay)
                # The rising Vout formulas are only valid forward from the VDD/2 crossing (t >= 0).
                # For negative t, e_factor = np.exp(-t / tau3) grows without bound (exp of a positive number),
                # which pushes the sampled curve far past the rails -> a spike in the plot.
                # So for the rising cases we clamp the sample window to t_lo = 0.0 (the crossing, where e_factor = 1, Vout = VDD/2).
                # Falling needs no clamp: there Vout(t_lo) evaluates exactly to Vint, so it stays bounded.
                seg_t, seg_v = sample_segment(
                    vout_local,  t_lo=max(0.0, delta_min - offset - delay), t_hi=arg_next,
                    origin=t_o - delta_min + offset,
                )

        elif case == Case.H:
            if delta_f_temp == float('-inf'):
                delta = -1e6    # saturated: T₁ has always been open, This is necessary, because otherwise delay formula helpers chalculations (chi) would crash, this only executes if case G/H is the first case
            else:
                delta = delta_f_temp - current_transition.t
            delta_e_temp = current_transition.t  # for computing delta in case g

            delay = case.delay_func(delta, Vint, params)  # g requires these 3 arguments, this could be solved more estetically, but is pragmatic
            t_o = current_transition.t + delay

            # Line 13 + Line 15 on
            # Cancellation check, so I dont have to add first and them remove again.
            is_cancelled = (t_o - delta_min < current_transition.t) or (t_o - delta_min > t_next)
            if not is_cancelled:
                O.append(OutputTransition(o=1, t_p=t_o))

            T = t_next - t_o
            # def Vout_case_g(t, delta, Vint, params, delay_g):
            vint_start = Vint
            offset = 0.0 if vint_start <= VDD / 2 else rising_trajectory_time_offset(params, vint_start)
            arg_next = T - offset + delta_min
            Vint = case.Vout_func(arg_next, delta, vint_start, params, delay)



            # for generating v_out traces
            seg_t, seg_v = [], []
            if debug:
                vout_local = lambda tl: case.Vout_func(tl, delta, vint_start, params, delay)
                # The rising Vout formulas are only valid forward from the VDD/2 crossing (t >= 0).
                # For negative t, e_factor = np.exp(-t / tau3) grows without bound (exp of a positive number),
                # which pushes the sampled curve far past the rails -> a spike in the plot.
                # So for the rising cases we clamp the sample window to t_lo = 0.0 (the crossing, where e_factor = 1, Vout = VDD/2).
                # Falling needs no clamp: there Vout(t_lo) evaluates exactly to Vint, so it stays bounded.
                seg_t, seg_v = sample_segment(
                    vout_local,  t_lo=max(0.0, delta_min - offset - delay), t_hi=arg_next,
                    origin=t_o - delta_min + offset,
                )

        if debug:
            debug_infos.append({
                "case": case.label,
                "t_o": t_o,
                "delay": delay,
                "Vint": Vint,
                "cancelled": is_cancelled,
                "input_t": current_transition.t,
                "vout_t": seg_t, "vout_v": seg_v,
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

def print_algorithm_report():
    parser = argparse.ArgumentParser(description="Algorithm report - comparing calculated to real delays side by side")
    parser.add_argument("config", nargs="?", default="gate_params.toml",
                        help="Path to gate_params.toml (Default: gate_params.toml)")
    args = parser.parse_args()

    delays, physical = load_config(args.config)
    params = parameterize(delays, physical)
    GAP = 1e-9
    R, F, L, H = InputState.RISING, InputState.FALLING, InputState.LOW, InputState.HIGH
    DUMMY = InputTransition(x=L, y=L, t=1e-6)

    print("=== Algorithm report: MIS-Delays calculated using Algorithm 1 ===\n")

    # === Test 1: δ↓_S(+∞): only A rises (Case a) ===
    inputs = [
        InputTransition(x=L, y=L, t=float('-inf')),
        InputTransition(x=R, y=L, t=0.0),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    print(f"δ↓_S(+∞):  expected={delays.S_fall_pos*1e12:.4f} ps,  got={real[0].t_p*1e12:.4f} ps")

    # === Test 2: δ↓_S(-∞): only B rises (Case b) ===
    inputs = [
        InputTransition(x=L, y=L, t=float('-inf')),
        InputTransition(x=L, y=R, t=0.0),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    print(f"δ↓_S(-∞):  expected={delays.S_fall_neg*1e12:.4f} ps,  got={real[0].t_p*1e12:.4f} ps")

    # === Test 3: δ↓_S(0):both rise at the same time ===
    inputs = [
        InputTransition(x=L, y=L, t=float('-inf')),
        InputTransition(x=R, y=L, t=0.0),
        InputTransition(x=H, y=R, t=0.0),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    print(f"δ↓_S(0):   expected={delays.S_fall_0*1e12:.4f} ps,  got={real[-1].t_p*1e12:.4f} ps")

    # === Test 4: δ↑_S(0): both falling at the same time ===
    inputs = [
        InputTransition(x=H, y=H, t=float('-inf')),
        InputTransition(x=F, y=H, t=0.0),
        InputTransition(x=L, y=F, t=0.0),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    print(f"δ↑_S(0):   expected={delays.S_rise_0*1e12:.4f} ps,  got={real[-1].t_p*1e12:.4f} ps")

    # === Test 5: δ↑_S(+∞): A falls a long time before B ===
    inputs = [
        InputTransition(x=H, y=H, t=float('-inf')),
        InputTransition(x=F, y=H, t=0.0),
        InputTransition(x=L, y=F, t=GAP),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    delay = real[-1].t_p - GAP
    print(f"δ↑_S(+∞):  expected={delays.S_rise_pos*1e12:.4f} ps,  got={delay*1e12:.4f} ps")

    # === Test 6: δ↑_S(-∞): B falls long before A ===
    inputs = [
        InputTransition(x=H, y=H, t=float('-inf')),
        InputTransition(x=H, y=F, t=0.0),
        InputTransition(x=F, y=L, t=GAP),
        DUMMY,
    ]
    O = algorithm1(inputs, params)
    real = [o for o in O if o.t_p != float('-inf')]
    delay = real[-1].t_p - GAP
    print(f"δ↑_S(-∞):  expected={delays.S_rise_neg*1e12:.4f} ps,  got={delay*1e12:.4f} ps")


if __name__ == "__main__":
    print_algorithm_report()

