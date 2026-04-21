"""
Supposed to get inputs and outputs into useful, readable format, marking if a transition is cancelled, between which input transitions an output ransition happened etc.
"""
from simulation_algorithm import InputTransition, OutputTransition


def visualize_input_output_transitions(
        input_transitions: list[InputTransition],
        output_transitions: list[OutputTransition],
        debug_infos: list[dict] = None
):
    print("\nStatistics:")
    print(f"Num. input transitions: {len(input_transitions)}")
    print(f"Num. output transitions: {len(output_transitions)}")
    if debug_infos:
        cancelled_count = sum(1 for d in debug_infos if d["cancelled"])
        print(f"  Num. cancelled: {cancelled_count}")
        print(f"  Num. real transitions: {len(debug_infos) - cancelled_count}")
    print("\n")

    # Initial Transitions:
    print("Initial transition pair:")
    print_input_transition(input_transitions[0])
    print_output_transition(output_transitions[0])  # t = -∞
    print()

    idx_out = 1

    # Transitions excluding initial and last (only indicates if cancelled or not)
    for i in range(1, len(input_transitions) - 1):
        debug = debug_infos[i - 1] if debug_infos else None

        print_input_transition(input_transitions[i])

        # checking if output transition belongs to this input transition, or if it is cancelled
        t_next_input = input_transitions[i + 1].t

        if idx_out < len(output_transitions) and output_transitions[idx_out].t_p < t_next_input :
            print_output_transition(output_transitions[idx_out], debug)
            idx_out += 1
        else:
            print_cancelled(debug)

        print()

    # Dummy/Final transition at the end:
    print("\nFinal Input transition (Dummy/Output cancellation):")
    print_input_transition(input_transitions[-1])



def print_input_transition(input_transition: InputTransition):
    print(f"  Input:  A: {input_transition.x.value}, B: {input_transition.y.value}, "
          f"t: {input_transition.t * 1e12:.4f} ps")

def print_output_transition(output_transition: OutputTransition, debug: dict = None):
    line = f"  Output: → {output_transition.o} at t: {output_transition.t_p * 1e12:.4f} ps"

    if debug:
        line += f"  |  Case {debug['case']}, Vint={debug['Vint']:.6f}, delay={debug['delay'] * 1e12:.4f} ps"

    print(line)


def print_cancelled(debug: dict = None):
    line = "  Output: → CANCELLED"

    if debug:
        line += f"  |  Case {debug['case']}, Vint={debug['Vint']:.6f}, t_o={debug['t_o'] * 1e12:.4f} ps, delay={debug['delay'] * 1e12:.4f} ps"

    print(line)