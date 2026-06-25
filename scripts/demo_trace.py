import argparse

from nor_simulator.algorithm import simulate_nor
from nor_simulator.inputs import make_demo_inputs
from nor_simulator.model.params import parameterize, load_config  # parameter creation
from nor_simulator.reporting.console_report import print_transition_report
from nor_simulator.reporting.timing_diagram import plot_timing_diagram


def generate_nor_trace():
    parser = argparse.ArgumentParser(description="Demo Trace Generator - prints a console-transition-report and generates a timing diagram (plot).")
    parser.add_argument("config", nargs="?", default="gate_params.toml",
                        help="Path to gate_params.toml (Default: gate_params.toml)")
    args = parser.parse_args()

    delays, physical = load_config(args.config)
    params = parameterize(delays, physical)

    inputs = make_demo_inputs()
    nor_output_transitions, debug = simulate_nor(inputs, params, debug=True)
    print_transition_report(inputs, nor_output_transitions, debug)
    plot_timing_diagram(inputs, nor_output_transitions, debug, vdd=params.physical.VDD)

if __name__ == "__main__":
    generate_nor_trace()