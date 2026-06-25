import argparse

from nor_simulator.algorithm import algorithm1
from nor_simulator.inputs import make_demo_inputs
from nor_simulator.model.params import parameterize, load_config  # parameter creation
from nor_simulator.reporting.console_report import print_transition_report
from nor_simulator.reporting.timing_diagram import plot_timing_diagram


def main():
    parser = argparse.ArgumentParser(description="Demo Trace Generator - prints a console-transition-report and generates a timing diagram (plot).")
    parser.add_argument("config", nargs="?", default="gate_params.toml",
                        help="Path to gate_params.toml (Default: gate_params.toml)")
    args = parser.parse_args()

    delays, physical = load_config(args.config)
    params = parameterize(delays, physical)

    inputs = make_demo_inputs()
    output, debug = algorithm1(inputs, params, debug=True)
    print_transition_report(inputs, output, debug)
    plot_timing_diagram(inputs, output, debug, vdd=params.physical.VDD)

if __name__ == "__main__":
    main()