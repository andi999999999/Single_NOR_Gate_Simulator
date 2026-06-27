import argparse

from nor_nand_simulator.algorithm import simulate_nor, simulate_nand
from nor_nand_simulator.inputs import make_demo_inputs_nor, make_demo_inputs_nand
from nor_nand_simulator.model.params import parameterize_nor, load_config, parameterize_nand  # parameter creation
from nor_nand_simulator.reporting.console_report import print_transition_report
from nor_nand_simulator.reporting.timing_diagram import plot_timing_diagram_nand, plot_timing_diagram_nor


def generate_nor_trace(config="nor_gate_params.toml"):
    delays, physical = load_config(config)
    params = parameterize_nor(delays, physical)

    inputs = make_demo_inputs_nor()
    nor_output_transitions, debug = simulate_nor(inputs, params, debug=True)
    print_transition_report(inputs, nor_output_transitions, debug)
    plot_timing_diagram_nor(inputs, nor_output_transitions, debug, vdd=params.physical.VDD)

def generate_nand_trace(config="nand_gate_params.toml"):
    delays, physical = load_config(config)
    params = parameterize_nand(delays, physical)

    inputs = make_demo_inputs_nand()
    nand_output_transitions, debug = simulate_nand(inputs, params, debug=True)
    print_transition_report(inputs, nand_output_transitions, debug)
    plot_timing_diagram_nand(inputs, nand_output_transitions, debug, vdd=params.physical.VDD)

def main():
    parser = argparse.ArgumentParser(
        description="Demo Trace Generator – Console-Report + Timing-Diagram per Gate."
    )
    parser.add_argument("--gate", choices=["nor", "nand", "both"], default="both",
                        help="What traces are generated (Default: both).")
    parser.add_argument("--nor-config", default="nor_gate_params.toml",
                        help="Path to NOR-Params-TOML.")
    parser.add_argument("--nand-config", default="nand_gate_params.toml",
                        help="Path to NAND-Params-TOML.")
    args = parser.parse_args()

    if args.gate in ("nor", "both"):
        generate_nor_trace(args.nor_config)
    if args.gate in ("nand", "both"):
        generate_nand_trace(args.nand_config)

if __name__ == "__main__":
    main()