from nor_simulator.algorithm import algorithm1
from nor_simulator.inputs import make_demo_inputs
from nor_simulator.model.params import basic_sanity_test as parameter_basic_sanity_test   # parameter creation
from nor_simulator.reporting.console_report import print_transition_report
from nor_simulator.reporting.timing_diagram import plot_timing_diagram


def main():
    params, _, _ = parameter_basic_sanity_test()
    inputs = make_demo_inputs()
    output, debug = algorithm1(inputs, params, debug=True)
    print_transition_report(inputs, output, debug)
    plot_timing_diagram(inputs, output, debug)

if __name__ == "__main__":
    main()