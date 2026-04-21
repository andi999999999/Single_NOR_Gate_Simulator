''' Used for Random input generation'''
import numpy as np

from simulation_algorithm import InputTransition, InputState, algorithm1
from transition_visualizer import visualize_input_output_transitions
from parameter import NORModelParams, DerivedConstants, PhysicalParams, CalculatedParams, basic_sanity_test as parameter_basic_sanity_test

''' 
Steps: check paper for how long delays are, how long can they be before stabalizing? Check through the paper and make random inputs accordingly
Will need to get random inputs that fit in succession, like compatible switches, eg 1,1 cant follow 0,0, one switch after the other, need to keep to these rules

Es bietet sich an eine methode die komplett alles random erzeugt auf grundlage eines wertes, zb 5, der angibt wie viele werte ich haben will,
der erzeugt die dann und feeded sie automatisch an das modell... und das aber irgendwie im zweiten schritt vlt über eine koordinator funktion,
die entweder diesen wert erhält, oder einen pfad zu einem testfile, das werte enthält die eingelesen werden und direkt gefeeded werden...
'''

R, F, L, H = InputState.RISING, InputState.FALLING, InputState.LOW, InputState.HIGH

def generate_random_inputs(
        n_transitions: int,         # number of transitions to be generated
        max_delay: float = 4.1e-12, # max delay of any transition
        t_max_factor: float = 3.0,  # delay*factor = max delay that will be produced by random input generaition
        seed: int = None
) -> list[InputTransition]:
    rng = np.random.default_rng(seed)

    # random starting state
    x, y = rng.choice([0, 1]), rng.choice([0, 1])
    transitions = [
        InputTransition(
            x=H if x == 1 else L,
            y=H if y == 1 else L,
            t=float('-inf')
        )
    ]

    t = 0.0
    for _ in range(n_transitions):
        # choosing a random successor
        options, weights = _next_states(x, y)
        # normalizing weights to probabilities
        probs = np.array(weights) / sum(weights)
        idx = rng.choice(len(options), p=probs)
        new_x, new_y = options[idx]

        # input separation as exponential distribution
        # sometimes no distance (to have both switch at the same time) (for Δ=0 Tests)
        dt = rng.exponential(scale=max_delay * t_max_factor / 2)
        if rng.random() < 0.05:  # 5% Chance -> Δ=0
            dt = 0.0
        t += dt

        transitions.append(InputTransition(
            x=_input_state(x, new_x),
            y=_input_state(y, new_y),
            t=t
        ))

        x, y = new_x, new_y

    # Dummy at the end, ensuring last switch is not cancelled here.
    transitions.append(InputTransition(x=L, y=L, t=t + 1e-6))

    return transitions


""" Helper functions """
# converting transition to InputState enum - paper convention
def _input_state(prev: int, new: int):
    if prev == new:
        return H if prev == 1 else L
    return R if new == 1 else F

# determining valid successor states (only one Input may toggle)
def _next_states(x: int, y: int):
    options = [
        (1 - x, y),  # A toggles
        (x, 1 - y),  # B toggles
    ]

    # I am making the results biased here, prefering 0,0, avoiding 1,1... so we wont get stuck at output=0
    weights = []
    for nx, ny in options:
        if (nx, ny) == (0, 0):
            weights.append(2.0)  # prefere
        elif (nx, ny) == (1, 1):
            weights.append(1.0)
        else:
            weights.append(1.0)  # neutral

    return options, weights









# simple sanity check
def test_random_inputs():
    params, delays, physical = parameter_basic_sanity_test()

    input_transitions = generate_random_inputs(
        n_transitions=30,
        max_delay=4.1e-12,
        t_max_factor=3.0,
        seed=1
    )
    O, debug_infos = algorithm1(input_transitions, params, debug=True)

    visualize_input_output_transitions(input_transitions, O, debug_infos)


if __name__== "__main__":
    test_random_inputs()