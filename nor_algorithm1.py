
from __future__ import annotations

from dataclasses import dataclass
from math import inf, log
from typing import Callable, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple

CaseName = Literal["a", "b", "c", "d", "e", "f", "g", "h"]
Level = Literal[0, 1]


@dataclass(frozen=True)
class InputTransition:


    x: Level
    y: Level
    t: float


@dataclass(frozen=True)
class OutputTransition:
    value: Level
    t: float
    is_virtual: bool = False


class DelayTrajectoryModel(Protocol):


    delta_min: float
    vdd: float

    def delta_down(self, case: CaseName, vint: float) -> float:
        ...

    def delta_up(self, case: CaseName, delta: float, vint: float) -> float:
        ...

    def vout_case(self, case: CaseName, t: float, vint: float, delta: Optional[float] = None) -> float:
        ...


@dataclass
class ConstantToyModel:


    delta_min: float = 0.0
    vdd: float = 1.0
    tau_fall: float = 10.0
    tau_rise: float = 10.0

    def delta_down(self, case: CaseName, vint: float) -> float:
        vint = max(min(vint, self.vdd), 1e-15)
        return self.tau_fall * log(2.0 * vint / self.vdd) + self.delta_min

    def delta_up(self, case: CaseName, delta: float, vint: float) -> float:
        remaining = max(self.vdd - vint, 1e-15)
        base = self.tau_rise * log(2.0 * remaining / self.vdd) + self.delta_min
        return max(self.delta_min, base + 0.01 * abs(delta))

    def vout_case(self, case: CaseName, t: float, vint: float, delta: Optional[float] = None) -> float:
        if case in {"a", "b", "c", "d", "e", "f"}:
            return vint * (2.718281828459045 ** (-t / max(self.tau_fall, 1e-15)))
        if case in {"g", "h"}:
            return self.vdd + (vint - self.vdd) * (2.718281828459045 ** (-t / max(self.tau_rise, 1e-15)))
        raise ValueError(f"Unknown case {case}")


def detect_case(prev_x: Level, prev_y: Level, x: Level, y: Level) -> CaseName:
    transition = (prev_x, prev_y, x, y)
    table = {
        (0, 0, 1, 0): "a",
        (0, 0, 0, 1): "b",
        (1, 0, 1, 1): "c",
        (0, 1, 1, 1): "d",
        (1, 1, 0, 1): "e",
        (1, 1, 1, 0): "f",
        (0, 1, 0, 0): "g",
        (1, 0, 0, 0): "h",
    }
    try:
        return table[transition]
    except KeyError as exc:
        raise ValueError(f"Unsupported transition {transition}; the algorithm assumes only one input changes at a time.") from exc


def _remove_last_matching(outputs: List[OutputTransition], value: Level, t: float) -> None:
    for idx in range(len(outputs) - 1, -1, -1):
        item = outputs[idx]
        if item.value == value and item.t == t:
            outputs.pop(idx)
            return


def simulate_nor_algorithm1(
    inputs: Sequence[InputTransition],
    model: DelayTrajectoryModel,
) -> List[OutputTransition]:

    if len(inputs) < 2:
        raise ValueError("Need at least an initial state and one later transition.")

    O: List[OutputTransition] = []
    T = inf
    Delta = inf
    delta_ef_temp = -inf  # used for Delta in case g/h, matching the pseudocode

    current = inputs[0]
    prev_x, prev_y, _ = current.x, current.y, current.t

    if prev_x == 0 and prev_y == 0:
        O.append(OutputTransition(1, -inf))
        Vint = model.vdd
    else:
        O.append(OutputTransition(0, -inf))
        Vint = 0.0

    idx = 1
    while idx < len(inputs):
        nxt = inputs[idx]
        t_current = nxt.t
        case = detect_case(prev_x, prev_y, nxt.x, nxt.y)

        if case in {"a", "b", "c", "d"}:
            t_o = t_current + model.delta_down(case, Vint)
            O.append(OutputTransition(0, t_o))
            idx += 1
            t_next = inputs[idx].t if idx < len(inputs) else inf

            if (t_o - model.delta_min < t_current) or (t_o - model.delta_min > t_next):
                _remove_last_matching(O, 0, t_o)
                T = t_next - t_o
                Vint = model.vout_case(case, T + model.delta_min, Vint)

        elif case in {"e", "f"}:
            delta_ef_temp = t_current
            t_o = t_current + model.delta_down(case, Vint)
            O.append(OutputTransition(0, t_o))
            idx += 1
            t_next = inputs[idx].t if idx < len(inputs) else inf

            if (t_o - model.delta_min < t_current) or (t_o - model.delta_min > t_next):
                _remove_last_matching(O, 0, t_o)
                T = t_next - t_o
                Vint = model.vout_case(case, T + model.delta_min, Vint)

        elif case == "g":
            Delta = t_current - delta_ef_temp
            delta_ef_temp = t_current
            t_o = t_current + model.delta_up(case, Delta, Vint)
            O.append(OutputTransition(1, t_o))
            idx += 1
            t_next = inputs[idx].t if idx < len(inputs) else inf

            if (t_o - model.delta_min < t_current) or (t_o - model.delta_min > t_next):
                _remove_last_matching(O, 1, t_o)
                T = t_next - t_o
                if Vint <= model.vdd / 2.0:
                    Vint = model.vout_case(case, T + model.delta_min, Vint, Delta)
                else:
                    shifted_t = T - 2.0 * 10.0 * log(model.vdd / max(2.0 * (model.vdd - Vint), 1e-15)) + model.delta_min
                    Vint = model.vout_case(case, shifted_t, Vint, Delta)

        elif case == "h":
            Delta = delta_ef_temp - t_current
            delta_ef_temp = t_current
            t_o = t_current + model.delta_up(case, abs(Delta), Vint)
            O.append(OutputTransition(1, t_o))
            idx += 1
            t_next = inputs[idx].t if idx < len(inputs) else inf

            if (t_o - model.delta_min < t_current) or (t_o - model.delta_min > t_next):
                _remove_last_matching(O, 1, t_o)
                T = t_next - t_o
                if Vint <= model.vdd / 2.0:
                    Vint = model.vout_case(case, T + model.delta_min, Vint, abs(Delta))
                else:
                    shifted_t = T - 2.0 * 10.0 * log(model.vdd / max(2.0 * (model.vdd - Vint), 1e-15)) + model.delta_min
                    Vint = model.vout_case(case, shifted_t, Vint, abs(Delta))

        else:
            raise AssertionError(f"Unhandled case {case}")

        prev_x, prev_y = nxt.x, nxt.y

    return O


if __name__ == "__main__":
    inputs = [
        InputTransition(0, 0, -inf),
        InputTransition(1, 0, 0.0),
        InputTransition(1, 1, 5.0),
        InputTransition(0, 1, 12.0),
        InputTransition(0, 0, 18.0),
    ]

    model = ConstantToyModel(delta_min=1.0, vdd=1.0, tau_fall=8.0, tau_rise=10.0)
    outputs = simulate_nor_algorithm1(inputs, model)

    print("Output transitions:")
    for out in outputs:
        print(out)
