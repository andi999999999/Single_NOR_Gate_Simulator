import matplotlib.pyplot as plt

from nor_simulator.transitions import InputTransition, OutputTransition, InputState


def plot_timing_diagram(
    input_transitions: list[InputTransition],
    output_transitions: list[OutputTransition],
    debug_infos: list[dict] = None,
    filename: str = "timing_diagram.png",
):
    # figuring out time window, removing start point (-infinity)
    real_times = [tr.t for tr in input_transitions if tr.t != float("-inf")]
    if len(real_times) > 1:
        span = max(t for t in real_times if t < real_times[-1])
    else:
        span = real_times[0]
    t_start = -0.05 * span
    t_end = 1.05 * span

    # InputState -> 0/1 mapping
    def level(state) -> int:
        return 1 if state in (InputState.RISING, InputState.HIGH) else 0

    def build_trace(get_level):
        times, levels = [t_start], [get_level(input_transitions[0])]
        for tr in input_transitions[1:]:
            t = tr.t if tr.t != float("-inf") else t_start
            if t > t_end:
                continue
            times.append(t)
            levels.append(get_level(tr))
        times.append(t_end)            # extending trace till the end
        levels.append(levels[-1])
        return times, levels

    a_t, a_l = build_trace(lambda tr: level(tr.x))
    b_t, b_l = build_trace(lambda tr: level(tr.y))

    # output trace (only real, non-canceled)
    o_t, o_l = [t_start], [output_transitions[0].o]
    for o in output_transitions:
        if o.t_p != float("-inf") and o.t_p <= t_end:
            o_t.append(o.t_p)
            o_l.append(o.o)
    o_t.append(t_end)
    o_l.append(o_l[-1])

    # generating plot, 3 traces same time scale
    fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True) #sharex: same x axis, figzize:
    traces = [
        (a_t, a_l, "Input A", "tab:blue"),
        (b_t, b_l, "Input B", "tab:green"),
        (o_t, o_l, "Output", "tab:red"),
    ]
    for ax, (t, l, label, color) in zip(axes, traces):
        ax.step([x * 1e12 for x in t], l, where="post", color=color, linewidth=2.2)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks([0, 1])
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # mark cancelled transitions as grey line
    if debug_infos:
        for d in debug_infos:
            if d["cancelled"]:
                axes[2].axvline(d["input_t"] * 1e12, color="gray", linestyle="--", linewidth=1, alpha=0.45)

    axes[-1].set_xlabel("Zeit [ps]", fontsize=11)
    fig.suptitle("NOR-Gate: Input/Output Trace", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    #plt.show()