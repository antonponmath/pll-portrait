from math import pi

import numpy as np

from pll_portrait.simulation import simulate

EPS = 0.001

def draw_comparison_portrait(system_left, system_right, t_max, axes):
    exclusion_upper = simulate(
        system_right,
        t_max=-t_max,
        initial_state=[pi - EPS, system_left.z_plus + EPS],
    )
    exclusion_lower = simulate(
        system_right,
        t_max=-t_max,
        initial_state=[-pi + EPS, system_left.z_minus - EPS],
    )
    lockin_upper = simulate(
        system_left,
        t_max=-t_max,
        initial_state=[pi - EPS, system_left.z_minus + EPS],
    )
    lockin_lower = simulate(
        system_left,
        t_max=-t_max,
        initial_state=[-pi + EPS, system_left.z_plus - EPS],
    )
    estimation = simulate(
        system_left,
        t_max=t_max,
        initial_state=[-pi + EPS, lockin_lower.trajectory[1, -1] / 2.0]
        if lockin_lower.trajectory[1, -1] > 0
        else [-pi + EPS, lockin_upper.trajectory[1, -1] / 2.0],
    )

    xlim = 1.15 * pi
    ylim = -1.1 * min(exclusion_lower.trajectory[1, :])

    for xshift in [-2 * pi, 0, 2 * pi]:
        # exclusion region fill
        axes.fill(
            np.append(exclusion_lower.trajectory[0, :], [pi, -pi]) + xshift,
            np.append(exclusion_lower.trajectory[1, :], [-ylim, -ylim]),
            "#fee",
        )
        axes.fill(
            np.append(exclusion_upper.trajectory[0, :], [-pi, pi]) + xshift,
            np.append(exclusion_upper.trajectory[1, :], [ylim, ylim]),
            "#fee",
        )

        # exclusion region bounds
        axes.plot(
            exclusion_upper.trajectory[0, :] + xshift,
            exclusion_upper.trajectory[1, :],
            color="red",
            linewidth=2,
        )
        axes.plot(
            exclusion_lower.trajectory[0, :] + xshift,
            exclusion_lower.trajectory[1, :],
            color="red",
            linewidth=2,
        )

        # lock-in domain fill
        if lockin_lower.trajectory[0, -1] > 0:
            axes.fill(
                np.append(lockin_lower.trajectory[0, :], lockin_upper.trajectory[0, :])
                + xshift,
                np.append(lockin_lower.trajectory[1, :], lockin_upper.trajectory[1, :]),
                "#eef",
            )
        else:
            axes.fill(
                lockin_lower.trajectory[0, :] + xshift,
                lockin_lower.trajectory[1, :],
                "#eef",
            )

        # lock-in domain bounds
        axes.plot(
            lockin_upper.trajectory[0, :] + xshift,
            lockin_upper.trajectory[1, :],
            color="blue",
            linewidth=2,
            linestyle="-" if lockin_lower.trajectory[0, -1] > 0 else ":",
        )
        axes.plot(
            lockin_lower.trajectory[0, :] + xshift,
            lockin_lower.trajectory[1, :],
            color="blue",
            linewidth=2,
        )

    # oscillation fill
    if estimation.cycle is not None:
        axes.fill(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            "#efe",
        )

    # sample locked-in trajectory
    axes.plot(
        estimation.trajectory[0, :],
        estimation.trajectory[1, :],
        color="#ccf",
        linewidth=1,
    )

    # oscillation bound
    if estimation.cycle is not None:
        axes.plot(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            color="green",
            linewidth=2,
        )

    # stationary points
    axes.plot(
        [0.0, 0.0],
        [system_left.z_minus, system_left.z_plus],
        linestyle="",
        marker="o",
        markersize=4,
        color="black",
    )
    axes.plot(
        [-pi, -pi, pi, pi],
        [
            system_left.z_minus,
            system_left.z_plus,
            system_left.z_minus,
            system_left.z_plus,
        ],
        linestyle="",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
    )

    axes.text(
        0.0,
        max(lockin_upper.trajectory[1, :]) / 2.0,
        "lock-in domain\nestimation",
        horizontalalignment="center",
        color="blue",
        family="monospace",
        size=10,
    )
    axes.set_xlim(xlim * np.array([-1, 1]))
    axes.set_ylim(ylim * np.array([-1, 1]))
    axes.set_xticks(
        [-pi, 0, pi], labels=["$-\\pi$", "$0$", "$\\pi$"], usetex=True, size=12
    )
    axes.set_yticks(np.linspace(-ylim, ylim, 7), labels=[])
    axes.grid(color="#ddd")
