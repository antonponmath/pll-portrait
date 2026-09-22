from math import pi

import numpy as np

from pll_portrait.simulation import simulate

EPS = 0.001

RED = "#b70404"
GREEN = "#069a8e"
BLUE = "#2155cd"
FILL_RED = "#f1cdcd"
FILL_GREEN = "#b4e1dd"
FILL_BLUE = "#d3ddf5"
LINE_BLUE = "#a6bbeb"
GRID_COLOR = "#ccc"


def draw_comparison_portrait(system_left, system_right, t_max, axes):
    slipping_upper = simulate(
        system_right,
        t_max=-t_max,
        initial_state=[pi - EPS, system_left.z_plus + EPS],
    )
    slipping_lower = simulate(
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

    lockin_wrapped = lockin_lower.escaped and lockin_lower.trajectory[0, -1] < 0
    lockin_cycled = not lockin_lower.escaped

    if not lockin_cycled:
        estimation = simulate(
            system_left,
            t_max=t_max,
            initial_state=[
                -pi + EPS,
                (lockin_lower.trajectory[1, -1] + system_left.z_plus) / 2.0,
            ]
            if lockin_wrapped
            else [
                -pi + EPS,
                (lockin_upper.trajectory[1, -1] + system_left.z_plus) / 2.0,
            ],
        )

    xlim = 1.15 * pi
    ylim = -1.2 * min(slipping_lower.trajectory[1, :])

    for xshift in [-2 * pi, 0, 2 * pi]:
        # slipping region fill
        axes.fill(
            np.append(slipping_lower.trajectory[0, :], [pi, -pi]) + xshift,
            np.append(slipping_lower.trajectory[1, :], [-ylim, -ylim]),
            FILL_RED,
        )
        axes.fill(
            np.append(slipping_upper.trajectory[0, :], [-pi, pi]) + xshift,
            np.append(slipping_upper.trajectory[1, :], [ylim, ylim]),
            FILL_RED,
        )

        # slipping region bounds
        axes.plot(
            slipping_upper.trajectory[0, :] + xshift,
            slipping_upper.trajectory[1, :],
            color=RED,
            linewidth=2,
        )
        axes.plot(
            slipping_lower.trajectory[0, :] + xshift,
            slipping_lower.trajectory[1, :],
            color=RED,
            linewidth=2,
        )

        # lock-in domain fill
        if not lockin_cycled:
            if lockin_wrapped:
                axes.fill(
                    lockin_lower.trajectory[0, :] + xshift,
                    lockin_lower.trajectory[1, :],
                    FILL_BLUE,
                )
            else:
                axes.fill(
                    np.append(lockin_lower.trajectory[0, :], lockin_upper.trajectory[0, :])
                    + xshift,
                    np.append(lockin_lower.trajectory[1, :], lockin_upper.trajectory[1, :]),
                    FILL_BLUE,
                )

        # lock-in domain bounds
        axes.plot(
            lockin_upper.trajectory[0, :] + xshift,
            lockin_upper.trajectory[1, :],
            color=BLUE,
            linewidth=2,
            linestyle=":" if lockin_wrapped or lockin_cycled else "-",
        )
        axes.plot(
            lockin_lower.trajectory[0, :] + xshift,
            lockin_lower.trajectory[1, :],
            color=BLUE,
            linewidth=2,
        )

    # sample locked-in trajectory
    if not lockin_cycled:
        axes.plot(
            estimation.trajectory[0, :],
            estimation.trajectory[1, :],
            color=LINE_BLUE,
            linewidth=1,
        )

    # oscillation estimation
    if not lockin_cycled and estimation.cycle is not None:
        axes.fill(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            "white",
        )
        axes.plot(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            color=BLUE,
            linewidth=2,
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
        "guaranteed lock-in",
        horizontalalignment="center",
        verticalalignment="center",
        color=BLUE,
        family="monospace",
        size=10,
    )
    axes.text(
        0.0,
        (max(slipping_upper.trajectory[1, :]) + ylim) / 2.0,
        "guaranteed slipping",
        horizontalalignment="center",
        verticalalignment="center",
        color=RED,
        family="monospace",
        size=10,
    )
    axes.set_xlim(xlim * np.array([-1, 1]))
    axes.set_ylim(ylim * np.array([-1, 1]))
    axes.set_xticks(
        [-pi, 0, pi], labels=["$-\\pi$", "$0$", "$\\pi$"], usetex=True, size=12
    )
    axes.set_yticks(np.linspace(-ylim, ylim, 7), labels=[])
    axes.grid(color=GRID_COLOR)
