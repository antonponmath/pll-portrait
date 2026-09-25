from math import pi
import numpy as np
from matplotlib.axes import Axes

from .models import System
from .simulation import simulate

EPS = 0.001

COL_SLIP_BORDER = "#b70404"
COL_SLIP_FILL = "#f1cdcd"
COL_ESTIM_BORDER = "#069a8e"
COL_ESTIM_FILL = "#b4e1dd"
COL_LOCKIN_BORDER = "#2155cd"
COL_LOCKIN_TRAJ = "#a6bbeb"
COL_LOCKIN_FILL = "#d3ddf5"
COL_GRID = "#ccc"


def draw_comparison_portrait(
    system_left: System, system_right: System, t_max: float, axes: Axes
) -> None:
    slipping_upper = simulate(
        system_right,
        t_max=-t_max,
        initial_state=(pi - EPS, system_left.z_plus + EPS),
    )
    slipping_lower = simulate(
        system_right,
        t_max=-t_max,
        initial_state=(-pi + EPS, system_left.z_minus - EPS),
    )
    lockin_upper = simulate(
        system_left,
        t_max=-t_max,
        initial_state=(pi - EPS, system_left.z_minus + EPS),
    )
    lockin_lower = simulate(
        system_left,
        t_max=-t_max,
        initial_state=(-pi + EPS, system_left.z_plus - EPS),
    )

    lockin_wrapped: bool = lockin_lower.escaped and lockin_lower.trajectory[0, -1] < 0
    lockin_cycled: bool = not lockin_lower.escaped

    if not lockin_cycled:
        estimation = simulate(
            system_left,
            t_max=t_max,
            initial_state=(
                -pi + EPS,
                (lockin_lower.trajectory[1, -1] + system_left.z_plus) / 2.0,
            )
            if lockin_wrapped
            else (
                -pi + EPS,
                (lockin_upper.trajectory[1, -1] + system_left.z_plus) / 2.0,
            ),
        )

    xlim: float = 1.15 * pi
    ylim: float = -1.2 * min(slipping_lower.trajectory[1, :])

    for xshift in [-2 * pi, 0, 2 * pi]:
        # slipping region fill
        axes.fill(
            np.append(slipping_lower.trajectory[0, :], [pi, -pi]) + xshift,
            np.append(slipping_lower.trajectory[1, :], [-ylim, -ylim]),
            COL_SLIP_FILL,
        )
        axes.fill(
            np.append(slipping_upper.trajectory[0, :], [-pi, pi]) + xshift,
            np.append(slipping_upper.trajectory[1, :], [ylim, ylim]),
            COL_SLIP_FILL,
        )

        # slipping region bounds
        axes.plot(
            slipping_upper.trajectory[0, :] + xshift,
            slipping_upper.trajectory[1, :],
            color=COL_SLIP_BORDER,
            linewidth=2,
        )
        axes.plot(
            slipping_lower.trajectory[0, :] + xshift,
            slipping_lower.trajectory[1, :],
            color=COL_SLIP_BORDER,
            linewidth=2,
        )

        # lock-in domain fill
        if not lockin_cycled:
            if lockin_wrapped:
                axes.fill(
                    lockin_lower.trajectory[0, :] + xshift,
                    lockin_lower.trajectory[1, :],
                    COL_LOCKIN_FILL,
                )
            else:
                axes.fill(
                    np.append(
                        lockin_lower.trajectory[0, :], lockin_upper.trajectory[0, :]
                    )
                    + xshift,
                    np.append(
                        lockin_lower.trajectory[1, :], lockin_upper.trajectory[1, :]
                    ),
                    COL_LOCKIN_FILL,
                )

        # lock-in domain bounds
        axes.plot(
            lockin_upper.trajectory[0, :] + xshift,
            lockin_upper.trajectory[1, :],
            color=COL_LOCKIN_BORDER,
            linewidth=2,
            linestyle=":" if lockin_wrapped or lockin_cycled else "-",
        )
        axes.plot(
            lockin_lower.trajectory[0, :] + xshift,
            lockin_lower.trajectory[1, :],
            color=COL_LOCKIN_BORDER,
            linewidth=2,
        )

    # sample locked-in trajectory
    if not lockin_cycled:
        axes.plot(
            estimation.trajectory[0, :],
            estimation.trajectory[1, :],
            color=COL_LOCKIN_TRAJ,
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
            color=COL_LOCKIN_BORDER,
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

    if lockin_cycled:
        axes.text(
            0.0,
            0.0,
            "phase lock\nmay be lost",
            horizontalalignment="center",
            verticalalignment="center",
            color=COL_LOCKIN_BORDER,
            family="monospace",
            size=10,
        )
    elif estimation.cycle is None:
        axes.text(
            0.0,
            max(lockin_upper.trajectory[1, :]) / 2.0,
            "guaranteed\nlock-in",
            horizontalalignment="center",
            # verticalalignment="center",
            color=COL_LOCKIN_BORDER,
            family="monospace",
            size=10,
        )
    else:
        axes.text(
            0.0,
            max(
                max(lockin_upper.trajectory[1, :]) / 2.0,
                max(estimation.cycle[1, :]) * 1.05,
            ),
            "guaranteed\nlock-in",
            horizontalalignment="center",
            # verticalalignment="center",
            color=COL_LOCKIN_BORDER,
            family="monospace",
            size=10,
        )

    axes.text(
        0.0,
        (max(slipping_upper.trajectory[1, :]) + ylim) / 2.0,
        "guaranteed slipping",
        horizontalalignment="center",
        verticalalignment="center",
        color=COL_SLIP_BORDER,
        family="monospace",
        size=10,
    )
    axes.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
    axes.set_xticks(
        [-pi, 0, pi], labels=["$-\\pi$", "$0$", "$\\pi$"], usetex=True, size=12
    )
    axes.set_yticks(np.linspace(-ylim, ylim, 7), labels=[])
    axes.grid(color=COL_GRID)
