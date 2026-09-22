from math import pi

import numpy as np
from matplotlib import pyplot as plt

from pll_portrait.models import (
    ComparisonSRFPLL,
    ForcedSRFPLL,
    LinearPendulum,
    VanDerPol,
)
from pll_portrait.simulation import simulate

EPS = 0.001

def main():
    # system = LinearPendulum()
    # system = VanDerPol()
    # system = ForcedSRFPLL()
    comparison_left = ComparisonSRFPLL(left_or_right="left", unbalance_factor=0.15)
    comparison_right = ComparisonSRFPLL(left_or_right="right", unbalance_factor=0.15)

    t_max = 50

    exclusion_upper = simulate(
        comparison_right,
        t_max=-t_max,
        initial_state=[pi - EPS, comparison_left.z_plus + EPS],
    )
    exclusion_lower = simulate(
        comparison_right,
        t_max=-t_max,
        initial_state=[-pi + EPS, comparison_left.z_minus - EPS],
    )
    lockin_upper = simulate(
        comparison_left,
        t_max=-t_max,
        initial_state=[pi - EPS, comparison_left.z_minus + EPS],
    )
    lockin_lower = simulate(
        comparison_left,
        t_max=-t_max,
        initial_state=[-pi + EPS, comparison_left.z_plus - EPS],
    )
    estimation = simulate(
        comparison_left,
        t_max=t_max,
        initial_state=[-pi + EPS, lockin_lower.trajectory[1, -1] / 2.0]
        if lockin_lower.trajectory[1, -1] > 0
        else [-pi + EPS, lockin_upper.trajectory[1, -1] / 2.0],
    )

    _, ax = plt.subplots()
    xlim = 1.1 * pi
    ylim = -1.1 * min(exclusion_lower.trajectory[1, :])

    # exclusion region fill
    ax.fill(
        np.append(exclusion_lower.trajectory[0, :], [pi, -pi]),
        np.append(exclusion_lower.trajectory[1, :], [-ylim, -ylim]),
        "#fee",
    )
    ax.fill(
        np.append(exclusion_upper.trajectory[0, :], [-pi, pi]),
        np.append(exclusion_upper.trajectory[1, :], [ylim, ylim]),
        "#fee",
    )

    # exclusion region bounds
    ax.plot(
        exclusion_upper.trajectory[0, :],
        exclusion_upper.trajectory[1, :],
        color="red",
        linewidth=2,
    )
    ax.plot(
        exclusion_lower.trajectory[0, :],
        exclusion_lower.trajectory[1, :],
        color="red",
        linewidth=2,
    )

    # lock-in domain fill
    if lockin_lower.trajectory[0, -1] > 0:
        ax.fill(
            np.append(lockin_lower.trajectory[0, :], lockin_upper.trajectory[0, :]),
            np.append(lockin_lower.trajectory[1, :], lockin_upper.trajectory[1, :]),
            "#eef",
        )
    else:
        ax.fill(
            lockin_lower.trajectory[0, :],
            lockin_lower.trajectory[1, :],
            "#eef",
        )

    # lock-in domain bounds
    ax.plot(
        lockin_upper.trajectory[0, :],
        lockin_upper.trajectory[1, :],
        color="blue",
        linewidth=2,
        linestyle="-" if lockin_lower.trajectory[0, -1] > 0 else ":",
    )
    ax.plot(
        lockin_lower.trajectory[0, :],
        lockin_lower.trajectory[1, :],
        color="blue",
        linewidth=2,
    )

    # oscillation fill
    if estimation.cycle is not None:
        ax.fill(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            "#efe",
        )

    # sample locked-in trajectory
    ax.plot(
        estimation.trajectory[0, :],
        estimation.trajectory[1, :],
        color="#ccf",
        linewidth=1,
    )

    # oscillation bound
    if estimation.cycle is not None:
        ax.plot(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            color="green",
            linewidth=2,
        )

    # stationary points
    ax.plot(
        [0.0, 0.0],
        [comparison_left.z_minus, comparison_left.z_plus],
        linestyle="",
        marker="o",
        color="black",
    )
    ax.plot(
        [-pi, -pi, pi, pi],
        [
            comparison_left.z_minus,
            comparison_left.z_plus,
            comparison_left.z_minus,
            comparison_left.z_plus,
        ],
        linestyle="",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
    )

    ax.set(
        xlim=xlim * np.array([-1, 1]),
        ylim=ylim * np.array([-1, 1]),
    )
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
