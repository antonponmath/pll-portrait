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
        initial_state=[pi - 0.01, comparison_left.z_plus + 0.01],
    )
    exclusion_lower = simulate(
        comparison_right,
        t_max=-t_max,
        initial_state=[-pi + 0.01, comparison_left.z_minus - 0.01],
    )
    lockin_upper = simulate(
        comparison_left,
        t_max=-t_max,
        initial_state=[pi - 0.01, comparison_left.z_minus + 0.01],
    )
    lockin_lower = simulate(
        comparison_left,
        t_max=-t_max,
        initial_state=[-pi + 0.01, comparison_left.z_plus - 0.01],
    )
    estimation = simulate(
        comparison_left,
        t_max=t_max,
        initial_state=[-pi + 0.01, lockin_lower.trajectory[1, -1] / 2.0]
        if lockin_lower.trajectory[1, -1] > 0
        else [-pi + 0.01, lockin_upper.trajectory[1, -1] / 2.0],
    )

    _, ax = plt.subplots()
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
    ax.plot(
        lockin_upper.trajectory[0, :],
        lockin_upper.trajectory[1, :],
        color="blue",
        linewidth=2,
        linestyle=":" if lockin_lower.trajectory[0, -1] < 0 else "-",
    )
    ax.plot(
        lockin_lower.trajectory[0, :],
        lockin_lower.trajectory[1, :],
        color="blue",
        linewidth=2,
    )
    ax.plot(
        estimation.trajectory[0, :],
        estimation.trajectory[1, :],
        color="gray",
        linewidth=1,
    )
    if estimation.cycle is not None:
        ax.plot(
            estimation.cycle[0, :],
            estimation.cycle[1, :],
            color="green",
            linewidth=2,
        )

    # ax.plot(initial_state[0], initial_state[1], marker="o", color="black")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
