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
    system_left = ComparisonSRFPLL(left_or_right="left", unbalance_factor=0.15)
    system_right = ComparisonSRFPLL(left_or_right="right", unbalance_factor=0.15)

    t_max = 30

    initial_state = [0.0, system_left.z_plus + 0.01]
    result_cycle = simulate(system_left, t_max=t_max, initial_state=initial_state)

    initial_state = [pi - 0.01, system_left.z_minus]
    result_lockin_upper = simulate(
        system_left, t_max=-t_max, initial_state=initial_state
    )

    initial_state = [-pi + 0.01, system_left.z_plus]
    result_lockin_lower = simulate(
        system_left, t_max=-t_max, initial_state=initial_state
    )

    initial_state = [pi - 0.01, system_left.z_plus]
    result_exclusion_upper = simulate(
        system_right, t_max=-t_max, initial_state=initial_state
    )

    initial_state = [-pi + 0.01, system_left.z_minus]
    result_exclusion_lower = simulate(
        system_right, t_max=-t_max, initial_state=initial_state
    )

    _, ax = plt.subplots()
    ax.plot(
        result_lockin_upper.trajectory[0, :],
        result_lockin_upper.trajectory[1, :],
        color="blue",
        linewidth=2,
    )
    ax.plot(
        result_lockin_lower.trajectory[0, :],
        result_lockin_lower.trajectory[1, :],
        color="blue",
        linewidth=2,
    )
    ax.plot(
        result_exclusion_upper.trajectory[0, :],
        result_exclusion_upper.trajectory[1, :],
        color="red",
        linewidth=2,
    )
    ax.plot(
        result_exclusion_lower.trajectory[0, :],
        result_exclusion_lower.trajectory[1, :],
        color="red",
        linewidth=2,
    )
    ax.plot(
        result_cycle.trajectory[0, :],
        result_cycle.trajectory[1, :],
        color="gray",
        linewidth=1,
    )
    if result_cycle.cycle is not None:
        ax.plot(
            result_cycle.cycle[0, :],
            result_cycle.cycle[1, :],
            color="green",
            linewidth=2,
        )

    # ax.plot(initial_state[0], initial_state[1], marker="o", color="black")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
