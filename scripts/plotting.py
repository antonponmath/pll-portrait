from math import pi

import numpy as np
from matplotlib import pyplot as plt

from pll_portrait.models import ForcedSRFPLL, LinearPendulum, VanDerPol, ComparisonSRFPLL
from pll_portrait.simulation import simulate


def main():
    # system = LinearPendulum()
    # system = VanDerPol()
    # system = ForcedSRFPLL()

    t_max = 20
    initial_state = [0.0, 1.0]
    system_left = ComparisonSRFPLL(left_or_right="left")
    result_left = simulate(system_left, t_max=t_max, initial_state=initial_state)

    system_right = ComparisonSRFPLL(left_or_right="right")
    result_right = simulate(system_right, t_max=t_max, initial_state=initial_state)

    _, ax = plt.subplots()
    ax.plot(
        result_left.trajectory[0, :], result_left.trajectory[1, :], color="blue", linewidth=1
    )
    ax.plot(
        result_right.trajectory[0, :], result_right.trajectory[1, :], color="red", linewidth=1
    )
    # if result.cycle is not None:
    #     ax.plot(result.cycle[0, :], result.cycle[1, :], color="red", linewidth=2)

    ax.plot(initial_state[0], initial_state[1], marker="o", color="black")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
