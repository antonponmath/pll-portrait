from math import pi, sqrt
import numpy as np

from matplotlib import pyplot as plt

from pll_portrait.models import SRFPLL, LinearPendulum
from pll_portrait.simulation import simulate


def main():
    # system = LinearPendulum(stiffness=1, damping=1)

    system = SRFPLL(
        kp=1,
        ki=1000,
        frequency=2 * pi * 50,
        positive_sequence_amplitude=200,
        unbalance_factor=0.1,
    )
    result = simulate(system, tmax=100, initial_state=[0, 1])

    # resample
    t_final = 20.0
    t_step = system.max_step
    times = np.arange(0.0, t_final + t_step, t_step)
    states = result.solution.sol(times)

    _, ax = plt.subplots()
    ax.plot(states[0, :], states[1, :], color="black", linewidth=2)

    plt.show()


if __name__ == "__main__":
    main()
