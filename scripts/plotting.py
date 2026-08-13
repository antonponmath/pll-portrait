from math import pi, sqrt

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
    result = simulate(system, tmax=30, initial_state=[0, 1])

    _, ax = plt.subplots()
    ax.plot(result.states[0, :], result.states[1, :], color="black", linewidth=2)

    plt.show()


if __name__ == "__main__":
    main()
