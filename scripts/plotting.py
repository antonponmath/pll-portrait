from math import pi

import numpy as np
from matplotlib import pyplot as plt

from pll_portrait.models import SRFPLL, LinearPendulum, VanDerPol
from pll_portrait.simulation import simulate


def main():
    # system = LinearPendulum()
    system = VanDerPol()
    # system = SRFPLL()

    t_final = 20.0
    initial_state = [0.0, 1.0]
    result = simulate(system, t_final=t_final, initial_state=initial_state)

    traj_times = np.arange(0.0, t_final + system.max_step, system.max_step)
    traj_states = result.solution.sol(traj_times)

    _, ax = plt.subplots()
    ax.plot(traj_states[0, :], traj_states[1, :], color="black", linewidth=1)

    if result.cycle_times is not None:
        t1, t2 = result.cycle_times
        cycle_times = np.arange(t1, t2 + system.max_step, system.max_step)
        cycle_states = result.solution.sol(cycle_times)
        ax.plot(cycle_states[0, :], cycle_states[1, :], color="red", linewidth=2)

    ax.plot(initial_state[0], initial_state[1], marker="o", color="black")
    plt.show()


if __name__ == "__main__":
    main()
