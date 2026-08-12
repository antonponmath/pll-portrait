from matplotlib import pyplot as plt

from pll_portrait.models import LinearPendulum
from pll_portrait.simulation import simulate


def main():
    system = LinearPendulum(stiffness=100, damping=10)
    result = simulate(system, tmax=10, initial_state=[0, 1])

    _, ax = plt.subplots()
    ax.plot(result.states[0, :], result.states[1, :], color="black", linewidth=2)

    plt.show()


if __name__ == "__main__":
    main()
