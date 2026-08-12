from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


class SimulationResult:
    __slots__ = ("converged", "states", "times")

    def __init__(self, *, times, states, converged):
        self.times = times
        self.states = states
        self.converged = converged


def simulate(system, tmax, initial_state):
    solution = solve_ivp(
        fun=system, t_span=(0.0, tmax), y0=initial_state, max_step=system.max_step
    )
    return SimulationResult(times=solution.t, states=solution.y, converged=False)
