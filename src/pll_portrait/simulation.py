from dataclasses import dataclass

from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationResult:
    times: ArrayLike
    states: ArrayLike
    converged: bool


def simulate(system, tmax, initial_state):
    solution = solve_ivp(
        fun=system, t_span=(0.0, tmax), y0=initial_state, max_step=system.max_step()
    )
    return SimulationResult(times=solution.t, states=solution.y, converged=False)
