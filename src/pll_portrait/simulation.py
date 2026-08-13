from scipy.integrate import solve_ivp
from numpy import linalg as LA


class SimulationResult:
    __slots__ = ("solution", "cycle_times")

    def __init__(self, *, solution, cycle_times):
        self.solution = solution
        self.cycle_times = cycle_times


def simulate(system, t_final, initial_state, cycle_tolerance=0.01):
    solution = solve_ivp(
        fun=system,
        t_span=(0.0, t_final + system.max_step),
        y0=initial_state,
        dense_output=True,
    )

    cycle_times = None
    if hasattr(system, "forcing_period"):
        t1 = t_final - system.forcing_period
        t2 = t_final
        y1 = solution.sol(t1)
        y2 = solution.sol(t2)
        if LA.norm(y1 - y2) < cycle_tolerance:
            cycle_times = (t1, t2)

    return SimulationResult(solution=solution, cycle_times=cycle_times)
