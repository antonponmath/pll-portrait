from scipy.integrate import solve_ivp


class SimulationResult:
    __slots__ = ("converged", "solution")

    def __init__(self, *, solution, converged):
        self.solution = solution
        self.converged = converged


def simulate(system, tmax, initial_state):
    solution = solve_ivp(
        fun=system, t_span=(0.0, tmax), y0=initial_state, dense_output=True
    )
    return SimulationResult(solution=solution, converged=False)
