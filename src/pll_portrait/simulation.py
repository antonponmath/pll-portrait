from math import pi

from numpy import linalg as LA
from scipy.integrate import solve_ivp


class SimulationResult:
    __slots__ = ("solution", "cycle_times")

    def __init__(self, *, solution, cycle_times):
        self.solution = solution
        self.cycle_times = cycle_times


def crossing_upward(_t, y):
    return y[1]


crossing_upward.direction = +1


def simulate(
    system,
    t_final,
    initial_state,
    cycle_tolerance=0.01,
    escape_bounds=(-pi, pi),
):
    is_forced = hasattr(system, "forcing_period")

    if cycle_tolerance is None or is_forced:
        events = []
    else:
        events = [crossing_upward]

    if escape_bounds is not None:

        def escape_event_left(_t, y):
            return y[0] - escape_bounds[0]

        def escape_event_right(_t, y):
            return y[0] - escape_bounds[1]

        escape_event_left.terminal = True
        escape_event_right.terminal = True
        events.append(escape_event_left)
        events.append(escape_event_right)

    solution = solve_ivp(
        fun=system,
        t_span=(0.0, t_final + system.max_step),
        y0=initial_state,
        events=events,
        dense_output=True,
    )

    cycle_times = None
    if is_forced:
        t1 = t_final - system.forcing_period
        t2 = t_final
        y1 = solution.sol(t1)
        y2 = solution.sol(t2)
        if LA.norm(y1 - y2) < cycle_tolerance:
            cycle_times = (t1, t2)
    else:
        crossing_times = solution.t_events[0]
        crossing_states = solution.y_events[0]
        if (
            len(crossing_times) > 1
            and abs(crossing_states[-1, 0] - crossing_states[-2, 0]) < cycle_tolerance
        ):
            cycle_times = crossing_times[-2:]

    return SimulationResult(solution=solution, cycle_times=cycle_times)
