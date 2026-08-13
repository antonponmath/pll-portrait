from math import pi

import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp


class SimulationResult:
    __slots__ = ("solution", "trajectory", "cycle")

    def __init__(self, *, solution, trajectory, cycle):
        self.solution = solution
        self.trajectory = trajectory
        self.cycle = cycle


def crossing_upward(_t, y):
    return y[1]


crossing_upward.direction = +1


def simulate(
    system,
    t_max,
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
        t_span=(0.0, t_max),
        y0=initial_state,
        events=events,
        dense_output=True,
    )

    # TODO adapt to inverse time integration
    t_final = solution.t[-1]
    trajectory_times = np.append(np.arange(0.0, t_final, system.max_step), t_final)
    trajectory = solution.sol(trajectory_times)

    cycle = None
    if escape_bounds is not None and (
        len(solution.t_events[-2]) > 0 or len(solution.t_events[-1]) > 0
    ):
        # escaped
        cycle = None
    elif is_forced:
        # check if there is a cycle at forcing period
        t1 = t_final - system.forcing_period
        t2 = t_final
        y1 = solution.sol(t1)
        y2 = solution.sol(t2)
        if LA.norm(y1 - y2) < cycle_tolerance:
            cycle_times = np.append(np.arange(t1, t2, system.max_step), t2)
            cycle = solution.sol(cycle_times)
    elif cycle_tolerance is not None:
        # check if there is a cycle between crossings
        t = solution.t_events[0]
        y = solution.y_events[0]
        if len(t) > 1 and abs(y[-1, 0] - y[-2, 0]) < cycle_tolerance:
            cycle_times = np.append(np.arange(t[-2], t[-1], system.max_step), t[-1])
            cycle = solution.sol(cycle_times)

    return SimulationResult(solution=solution, trajectory=trajectory, cycle=cycle)
