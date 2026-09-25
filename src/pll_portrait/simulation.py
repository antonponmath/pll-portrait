from dataclasses import dataclass
from math import copysign, pi

import numpy as np
from scipy.integrate import solve_ivp

from .models import PlanarState, System

type TimeRange = np.ndarray[tuple[int], np.dtype[np.float64]]
type Trajectory = np.ndarray[tuple[int, int], np.dtype[np.float64]]


@dataclass(kw_only=True)
class SimulationResult:
    trajectory: Trajectory
    cycle: Trajectory | None
    escaped: bool


def crossing_upward(_t: float, y: PlanarState) -> float:
    return y[1]
crossing_upward.direction = +1


def simulate(
    system: System,
    t_max: float,
    initial_state: PlanarState,
    cycle_tolerance: float = 0.01,
    escape_bounds: tuple[float, float] | None = (-pi, pi),
) -> SimulationResult:
    events = []
    
    if cycle_tolerance > 0.0 and not system.is_forced():
        events.append(crossing_upward)
    
    def sliding_event(_t: float, y: PlanarState) -> float:
        return system.is_sliding(y)
    sliding_event.terminal = True
    events.append(sliding_event)

    if escape_bounds is not None:
        def escape_event_left(_t: float, y: PlanarState) -> float:
            return y[0] - escape_bounds[0]
        def escape_event_right(_t: float, y: PlanarState) -> float:
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

    def time_range(t0: float, t1: float) -> TimeRange:
        # steps from t0 to t1 with max_step
        # allows t1 < t0 and includes t1
        return np.append(
            t0 + np.arange(0.0, abs(t1 - t0), system.max_step) * copysign(1.0, t1 - t0),
            t1,
        )
    t_final: float = solution.t[-1]
    trajectory_times: TimeRange = time_range(0.0, t_final)
    trajectory: Trajectory = solution.sol(trajectory_times)
    cycle_times: TimeRange = np.array([])
    cycle: Trajectory | None = None
    escaped: bool = False

    if escape_bounds is not None and (
        len(solution.t_events[-2]) > 0 or len(solution.t_events[-1]) > 0
    ):
        escaped = True
    elif system.is_forced():
        # check if there is a cycle at forcing period
        t1: float = t_final - copysign(system.forcing_period, t_final)
        t2: float = t_final
        y1: PlanarState = solution.sol(t1)
        y2: PlanarState = solution.sol(t2)
        if abs(y1[0] - y2[0]) + abs(y1[1] - y2[1]) < cycle_tolerance:
            cycle_times = time_range(t1, t2)
            cycle = solution.sol(cycle_times)
    elif cycle_tolerance > 0.0:
        # check if there is a cycle between crossings
        t: TimeRange = solution.t_events[0]
        y: Trajectory = solution.y_events[0]
        if len(t) > 1 and abs(y[-1, 0] - y[-2, 0]) < cycle_tolerance:
            cycle_times = time_range(t[-2], t[-1])
            cycle = solution.sol(cycle_times)

    return SimulationResult(
        trajectory=trajectory,
        cycle=cycle,
        escaped=escaped,
    )
