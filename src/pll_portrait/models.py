from dataclasses import dataclass
from math import pi, sin, sqrt


class System:
    def __call__(self, time, state):
        raise NotImplementedError(
            f"Right-hand side undefined for {type(self).__name__}."
        )

    def max_step(self):
        raise NotImplementedError(
            f"Maximal integration step undefined for {type(self).__name__}."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearPendulum(System):
    stiffness: float
    damping: float

    def __call__(self, _time, state):
        x, y = state
        dx = y
        dy = -self.stiffness * x - self.damping * y
        return [dx, dy]

    def __discriminant(self):
        return self.damping**2 - 4 * self.stiffness

    def max_step(self):
        D = self.__discriminant()
        if D < 0:
            # oscillatory case: take 100 samples over one period
            frequency = sqrt(-D) / 2
            period = 2 * pi / frequency
            return period / 100
        else:
            # overdamped case: take 100 samples over the 95% settling time
            convergence_rate = (self.damping - sqrt(D)) / 2
            settling_time = 3.0 / convergence_rate  # 3.0 ≈ -ln(0.05)
            return settling_time / 100


@dataclass(frozen=True, slots=True, kw_only=True)
class SRFPLL(System):
    """
    SRF PLL
    """

    kp: float
    ki: float

    def __call__(self, _time, state):
        x, y = state
        dx = y
        dy = -self.ki * sin(x) - self.kp * y
        return [dx, dy]

    def max_step(self):
        return 0.5


@dataclass(frozen=True, slots=True, kw_only=True)
class SRFPLLComparisonUnbalanced(System):
    """
    Comparison system for the SRF PLL under unbalanced voltage.
    """

    kp: float
    ki: float
    unbalance_factor: float
    # def __call__(self, _time: float, state: State) -> State:
    #     x, y = state
    #     ...
    #     return [dx, dy]
    # def max_step(self) -> float:
    #     return 0.1
