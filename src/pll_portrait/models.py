from math import pi, sin, cos, sqrt


class System:
    __slots__ = ("max_step",)

    def __call__(self, time, state):
        raise NotImplementedError(
            f"Right-hand side undefined for {type(self).__name__}."
        )


class LinearPendulum(System):
    """Linear pendulum for testing."""

    __slots__ = (
        "damping",
        "stiffness",
    )

    def __init__(self, *, stiffness, damping):
        self.stiffness = stiffness
        self.damping = damping
        D = damping**2 - 4 * stiffness  # discriminant
        if D < 0:
            # oscillatory case: take 100 samples over one period
            frequency = sqrt(-D) / 2
            period = 2 * pi / frequency
            self.max_step = period / 100
        else:
            # overdamped case: take 100 samples over the 95% settling time
            convergence_rate = (self.damping - sqrt(D)) / 2
            settling_time = 3.0 / convergence_rate  # 3.0 ≈ -ln(0.05)
            self.max_step = settling_time / 100

    def __call__(self, _time, state):
        x, y = state
        dx = y
        dy = -self.stiffness * x - self.damping * y
        return [dx, dy]


class SRFPLL(System):
    """SRF-PLL under unbalanced voltage"""

    __slots__ = (
        "__C1",
        "__C2",
        "frequency",
        "ki",
        "kp",
        "positive_sequence_amplitude",
        "unbalance_factor",
    )

    def __init__(
        self, *, kp, ki, frequency, positive_sequence_amplitude, unbalance_factor
    ):
        self.kp = kp
        self.ki = ki
        self.frequency = frequency
        self.positive_sequence_amplitude = positive_sequence_amplitude
        self.unbalance_factor = unbalance_factor
        self.__C1 = kp * positive_sequence_amplitude / frequency
        self.__C2 = ki * positive_sequence_amplitude / frequency**2
        self.max_step = 0.1

    def __call__(self, time, state):
        x, y = state
        mu = sqrt(1 + 2 * self.unbalance_factor * cos(time) + self.unbalance_factor**2)
        F = 1 - (1 - self.unbalance_factor**2) / mu**2
        dx = -self.__C1 * mu * sin(x) + y + F
        dy = -self.__C2 * mu * sin(x)
        return [dx, dy]


class ComparisonSRFPLL(SRFPLL):
    """Comparison system for the SRF-PLL under unbalanced voltage."""
