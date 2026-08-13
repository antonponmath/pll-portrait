from math import pi, sin, cos, sqrt


class System:
    __slots__ = ("forcing_period", "max_step")

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

    def __init__(self, *, stiffness=1.0, damping=1.0):
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


class VanDerPol(System):
    """Van der Pol oscillator for testing."""

    __slots__ = "damping"

    def __init__(self, *, damping=1.0):
        self.damping = damping
        self.max_step = 2 * pi / 100

    def __call__(self, _time, state):
        x, y = state
        dx = y
        dy = self.damping * (1 - x**2) * y - x
        return [dx, dy]


class ForcedSRFPLL(System):
    """SRF-PLL under unbalanced voltage in normalized time"""

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
        self,
        *,
        kp=1,
        ki=1000,
        frequency=2 * pi * 50,
        positive_sequence_amplitude=200,
        unbalance_factor=0.1,
    ):
        self.kp = kp
        self.ki = ki
        self.frequency = frequency
        self.positive_sequence_amplitude = positive_sequence_amplitude
        self.unbalance_factor = unbalance_factor
        self.__C1 = kp * positive_sequence_amplitude / frequency
        self.__C2 = ki * positive_sequence_amplitude / frequency**2
        self.max_step = 0.1
        self.forcing_period = pi

    def __call__(self, time, state):
        b, z = state
        mu = sqrt(
            1 + 2 * self.unbalance_factor * cos(2 * time) + self.unbalance_factor**2
        )
        F = 1 - (1 - self.unbalance_factor**2) / mu**2
        db = -self.__C1 * mu * sin(b) + z + F
        dz = -self.__C2 * mu * sin(b)
        return [db, dz]


class ComparisonSRFPLL(ForcedSRFPLL):
    """Comparison system for the SRF-PLL under unbalanced voltage."""

    __slots__ = (
        "__C1",
        "__C2",
        "__left_right_sign",
        "__mu_max",
        "__mu_min",
        "__z_bottom",
        "__z_minus",
        "__z_plus",
        "__z_top",
    )

    def __init__(
        self,
        *,
        kp=1,
        ki=1000,
        frequency=2 * pi * 50,
        positive_sequence_amplitude=200,
        unbalance_factor=0.1,
        left_or_right="left",
    ):
        self.kp = kp
        self.ki = ki
        self.frequency = frequency
        self.positive_sequence_amplitude = positive_sequence_amplitude
        self.unbalance_factor = unbalance_factor
        match left_or_right:
            case "left":
                self.__left_right_sign = 1
            case "right":
                self.__left_right_sign = -1
            case _:
                raise ValueError("Comparison system must be left or right.")
        self.__C1 = kp * positive_sequence_amplitude / frequency
        self.__C2 = ki * positive_sequence_amplitude / frequency**2
        self.__z_minus = -2 * unbalance_factor / (1 + unbalance_factor)
        self.__z_plus = 2 * unbalance_factor / (1 - unbalance_factor)
        self.__z_top = 2 * (1 + 2 * unbalance_factor) / (1 - unbalance_factor)
        self.__z_bottom = 2 * (1 - 2 * unbalance_factor) / (1 + unbalance_factor)
        self.__mu_min = 1 - unbalance_factor
        self.__mu_max = 1 + unbalance_factor
        self.max_step = 0.1

    def __call__(self, _time, state):
        b, z = state
        if self.__left_right_sign * sin(b) < 0:
            G = min(
                (z - self.__z_minus) / self.__mu_max,
                (z - self.__z_plus) / self.__mu_min,
            )
        else:
            if z >= self.__z_top:
                G = (z - self.__z_plus) * self.__mu_max
            elif z < self.__z_bottom:
                G = (z - self.__z_minus) * self.__mu_min
            else:
                G = 2 / 3 * sqrt((z + 1) ** 3 / 3 / (1 - self.unbalance_factor**2))
        db = -self.__C1 * sin(b) + G
        dz = -self.__C2 * sin(b)
        return [db, dz]
