from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt

type PlanarState = tuple[float, float]


@dataclass(kw_only=True)
class System:
    forcing_period: float = 0.0
    max_step: float = field(init=False)

    def is_forced(self) -> bool:
        return self.forcing_period > 0.0

    def is_sliding(self, state: PlanarState) -> float:
        return -1.0

    def __call__(self, time: float, state: PlanarState) -> PlanarState:
        raise NotImplementedError(
            f"Right-hand side undefined for {type(self).__name__}."
        )


@dataclass(kw_only=True)
class LinearPendulum(System):
    """Linear pendulum for testing."""

    stiffness: float = 1.0
    damping: float = 1.0

    def __post_init__(self) -> None:
        D = self.damping**2 - 4 * self.stiffness  # discriminant
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

    def __call__(self, _time: float, state: PlanarState) -> PlanarState:
        x, y = state
        dx = y
        dy = -self.stiffness * x - self.damping * y
        return (dx, dy)


@dataclass(kw_only=True)
class VanDerPol(System):
    """Van der Pol oscillator for testing."""

    damping: float = 1.0

    def __post_init__(self) -> None:
        self.max_step = 2 * pi / 100

    def __call__(self, _time: float, state: PlanarState) -> PlanarState:
        x, y = state
        dx = y
        dy = self.damping * (1 - x**2) * y - x
        return (dx, dy)


@dataclass(kw_only=True)
class MetaSRFPLL(System):
    """SRF-PLL template to handle basic parameters"""

    frequency: float = 2 * pi * 50.0
    positive_sequence_amplitude: float = 200.0
    unbalance_factor: float = 0.1
    kp: float = 1.0
    ki: float = 1000.0
    C1: float = field(init=False)
    C2: float = field(init=False)

    def __post_init__(self) -> None:
        self.C1 = self.kp * self.positive_sequence_amplitude / self.frequency
        self.C2 = self.ki * self.positive_sequence_amplitude / self.frequency**2


@dataclass(kw_only=True)
class ForcedSRFPLL(MetaSRFPLL):
    """SRF-PLL under unbalanced voltage in normalized time"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.forcing_period = pi
        self.max_step = 0.1

    def __call__(self, time: float, state: PlanarState) -> PlanarState:
        b, z = state
        mu = sqrt(
            1 + 2 * self.unbalance_factor * cos(2 * time) + self.unbalance_factor**2
        )
        F = 1 - (1 - self.unbalance_factor**2) / mu**2
        db = -self.C1 * mu * sin(b) + z + F
        dz = -self.C2 * mu * sin(b)
        return (db, dz)


@dataclass(kw_only=True)
class ComparisonSRFPLL(MetaSRFPLL):
    """Comparison system for the SRF-PLL under unbalanced voltage."""

    left_or_right: str = "left"
    z_minus: float = field(init=False)
    z_plus: float = field(init=False)
    z_top: float = field(init=False)
    z_bottom: float = field(init=False)
    mu_min: float = field(init=False)
    mu_max: float = field(init=False)

    def is_sliding(self, state: PlanarState) -> float:
        # is positive when state is in a narrow ellipse around the sliding interval
        z_mid = (self.z_minus + self.z_plus) / 2
        z_dif = self.z_plus - z_mid
        squeeze = 1e6
        a1 = z_dif**2 / squeeze
        a2 = 1 / (squeeze + 1)
        return a1 - state[0] ** 2 - a2 * (state[1] - z_mid) ** 2

    def __post_init__(self) -> None:
        super().__post_init__()

        match self.left_or_right:
            case "left":
                self.__left_right_sign = 1
            case "right":
                self.__left_right_sign = -1
            case _:
                raise ValueError("Comparison system must be left or right.")

        k = self.unbalance_factor
        self.z_minus = -2 * k / (1 + k)
        self.z_plus = 2 * k / (1 - k)
        self.z_top = 2 * (1 + 2 * k) / (1 - k)
        self.z_bottom = 2 * (1 - 2 * k) / (1 + k)
        self.mu_min = 1 - k
        self.mu_max = 1 + k
        self.max_step = 0.1  # TODO relate max_step to the dynamics

    def __call__(self, _time: float, state: PlanarState) -> PlanarState:
        b, z = state
        if self.__left_right_sign * sin(b) < 0:
            G = min(
                (z - self.z_minus) / self.mu_max,
                (z - self.z_plus) / self.mu_min,
            )
        else:
            if z >= self.z_top:
                G = (z - self.z_plus) * self.mu_max
            elif z < self.z_bottom:
                G = (z - self.z_minus) * self.mu_min
            else:
                G = 2 / 3 * sqrt((z + 1) ** 3 / 3 / (1 - self.unbalance_factor**2))
        db = -self.C1 * sin(b) + G
        dz = -self.C2 * sin(b)
        return (db, dz)
