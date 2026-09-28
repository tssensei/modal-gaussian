"""Small-displacement modal interaction; no cameras, artifacts or viewer state."""
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True)
class InteractionConfig:
    damping: float = 0.05
    strength: float = 1.0
    drag_radius_fraction: float = 0.05
    foreground_alpha_minimum: float = 0.1

    def __post_init__(self):
        if not (np.isfinite(list(vars(self).values())).all()
                and 0 <= self.damping <= 1 and 0 <= self.strength <= 5
                and 0 < self.drag_radius_fraction <= 1
                and 0 < self.foreground_alpha_minimum <= 1):
            raise ValueError("Invalid interaction settings")


def mode_factors(modes: Iterable[np.ndarray]) -> np.ndarray:
    """Return exp(-i phase)/RMS, streaming one [N,3] displacement field at a time.

    RMS counts supported nodes, not scalar coordinates. Degenerate complex PCA
    uses the first largest component. Only the small factors are retained.
    """
    factors = []
    for mode in modes:
        value = np.asarray(mode, dtype=np.complex128)
        if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
            raise ValueError("Interaction field must be finite [N,3]")
        power = np.sum(np.abs(value) ** 2, axis=1)
        count = np.count_nonzero(power)
        scale = np.sqrt(power.sum() / count) if count else 0.0
        if scale <= 1e-12:
            factors.append(0j)
            continue
        orientation = np.sum(value * value)
        phase = (0.5 * np.angle(orientation) if abs(orientation) > 1e-6 * power.sum()
                 else np.angle(value.flat[int(np.argmax(np.abs(value)))]))
        factors.append(np.exp(-1j * phase) / scale)
    return np.asarray(factors, dtype=np.complex128)


class ModalInteraction:
    """Caller supplies monotonic timestamps and serializes access to this state."""

    def __init__(self, frequencies, factors, *, now=0.0, config=InteractionConfig()):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        self.factors = np.array(factors, dtype=np.complex128, copy=True)
        if (frequencies.ndim != 1 or self.factors.shape != frequencies.shape
                or not np.isfinite(frequencies).all() or not np.isfinite(self.factors).all()):
            raise ValueError("Finite matching frequency/factor vectors required")
        self.valid = (frequencies > 0) & (self.factors != 0)
        self.factors[~self.valid] = 0
        self.omega = np.where(self.valid, 2 * np.pi * frequencies, 1.0)
        self.damping = config.damping
        self.rho = np.zeros_like(frequencies)
        self.velocity = np.zeros_like(frequencies)
        self.paused = False
        self.drag_start = None
        self.local_field = None
        self.time = self._time(now)

    @staticmethod
    def _time(value):
        if not np.isfinite(value):
            raise ValueError("Simulation time must be finite")
        return float(value)

    @property
    def z(self):
        return self.rho - 1j * self.velocity / self.omega

    def advance(self, now):
        now = self._time(now)
        if now < self.time:
            raise ValueError("Simulation clock moved backwards")
        dt = now - self.time
        self.time = now
        if self.paused or self.drag_start is not None or dt == 0:
            return
        gamma = self.damping * self.omega
        b = self.omega * np.sqrt(1 - self.damping**2) * dt
        s, c, e = dt * np.sinc(b / np.pi), np.cos(b), np.exp(-gamma * dt)
        rho, velocity = self.rho, self.velocity
        self.rho = e * ((c + gamma * s) * rho + s * velocity)
        self.velocity = e * (-self.omega**2 * s * rho + (c - gamma * s) * velocity)

    def coordinates(self, now):
        self.advance(now)
        return np.asarray(self.factors * self.z, dtype=np.complex64)

    def set_state(self, z, now):
        z = np.asarray(z, dtype=np.complex128)
        if z.shape != self.rho.shape or not np.isfinite(z).all():
            raise ValueError("Finite matching modal state required")
        self.time = self._time(now)
        self.rho = np.where(self.valid, z.real, 0).copy()
        self.velocity = np.where(self.valid, -self.omega * z.imag, 0).copy()

    def begin_drag(self, field_at_point, displayed_z, now, *, normalize_support=True):
        value = np.asarray(field_at_point, dtype=np.complex128)
        if value.shape != (len(self.rho), 3) or not np.isfinite(value).all():
            raise ValueError("Finite point field [K,3] required")
        local = self.factors[:, None] * value
        support = float(np.sum(np.abs(local)**2))
        if support <= 1e-12:
            return False
        self.set_state(displayed_z, now)
        self.drag_start = self.z.copy()
        self.local_field = local / support if normalize_support else local
        return True

    def drag(self, displacement, *, strength, maximum):
        d = np.array(displacement, dtype=np.float64, copy=True)
        if (d.shape != (3,) or not np.isfinite(d).all() or not np.isfinite(strength)
                or not 0 <= strength <= 5 or not np.isfinite(maximum) or maximum <= 0):
            raise ValueError("Invalid drag displacement, strength or limit")
        if self.drag_start is None:
            return False
        length = float(np.linalg.norm(d))
        limited = length > maximum
        if limited:
            d *= maximum / length
        self.set_state(self.drag_start + strength * np.conj(self.local_field @ d), self.time)
        return limited

    def release(self, now):
        self.advance(now)
        self.drag_start = self.local_field = None

    def set_damping(self, value, now):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Damping ratio must be in [0,1]")
        self.advance(now)
        self.damping = float(value)

    def set_paused(self, paused, now):
        self.advance(now)
        self.paused = bool(paused)

    def reset(self, now):
        self.drag_start = self.local_field = None
        self.set_state(np.zeros_like(self.rho), now)
