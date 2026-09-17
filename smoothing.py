"""
Smoothing filters for cursor movement.
Provides Exponential Moving Average (EMA), One-Euro Filter (1€ filter),
and deadzone filtering to eliminate camera noise, hand tremors, and clicking jitter.
"""
import math
import time
from typing import Optional, Tuple
import numpy as np

from config import SmoothingConfig


class LowPassFilter:
    """Standard 1st-order low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def reset(self):
        self.s = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter1D:
    """
    1D implementation of the 1€ Filter (Casiez et al., CHI 2012).
    Dynamically adjusts filtering based on movement speed:
    - Slow speed: heavy filtering (removes jitter/tremor).
    - Fast speed: minimal filtering (no lag during rapid gestures).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(x, 1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        # Guard against zero or negative dt
        if dt <= 1e-6:
            dt = 1e-3

        # Estimate derivative (velocity)
        prev_x = self.x_filter.s if self.x_filter.s is not None else x
        dx = (x - prev_x) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff, dt))


class CursorSmoother:
    """
    Complete 2D cursor smoothing pipeline combining:
    1. One-Euro Filter or Exponential Moving Average
    2. Deadzone jitter reduction
    """

    def __init__(self, config: Optional[SmoothingConfig] = None):
        self.config = config or SmoothingConfig()
        self.filter_x = OneEuroFilter1D(
            min_cutoff=self.config.min_cutoff,
            beta=self.config.beta,
            d_cutoff=self.config.d_cutoff,
        )
        self.filter_y = OneEuroFilter1D(
            min_cutoff=self.config.min_cutoff,
            beta=self.config.beta,
            d_cutoff=self.config.d_cutoff,
        )
        self.last_smoothed: Optional[Tuple[float, float]] = None

    def reset(self):
        """Reset filter states on hand lost or tracking interruption."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.last_smoothed = None

    def smooth(
        self,
        target_x: float,
        target_y: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Smooth target screen coordinates (x, y).
        """
        # First sample
        if self.last_smoothed is None:
            if self.config.use_one_euro:
                sx = self.filter_x.filter(target_x, timestamp)
                sy = self.filter_y.filter(target_y, timestamp)
            else:
                sx, sy = target_x, target_y
            self.last_smoothed = (sx, sy)
            return sx, sy

        prev_x, prev_y = self.last_smoothed

        # Deadzone check (prevent jitter when resting)
        dist = math.hypot(target_x - prev_x, target_y - prev_y)
        if dist < self.config.deadzone_pixels:
            return prev_x, prev_y

        if self.config.use_one_euro:
            smooth_x = self.filter_x.filter(target_x, timestamp)
            smooth_y = self.filter_y.filter(target_y, timestamp)
        else:
            # Standard Exponential Moving Average
            factor = max(1.0, self.config.smoothing_factor)
            smooth_x = prev_x + (target_x - prev_x) / factor
            smooth_y = prev_y + (target_y - prev_y) / factor

        self.last_smoothed = (smooth_x, smooth_y)
        return smooth_x, smooth_y
