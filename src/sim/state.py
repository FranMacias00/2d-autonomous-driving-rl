"""Simulation state structures."""

from dataclasses import dataclass


@dataclass
class CarPose:
    """Pose of the car in 2D space.

    Attributes:
        x: X position of the car center.
        y: Y position of the car center.
        angle: Orientation in radians (counter-clockwise).
    """

    x: float
    y: float
    angle: float
