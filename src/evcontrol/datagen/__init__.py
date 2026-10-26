"""Data generation segment: references, disturbances, and (later) dataset loading."""

from evcontrol.datagen.generator import (
    generate_disturbance,
    generate_reference,
    time_vector,
)

__all__ = ["generate_disturbance", "generate_reference", "time_vector"]
