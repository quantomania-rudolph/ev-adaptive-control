"""Actuator segment: saturation and (later) motor/inverter dynamics."""

from evcontrol.actuator.saturation import Actuator, saturate, saturation_mismatch

__all__ = ["Actuator", "saturate", "saturation_mismatch"]
