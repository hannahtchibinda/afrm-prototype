"""Cruise and holding fuel, (from the Breguet range equation for a jet.).

With fuel flow proportional to thrust, and thrust = weight / (L/D):

    dW/dt = -c * W / (L/D)     so W_start / W_end = exp(k)

where k is equal to the air_distance *c / (TAS *L/D) in cruise, or k = hours * c / (L/D) in a hold.
These functions have no state, and no file access, so they are easy to test and replace.
"""

from __future__ import annotations

import math

from afrm.errors import DataError


def air_distance_nm(ground_nm: float, tas_kt: float, wind_kt: float) -> float:
    """Distance flown through the air. A headwind (negative wind) makes it longer."""
    ground_speed = tas_kt + wind_kt
    if ground_speed <= 0:
        raise DataError(f"Ground speed must be positive (TAS {tas_kt}, wind {wind_kt})")
    return ground_nm * tas_kt / ground_speed


def cruise_exponent(air_nm: float, tas_kt: float, ld: float, tsfc_per_h: float) -> float:
    return air_nm * tsfc_per_h / (tas_kt * ld)


def hold_exponent(hours: float, ld: float, tsfc_per_h: float) -> float:
    return hours * tsfc_per_h / ld


def fuel_from_end_weight(end_weight_kg: float, exponent: float) -> float:
    """Fuel burned to arrive at a known end weight (used when planning backwards)."""
    return end_weight_kg * math.expm1(exponent)


def fuel_from_start_weight(start_weight_kg: float, exponent: float) -> float:
    """Fuel burned from a known start weight (used when monitoring forwards)."""
    return -start_weight_kg * math.expm1(-exponent)


def air_range_nm(
    start_weight_kg: float,
    fuel_kg: float,
    tas_kt: float,
    ld: float,
    tsfc_per_h: float,
    burn_ratio: float = 1.0,
) -> float:
    """Air distance achievable on fuel_kg: the Breguet equation solved for range."""
    if fuel_kg <= 0:
        return 0.0
    if fuel_kg >= start_weight_kg:
        raise DataError("Fuel can't equal or exceed theaircraft weight")
    return (
        tas_kt
        / tsfc_per_h
        * ld
        / burn_ratio
        * math.log(start_weight_kg / (start_weight_kg - fuel_kg))
    )


def ground_from_air_nm(air_nm: float, tas_kt: float, wind_kt: float) -> float:
    return air_nm * max(tas_kt + wind_kt, 0.0) / tas_kt
