"""Load aircraft profiles and operational criteria from TOML, validating as they arrive."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

from afrm.errors import DataError
from afrm.models import (
    AircraftProfile,
    Criteria,
    DispatchCriteria,
    InFlightCriteria,
    PlanningCriteria,
)

SUPPORTED_SCHEMA = 1


def _read_toml(path: Path | None, default_name: str) -> dict[str, Any]:
    try:
        if path is None:
            resource = files("afrm.defaults").joinpath(default_name)
            return tomllib.loads(resource.read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataError(f"Cannot read {path or default_name}: {exc}") from exc


def load_aircraft(path: Path | None = None) -> dict[str, AircraftProfile]:
    table = _read_toml(path, "aircraft.toml")
    fleet: dict[str, AircraftProfile] = {}
    for code, values in table.items():
        params = {key: value for key, value in values.items() if key != "description"}
        try:
            fleet[code] = AircraftProfile(type_code=code, **params)
        except TypeError as exc:
            raise DataError(f"Aircraft {code}: {exc}") from exc
    return fleet


def load_criteria(path: Path | None = None) -> Criteria:
    table = _read_toml(path, "criteria.toml")
    if table.get("schema_version") != SUPPORTED_SCHEMA:
        raise DataError(f"Criteria schema_version must be {SUPPORTED_SCHEMA}")
    try:
        criteria = Criteria(
            planning=PlanningCriteria(**table["planning"]),
            dispatch=DispatchCriteria(**table["dispatch"]),
            inflight=InFlightCriteria(**table["inflight"]),
        )
    except (KeyError, TypeError) as exc:
        raise DataError(f"Criteria file is incomplete or has unknown keys: {exc}") from exc
    _check_ranges(criteria)
    return criteria


def _check_ranges(criteria: Criteria) -> None:
    """Reject values that would silently weaken a safety margin or break the algorithm."""
    planning, inflight = criteria.planning, criteria.inflight
    checks = (
        (0 <= planning.contingency_fraction < 1, "contingency_fraction must be in [0, 1)"),
        (planning.contingency_floor_min >= 0, "contingency_floor_min must not be negative"),
        (planning.final_reserve_min > 0, "final_reserve_min must be positive"),
        (criteria.dispatch.marginal_fraction >= 0, "marginal_fraction must not be negative"),
        (inflight.min_fuel_buffer_min >= 0, "min_fuel_buffer_min must not be negative"),
        (inflight.burn_min_segments >= 1, "burn_min_segments must be at least 1"),
        (
            inflight.burn_window_segments >= inflight.burn_min_segments,
            "burn_window_segments must be at least burn_min_segments",
        ),
        (inflight.discrepancy_ratio > 1, "discrepancy_ratio must exceed 1"),
        (inflight.noise_band_fraction >= 0, "noise_band_fraction must not be negative"),
        (inflight.fob_high_tolerance_fraction >= 0, "fob_high_tolerance must not be negative"),
        (inflight.invalid_escalation_count >= 1, "invalid_escalation_count must be at least 1"),
    )

    for passed, message in checks:
        if not passed:
            raise DataError(f"Criteria out of range: {message}")
