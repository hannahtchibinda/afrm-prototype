"""Domain types. All of them are immutable; slots keeping per-record memory low."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from afrm.errors import DataError


def _require_positive(owner: str, **values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise DataError(f"{owner}: {name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class AircraftProfile:
    type_code: str
    oew_kg: float
    mzfw_kg: float
    mtow_kg: float
    mlw_kg: float
    fuel_capacity_kg: float
    cruise_tas_kt: float
    cruise_ld: float
    cruise_tsfc_per_h: float
    hold_ld: float
    hold_tsfc_per_h: float
    taxi_fuel_kg: float
    climb_allowance_nm: float

    def __post_init__(self) -> None:
        _require_positive(
            self.type_code,
            oew_kg=self.oew_kg,
            fuel_capacity_kg=self.fuel_capacity_kg,
            cruise_tas_kt=self.cruise_tas_kt,
            cruise_ld=self.cruise_ld,
            cruise_tsfc_per_h=self.cruise_tsfc_per_h,
            hold_ld=self.hold_ld,
            hold_tsfc_per_h=self.hold_tsfc_per_h,
            taxi_fuel_kg=self.taxi_fuel_kg,
            climb_allowance_nm=self.climb_allowance_nm,
        )
        if not self.oew_kg < self.mzfw_kg < self.mlw_kg <= self.mtow_kg:
            raise DataError(f"{self.type_code}: require OEW < MZFW < MLW <= MTOW")


@dataclass(frozen=True, slots=True)
class FlightPlan:
    flight_id: str
    aircraft_type: str
    origin: str
    destination: str
    alternate: str
    distance_nm: float
    alternate_distance_nm: float
    payload_kg: float
    planned_fob_kg: float
    forecast_wind_kt: float  # Positive is a tailwind.
    scenario: str = "UNSPECIFIED"  # For testing and analysis; never read by the algorithm.

    def __post_init__(self) -> None:
        _require_positive(
            self.flight_id,
            distance_nm=self.distance_nm,
            alternate_distance_nm=self.alternate_distance_nm,
            planned_fob_kg=self.planned_fob_kg,
        )
        if not (math.isfinite(self.payload_kg) and self.payload_kg >= 0):
            raise DataError(f"{self.flight_id}: payload_kg must not be negative")
        if not math.isfinite(self.forecast_wind_kt):
            raise DataError(f"{self.flight_id}: forecast_wind_kt must be finite")


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One telemetry sample. Unvalidated on purpose: the monitor handles bad readings."""

    flight_id: str
    seq: int
    elapsed_h: float
    distance_flown_nm: float
    fob_kg: float
    wind_kt: float


@dataclass(frozen=True, slots=True)
class PlanningCriteria:
    contingency_fraction: float
    contingency_floor_min: float
    final_reserve_min: float


@dataclass(frozen=True, slots=True)
class DispatchCriteria:
    marginal_fraction: float


@dataclass(frozen=True, slots=True)
class InFlightCriteria:
    min_fuel_buffer_min: float
    burn_window_segments: int
    burn_min_segments: int
    discrepancy_ratio: float
    noise_band_fraction: float
    fob_high_tolerance_fraction: float
    invalid_escalation_count: int


@dataclass(frozen=True, slots=True)
class Criteria:
    planning: PlanningCriteria
    dispatch: DispatchCriteria
    inflight: InFlightCriteria


@dataclass(frozen=True, slots=True)
class FuelPlan:
    """Minimum required fuel by component, (kgs)."""

    taxi_kg: float
    trip_kg: float
    contingency_kg: float
    alternate_kg: float
    final_reserve_kg: float
    iterations: int

    @property
    def required_kg(self) -> float:
        return (
            self.taxi_kg
            + self.trip_kg
            + self.contingency_kg
            + self.alternate_kg
            + self.final_reserve_kg
        )

    @property
    def diversion_minimum_kg(self) -> float:
        """Fuel needed on arrival to reach the alternate and still hold the final reserve."""
        return self.alternate_kg + self.final_reserve_kg


class DispatchStatus(StrEnum):
    GO = "GO"
    MARGINAL = "MARGINAL"
    NO_GO = "NO_GO"


class FuelStatus(IntEnum):
    """In-flight status. Ordered by severity, so max() gives the worst of a flight."""

    NORMAL = 0
    CAUTION = 1
    MINIMUM_FUEL = 2
    EMERGENCY = 3


@dataclass(frozen=True, slots=True)
class DispatchAssessment:
    status: DispatchStatus
    plan: FuelPlan
    usable_margin_kg: float
    tankering_penalty_kg: float
    takeoff_weight_kg: float
    landing_weight_kg: float
    protected_range_nm: float
    range_margin_nm: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointAssessment:
    seq: int
    status: FuelStatus
    data_valid: bool
    efob_destination_kg: float | None  # Estimated arrival fuel; None for a rejected reading.
    burn_ratio: float
    range_to_final_reserve_nm: float | None
    range_margin_nm: float | None
    discrepancy: bool


@dataclass(frozen=True, slots=True)
class FlightAssessment:
    flight: FlightPlan
    dispatch: DispatchAssessment
    checkpoints: tuple[CheckpointAssessment, ...]

    @property
    def worst_status(self) -> FuelStatus:
        return max((c.status for c in self.checkpoints), default=FuelStatus.NORMAL)
