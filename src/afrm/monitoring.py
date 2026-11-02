"""Monitors fuel during a flight. One FlightMonitor per flight, updated at each reading."""

from __future__ import annotations

import math
from collections import deque

from afrm.models import (
    AircraftProfile,
    Checkpoint,
    CheckpointAssessment,
    FlightPlan,
    FuelPlan,
    FuelStatus,
    InFlightCriteria,
)
from afrm.performance import (
    air_distance_nm,
    air_range_nm,
    cruise_exponent,
    fuel_from_end_weight,
    fuel_from_start_weight,
    ground_from_air_nm,
    hold_exponent,
)

# Positions are rounded, so the last reading can land just past the end of the route.
_POSITION_TOLERANCE_NM = 0.1


def classify(efob_kg: float, plan: FuelPlan, min_fuel_buffer_kg: float) -> FuelStatus:
    """Turns estimated arrival fuel into a status. Less fuel can never give a milder status."""
    if efob_kg < plan.final_reserve_kg:
        return FuelStatus.EMERGENCY
    if efob_kg < plan.final_reserve_kg + min_fuel_buffer_kg:
        return FuelStatus.MINIMUM_FUEL
    if efob_kg < plan.diversion_minimum_kg:
        return FuelStatus.CAUTION
    return FuelStatus.NORMAL


class FlightMonitor:
    def __init__(
        self,
        aircraft: AircraftProfile,
        flight: FlightPlan,
        plan: FuelPlan,
        criteria: InFlightCriteria,
    ) -> None:
        self._aircraft = aircraft
        self._flight = flight
        self._plan = plan
        self._criteria = criteria
        self._zero_fuel_kg = aircraft.oew_kg + flight.payload_kg
        self._noise_band_kg = criteria.noise_band_fraction * aircraft.fuel_capacity_kg
        self._high_limit_kg = criteria.fob_high_tolerance_fraction * aircraft.fuel_capacity_kg
        self._min_fuel_buffer_kg = fuel_from_end_weight(
            self._zero_fuel_kg + plan.final_reserve_kg,
            hold_exponent(
                criteria.min_fuel_buffer_min / 60, aircraft.hold_ld, aircraft.hold_tsfc_per_h
            ),
        )
        #  good readings: (distance, fuel, model burn for the segment ending).
        self._history: deque[tuple[float, float, float]] = deque(
            maxlen=criteria.burn_window_segments + 1
        )
        self._consecutive_invalid = 0
        self._last_status = FuelStatus.NORMAL

    def _in_physical_range(self, sample: Checkpoint) -> bool:
        aircraft = self._aircraft
        return (
            all(math.isfinite(v) for v in (sample.fob_kg, sample.distance_flown_nm, sample.wind_kt))
            and 0 <= sample.fob_kg <= aircraft.fuel_capacity_kg + self._high_limit_kg
            and 0 <= sample.distance_flown_nm <= self._flight.distance_nm + _POSITION_TOLERANCE_NM
            and sample.wind_kt > -aircraft.cruise_tas_kt
        )

    def _is_valid(self, sample: Checkpoint) -> bool:
        if not self._in_physical_range(sample):
            return False
        if self._history:
            last_distance, last_fob, _ = self._history[-1]
            if sample.distance_flown_nm < last_distance:
                return False
            # A reading well above prediction could hide a shortage, so it's rejected. Checking
            # against the last reading isn't enough, as a long segment's burn can mask a spike.
            # A reading below prediction is kept, because it could be a real leak.
            expected = max(last_fob - self._segment_model(sample), 0.0)
            if sample.fob_kg > expected + self._high_limit_kg:
                return False
        return True

    def _segment_model(self, sample: Checkpoint) -> float:
        """Model burn from the last good reading to this one."""
        last_distance, last_fob, _ = self._history[-1]
        return self._model_fuel(
            self._zero_fuel_kg + last_fob,
            sample.distance_flown_nm - last_distance,
            sample.wind_kt,
            self._climb_extra_nm(last_distance, sample.distance_flown_nm),
        )

    def _climb_extra_nm(self, from_nm: float, to_nm: float) -> float:
        """The first miles of a flight count double, so a climb across short segments is shared."""
        return max(min(to_nm, self._aircraft.climb_allowance_nm) - from_nm, 0.0)

    def _model_fuel(
        self, start_weight_kg: float, ground_nm: float, wind_kt: float, extra_air_nm: float = 0.0
    ) -> float:
        aircraft = self._aircraft
        air_nm = air_distance_nm(ground_nm, aircraft.cruise_tas_kt, wind_kt) + extra_air_nm
        return fuel_from_start_weight(
            start_weight_kg,
            cruise_exponent(
                air_nm, aircraft.cruise_tas_kt, aircraft.cruise_ld, aircraft.cruise_tsfc_per_h
            ),
        )

    def _burn_ratio(self) -> float:
        """Actual burn divided by model burn over recent segments.

        Stays at 1.0 unless the extra burn is bigger than gauge noise. Burning less than the
        model earns no credit, so this can only ever add to the fuel we expect to need.
        """
        segments = len(self._history) - 1
        if segments < self._criteria.burn_min_segments:
            return 1.0
        observed = self._history[0][1] - self._history[-1][1]
        modelled = sum(entry[2] for entry in list(self._history)[1:])
        if modelled <= 0 or observed - modelled <= self._noise_band_kg:
            return 1.0
        return observed / modelled

    def update(self, sample: Checkpoint) -> CheckpointAssessment:
        if not self._is_valid(sample):
            return self._invalid(sample)
        self._consecutive_invalid = 0
        aircraft = self._aircraft

        segment_model = self._segment_model(sample) if self._history else 0.0
        self._history.append((sample.distance_flown_nm, sample.fob_kg, segment_model))

        ratio = self._burn_ratio()
        remaining_nm = max(self._flight.distance_nm - sample.distance_flown_nm, 0.0)
        weight_kg = self._zero_fuel_kg + sample.fob_kg
        extra_air = self._climb_extra_nm(sample.distance_flown_nm, self._flight.distance_nm)
        to_destination = self._model_fuel(weight_kg, remaining_nm, sample.wind_kt, extra_air)
        efob = sample.fob_kg - to_destination * ratio

        # How far the aircraft can go before it's down to the final reserve.
        air_nm = air_range_nm(
            weight_kg,
            sample.fob_kg - self._plan.final_reserve_kg,
            aircraft.cruise_tas_kt,
            aircraft.cruise_ld,
            aircraft.cruise_tsfc_per_h,
            ratio,
        )
        range_nm = ground_from_air_nm(
            max(air_nm - extra_air, 0.0), aircraft.cruise_tas_kt, sample.wind_kt
        )
        status = classify(efob, self._plan, self._min_fuel_buffer_kg)
        self._last_status = status
        return CheckpointAssessment(
            seq=sample.seq,
            status=status,
            data_valid=True,
            efob_destination_kg=efob,
            burn_ratio=ratio,
            range_to_final_reserve_nm=range_nm,
            range_margin_nm=range_nm - remaining_nm,
            discrepancy=ratio >= self._criteria.discrepancy_ratio,
        )

    def _invalid(self, sample: Checkpoint) -> CheckpointAssessment:
        """Rejects the reading and keeps the last status, raising it if bad data continues."""
        self._consecutive_invalid += 1
        status = self._last_status
        if self._consecutive_invalid >= self._criteria.invalid_escalation_count:
            status = max(status, FuelStatus.CAUTION)
        return CheckpointAssessment(
            seq=sample.seq,
            status=status,
            data_valid=False,
            efob_destination_kg=None,
            burn_ratio=1.0,
            range_to_final_reserve_nm=None,
            range_margin_nm=None,
            discrepancy=False,
        )
