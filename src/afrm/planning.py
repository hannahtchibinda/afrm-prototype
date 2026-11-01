"""Works out how much fuel a flight needs and whether it can be dispatched."""

from __future__ import annotations

import math

from afrm.errors import ConvergenceError, DataError
from afrm.models import (
    AircraftProfile,
    Criteria,
    DispatchAssessment,
    DispatchStatus,
    FlightPlan,
    FuelPlan,
    PlanningCriteria,
)
from afrm.performance import (
    air_distance_nm,
    air_range_nm,
    cruise_exponent,
    fuel_from_end_weight,
    ground_from_air_nm,
    hold_exponent,
)

# The loop stops once trip fuel changes by less than this.
_TOLERANCE_KG = 0.01
# Stops the loop running forever if the numbers never settle.
_MAX_ITERATIONS = 50


def trip_exponent(aircraft: AircraftProfile, flight: FlightPlan) -> float:
    """Gets k for the whole trip, including wind and the climb allowance."""
    air_nm = air_distance_nm(flight.distance_nm, aircraft.cruise_tas_kt, flight.forecast_wind_kt)
    return cruise_exponent(
        air_nm + aircraft.climb_allowance_nm,
        aircraft.cruise_tas_kt,
        aircraft.cruise_ld,
        aircraft.cruise_tsfc_per_h,
    )


def compute_fuel_plan(
    aircraft: AircraftProfile, flight: FlightPlan, criteria: PlanningCriteria
) -> FuelPlan:
    """Works out the fuel needed, starting from the end of the flight.

    Zero-fuel weight is the only weight known at the start, so we work backwards from it.
    Trip fuel and contingency depend on each other, so the loop repeats until they settle.
    """
    zero_fuel_kg = aircraft.oew_kg + flight.payload_kg

    # Final reserve is burned last, so it ends at zero-fuel weight.
    final_reserve = fuel_from_end_weight(
        zero_fuel_kg,
        hold_exponent(criteria.final_reserve_min / 60, aircraft.hold_ld, aircraft.hold_tsfc_per_h),
    )

    # The alternate leg ends with the final reserve still on board.
    alternate = fuel_from_end_weight(
        zero_fuel_kg + final_reserve,
        cruise_exponent(
            flight.alternate_distance_nm,  # Assumes no wind, as the plan has no diversion forecast.
            aircraft.cruise_tas_kt,
            aircraft.cruise_ld,
            aircraft.cruise_tsfc_per_h,
        ),
    )

    arrival_kg = zero_fuel_kg + final_reserve + alternate

    # Contingency can't go below this, which matters on short flights.
    contingency_floor = fuel_from_end_weight(
        arrival_kg,
        hold_exponent(
            criteria.contingency_floor_min / 60, aircraft.hold_ld, aircraft.hold_tsfc_per_h
        ),
    )
    exponent = trip_exponent(aircraft, flight)

    # Start from zero and repeat until trip fuel stops changing.
    trip = 0.0
    for iteration in range(1, _MAX_ITERATIONS + 1):
        contingency = max(criteria.contingency_fraction * trip, contingency_floor)
        next_trip = fuel_from_end_weight(arrival_kg + contingency, exponent)
        settled = abs(next_trip - trip) < _TOLERANCE_KG
        trip = next_trip
        if settled:
            return FuelPlan(
                taxi_kg=aircraft.taxi_fuel_kg,
                trip_kg=trip,
                contingency_kg=max(criteria.contingency_fraction * trip, contingency_floor),
                alternate_kg=alternate,
                final_reserve_kg=final_reserve,
                iterations=iteration,
            )
    raise ConvergenceError(f"{flight.flight_id}: fuel plan did not settle")


def assess_dispatch(
    aircraft: AircraftProfile, flight: FlightPlan, criteria: Criteria
) -> DispatchAssessment:
    """Checks the planned fuel against the fuel needed and the aircraft's weight limits."""
    if aircraft.type_code != flight.aircraft_type:
        raise DataError(f"{flight.flight_id}: aircraft profile mismatch")
    plan = compute_fuel_plan(aircraft, flight, criteria.planning)
    zero_fuel_kg = aircraft.oew_kg + flight.payload_kg
    takeoff_kg = zero_fuel_kg + flight.planned_fob_kg - plan.taxi_kg

    # Extra fuel makes the aircraft heavier, so some of it gets burned just carrying it.
    actual_trip = -takeoff_kg * math.expm1(-trip_exponent(aircraft, flight))
    penalty = actual_trip - plan.trip_kg
    margin = flight.planned_fob_kg - plan.required_kg - penalty
    landing_kg = takeoff_kg - actual_trip

    # Range uses only the fuel left after taxi and reserves are set aside.
    reserves = plan.contingency_kg + plan.alternate_kg + plan.final_reserve_kg
    trip_available = flight.planned_fob_kg - plan.taxi_kg - reserves
    air_nm = air_range_nm(
        takeoff_kg,
        trip_available,
        aircraft.cruise_tas_kt,
        aircraft.cruise_ld,
        aircraft.cruise_tsfc_per_h,
    )
    protected_range = ground_from_air_nm(
        max(air_nm - aircraft.climb_allowance_nm, 0.0),
        aircraft.cruise_tas_kt,
        flight.forecast_wind_kt,
    )

    # Each pair is a failed check and the reason to report for it.
    limits = (
        (flight.planned_fob_kg > aircraft.fuel_capacity_kg, "fuel exceeds tank capacity"),
        (zero_fuel_kg > aircraft.mzfw_kg, "zero-fuel weight exceeds MZFW"),
        (takeoff_kg > aircraft.mtow_kg, "take-off weight exceeds MTOW"),
        (landing_kg > aircraft.mlw_kg, "landing weight exceeds MLW"),
        (flight.planned_fob_kg < plan.required_kg, "fuel below minimum required"),
    )
    reasons = tuple(text for failed, text in limits if failed)

    if reasons:
        status = DispatchStatus.NO_GO
    elif margin < criteria.dispatch.marginal_fraction * plan.required_kg:
        status = DispatchStatus.MARGINAL
        reasons = ("usable margin below dispatch threshold",)
    else:
        status = DispatchStatus.GO

    return DispatchAssessment(
        status=status,
        plan=plan,
        usable_margin_kg=margin,
        tankering_penalty_kg=penalty,
        takeoff_weight_kg=takeoff_kg,
        landing_weight_kg=landing_kg,
        protected_range_nm=protected_range,
        range_margin_nm=protected_range - flight.distance_nm,
        reasons=reasons,
    )
