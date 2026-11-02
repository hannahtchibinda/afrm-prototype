"""Generates simulated flights and fuel readings from a seed.

The simulator has its own fuel behaviour, separate from the algorithm's Breguet model: it steps
through climb, cruise and descent, and aerodynamic efficiency changes with weight. This stops the
algorithm being tested against its own equation. The algorithm only sees the flight plan and the
noisy gauge readings, as it would with real data.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import replace

from afrm.models import AircraftProfile, Checkpoint, Criteria, FlightPlan
from afrm.planning import compute_fuel_plan, trip_exponent

# Share of generated flights by scenario. (These are weights, so they needn't add up to 100.)
SCENARIOS: dict[str, int] = {
    "NOMINAL": 46,
    "HEADWIND": 12,
    "MARGINAL_DISPATCH": 6,
    "LOW_FUEL_DISPATCH": 6,
    "OVERWEIGHT": 4,
    "FUEL_LEAK": 10,
    "SEVERE_LEAK": 6,
    "GAUGE_HIGH": 6,
    "GAUGE_LOW": 4,
}

_ROUTE_BOUNDS_NM = {"GRJ": (250, 1300), "GNB": (400, 2600), "GWB": (2000, 7000)}
_GAUGE_SIGMA_FRACTION = 0.001  # Gauge noise, as a fraction of tank capacity.
_MIN_DISTANCE_NM = 200.0
_STEP_NM = 10.0  # Fuel burn is stepped along the route in lengths of about this size.
_CLIMB_FLOW_FACTOR = 1.9  # Climb uses more fuel per mile than cruise...
_DESCENT_FLOW_FACTOR = 0.4  # ...and descent uses much less.
_LD_WEIGHT_PENALTY = 0.5  # How quickly efficiency falls away from the best weight.
_BEST_WEIGHT_FRACTION = 0.80  # Best efficiency, as a fraction of maximum take-off weight.


def _airport(rng: random.Random) -> str:
    """Made-up four-letter code. The ZZ prefix avoids implying a real airfield."""
    return "ZZ" + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))


def _with_margin(
    aircraft: AircraftProfile, flight: FlightPlan, criteria: Criteria, margin_fraction: float
) -> tuple[FlightPlan, float]:
    """Returns the flight with enough fuel to leave the given usable margin after carriage."""
    plan = compute_fuel_plan(aircraft, flight, criteria.planning)
    margin = margin_fraction * plan.required_kg
    fob = plan.required_kg + margin * math.exp(trip_exponent(aircraft, flight))
    return replace(flight, planned_fob_kg=fob), plan.required_kg


def _is_dispatchable(aircraft: AircraftProfile, flight: FlightPlan) -> bool:
    takeoff = aircraft.oew_kg + flight.payload_kg + flight.planned_fob_kg - aircraft.taxi_fuel_kg
    landing = takeoff * math.exp(-trip_exponent(aircraft, flight))
    return (
        flight.planned_fob_kg <= aircraft.fuel_capacity_kg
        and takeoff <= aircraft.mtow_kg
        and landing <= aircraft.mlw_kg
    )


def generate_flight(
    index: int, rng: random.Random, fleet: dict[str, AircraftProfile], criteria: Criteria
) -> FlightPlan:
    scenario = rng.choices(list(SCENARIOS), weights=list(SCENARIOS.values()))[0]
    type_code = rng.choice(sorted(fleet))
    aircraft = fleet[type_code]
    low, high = _ROUTE_BOUNDS_NM[type_code]
    max_payload = aircraft.mzfw_kg - aircraft.oew_kg
    margin_fraction = rng.uniform(0.03, 0.08)

    flight = FlightPlan(
        flight_id=f"SIM{index:05d}",
        aircraft_type=type_code,
        origin=_airport(rng),
        destination=_airport(rng),
        alternate=_airport(rng),
        distance_nm=rng.uniform(low, high),
        alternate_distance_nm=rng.uniform(80, 300),
        payload_kg=max_payload * rng.uniform(0.55, 0.95),
        planned_fob_kg=1.0,  # Replaced below once the fuel needed is known.
        forecast_wind_kt=max(-90.0, min(90.0, rng.gauss(0, 30))),
        scenario=scenario,
    )
    # Shorten routes the aircraft can't lift, so only the intended scenarios fail dispatch.
    flight, required = _with_margin(aircraft, flight, criteria, margin_fraction)
    while not _is_dispatchable(aircraft, flight) and flight.distance_nm > _MIN_DISTANCE_NM:
        flight = replace(flight, distance_nm=flight.distance_nm * 0.9)
        flight, required = _with_margin(aircraft, flight, criteria, margin_fraction)

    if scenario == "MARGINAL_DISPATCH":
        flight, _ = _with_margin(aircraft, flight, criteria, rng.uniform(0.002, 0.012))
    elif scenario == "LOW_FUEL_DISPATCH":
        flight = replace(flight, planned_fob_kg=required * rng.uniform(0.90, 0.985))
    elif scenario == "OVERWEIGHT":
        flight = replace(flight, payload_kg=max_payload * rng.uniform(1.02, 1.10))
    return flight


def _true_wind(flight: FlightPlan, rng: random.Random) -> float:
    if flight.scenario == "HEADWIND":
        return flight.forecast_wind_kt - rng.uniform(40, 70)
    return flight.forecast_wind_kt + rng.gauss(0, 6)


def _leak_rate_kg_h(flight: FlightPlan, aircraft: AircraftProfile) -> float:
    """Leak size relative to cruise fuel flow, so it means the same for each aircraft class."""
    cruise_flow = (
        (aircraft.oew_kg + flight.payload_kg) * aircraft.cruise_tsfc_per_h / aircraft.cruise_ld
    )
    return cruise_flow * {"FUEL_LEAK": 0.20, "SEVERE_LEAK": 0.60}.get(flight.scenario, 0.0)


def _phase_factor(flown_nm: float, flight: FlightPlan, aircraft: AircraftProfile) -> float:
    if flown_nm < aircraft.climb_allowance_nm * 1.2:
        return _CLIMB_FLOW_FACTOR
    if flown_nm > flight.distance_nm - aircraft.climb_allowance_nm:
        return _DESCENT_FLOW_FACTOR
    return 1.0


def _true_ld(weight_kg: float, aircraft: AircraftProfile) -> float:
    """Efficiency is best at one weight, and falls away either side of it."""
    best_weight = _BEST_WEIGHT_FRACTION * aircraft.mtow_kg
    return aircraft.cruise_ld * (1 - _LD_WEIGHT_PENALTY * (weight_kg / best_weight - 1) ** 2)


def _segment_burn(
    flight: FlightPlan,
    aircraft: AircraftProfile,
    start: tuple[float, float],
    segment_nm: float,
    conditions: tuple[float, float, float],
) -> tuple[float, float]:
    """Steps the true fuel state through one segment. Returns the new fuel and hours taken."""
    flown_nm, fob = start
    wind, tsfc, leak_kg_h = conditions
    ground_speed = aircraft.cruise_tas_kt + wind
    steps = math.ceil(segment_nm / _STEP_NM)
    step_nm = segment_nm / steps
    step_hours = step_nm / ground_speed
    zero_fuel = aircraft.oew_kg + flight.payload_kg
    for step in range(steps):
        weight = zero_fuel + fob
        position = flown_nm + step * step_nm
        flow = weight * tsfc / _true_ld(weight, aircraft)
        flow *= _phase_factor(position, flight, aircraft)
        fob = max(fob - (flow + leak_kg_h) * step_hours, 0.0)
    return fob, step_hours * steps


def generate_telemetry(
    flight: FlightPlan, aircraft: AircraftProfile, rng: random.Random
) -> Iterator[Checkpoint]:
    """Steps along the route and yields the noisy gauge reading at each checkpoint."""
    sigma = _GAUGE_SIGMA_FRACTION * aircraft.fuel_capacity_kg
    segments = max(6, min(14, round(flight.distance_nm / 250)))
    segment_nm = flight.distance_nm / segments
    wind = _true_wind(flight, rng)
    tsfc = aircraft.cruise_tsfc_per_h * (1 + rng.gauss(0, 0.012))
    leak_kg_h = _leak_rate_kg_h(flight, aircraft)
    leak_start = rng.randint(1, max(1, segments // 2))
    is_fault = flight.scenario in ("GAUGE_HIGH", "GAUGE_LOW")
    fault_at = rng.randint(2, segments - 1) if is_fault else -1
    fault_sign = 1 if flight.scenario == "GAUGE_HIGH" else -1
    fob = flight.planned_fob_kg - aircraft.taxi_fuel_kg
    elapsed_h = 0.0

    for seq in range(segments + 1):
        reading = fob + rng.gauss(0, sigma)
        if seq == fault_at:
            reading = fob + fault_sign * aircraft.fuel_capacity_kg * rng.uniform(0.05, 0.15)
        yield Checkpoint(
            flight_id=flight.flight_id,
            seq=seq,
            elapsed_h=round(elapsed_h, 4),
            distance_flown_nm=round(seq * segment_nm, 2),
            fob_kg=round(max(reading, 0.0), 1),
            wind_kt=round(wind, 1),
        )
        if seq == segments:
            break
        wind_now = wind + rng.gauss(0, 2)
        leak_now = leak_kg_h if seq >= leak_start else 0.0
        fob, hours = _segment_burn(
            flight, aircraft, (seq * segment_nm, fob), segment_nm, (wind_now, tsfc, leak_now)
        )
        elapsed_h += hours


def generate_dataset(
    seed: int, count: int, fleet: dict[str, AircraftProfile], criteria: Criteria
) -> Iterator[tuple[FlightPlan, list[Checkpoint]]]:
    """Yields flights one at a time, so a large dataset never has to fit in memory."""
    rng = random.Random(seed)
    for index in range(1, count + 1):
        flight = generate_flight(index, rng, fleet, criteria)
        yield flight, list(generate_telemetry(flight, fleet[flight.aircraft_type], rng))
