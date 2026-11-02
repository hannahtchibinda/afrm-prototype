"""Join flight plans and telemetry and run each flight through the algorithm."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from afrm.errors import DataError
from afrm.models import AircraftProfile, Checkpoint, Criteria, FlightAssessment, FlightPlan
from afrm.monitoring import FlightMonitor
from afrm.planning import assess_dispatch


def assess_flight(
    flight: FlightPlan,
    telemetry: Iterable[Checkpoint],
    fleet: dict[str, AircraftProfile],
    criteria: Criteria,
) -> FlightAssessment:
    aircraft = fleet.get(flight.aircraft_type)
    if aircraft is None:
        raise DataError(f"{flight.flight_id}: unknown aircraft type {flight.aircraft_type}")
    dispatch = assess_dispatch(aircraft, flight, criteria)
    monitor = FlightMonitor(aircraft, flight, dispatch.plan, criteria.inflight)
    return FlightAssessment(flight, dispatch, tuple(monitor.update(s) for s in telemetry))


def assess_all(
    flights: Iterable[FlightPlan],
    telemetry: Iterable[tuple[str, Iterable[Checkpoint]]],
    fleet: dict[str, AircraftProfile],
    criteria: Criteria,
) -> Iterator[FlightAssessment]:
    """Yield one assessment per flight, in telemetry order. A flight with no telemetry is still
    assessed for dispatch; telemetry for an unknown flight is an error, not silently dropped."""
    pending = {flight.flight_id: flight for flight in flights}
    for flight_id, samples in telemetry:
        flight = pending.pop(flight_id, None)
        if flight is None:
            raise DataError(f"Telemetry refers to unknown flight {flight_id}")
        yield assess_flight(flight, samples, fleet, criteria)
    for flight in pending.values():
        yield assess_flight(flight, (), fleet, criteria)
