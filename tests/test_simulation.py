from collections import defaultdict

import pytest

from afrm.config import load_aircraft, load_criteria
from afrm.models import (
    Checkpoint,
    DispatchStatus,
    FlightAssessment,
    FlightPlan,
    FuelStatus,
)
from afrm.monitoring import FlightMonitor
from afrm.planning import assess_dispatch
from afrm.simulation import SCENARIOS, generate_dataset

FLEET = load_aircraft()
CRITERIA = load_criteria()


def assess(flight: FlightPlan, samples: list[Checkpoint]) -> FlightAssessment:
    aircraft = FLEET[flight.aircraft_type]
    dispatch = assess_dispatch(aircraft, flight, CRITERIA)
    monitor = FlightMonitor(aircraft, flight, dispatch.plan, CRITERIA.inflight)
    return FlightAssessment(flight, dispatch, tuple(monitor.update(s) for s in samples))


@pytest.fixture(scope="module")
def outcomes() -> dict[str, list[FlightAssessment]]:
    grouped: dict[str, list[FlightAssessment]] = defaultdict(list)
    for flight, samples in generate_dataset(11, 3000, FLEET, CRITERIA):
        grouped[flight.scenario].append(assess(flight, samples))
    return grouped


def share(items: list[FlightAssessment], flagged: list[FlightAssessment]) -> float:
    return len(flagged) / len(items)


def test_same_seed_gives_identical_data() -> None:
    first = list(generate_dataset(7, 30, FLEET, CRITERIA))
    second = list(generate_dataset(7, 30, FLEET, CRITERIA))
    assert first == second


def test_different_seed_gives_different_data() -> None:
    first = list(generate_dataset(7, 10, FLEET, CRITERIA))
    second = list(generate_dataset(8, 10, FLEET, CRITERIA))
    assert first != second


def test_all_scenarios_are_generated(outcomes: dict[str, list[FlightAssessment]]) -> None:
    assert set(outcomes) == set(SCENARIOS)


def test_nominal_flights_dispatch_and_rarely_alarm(
    outcomes: dict[str, list[FlightAssessment]],
) -> None:

    items = outcomes["NOMINAL"]
    assert all(i.dispatch.status is DispatchStatus.GO for i in items)
    raised = [i for i in items if i.worst_status is not FuelStatus.NORMAL]
    assert share(items, raised) < 0.02
    # Known limit: extra climb burn can cause a brief false EMERGENCY on a long flight.
    emergencies = [i for i in items if i.worst_status is FuelStatus.EMERGENCY]
    assert share(items, emergencies) < 0.005


def test_dispatch_scenarios_are_caught(outcomes: dict[str, list[FlightAssessment]]) -> None:
    for item in outcomes["MARGINAL_DISPATCH"]:
        assert item.dispatch.status is DispatchStatus.MARGINAL
    for name in ("LOW_FUEL_DISPATCH", "OVERWEIGHT"):
        assert all(i.dispatch.status is DispatchStatus.NO_GO for i in outcomes[name])


def test_leaks_are_detected(outcomes: dict[str, list[FlightAssessment]]) -> None:
    for name in ("FUEL_LEAK", "SEVERE_LEAK"):
        items = outcomes[name]
        flagged = [i for i in items if any(c.discrepancy for c in i.checkpoints)]
        assert share(items, flagged) > 0.95


def test_high_gauge_fault_is_rejected(outcomes: dict[str, list[FlightAssessment]]) -> None:
    for item in outcomes["GAUGE_HIGH"]:
        assert any(not c.data_valid for c in item.checkpoints)
        assert item.worst_status is not FuelStatus.EMERGENCY


def test_low_gauge_fault_raises_an_alarm_known_limit(
    outcomes: dict[str, list[FlightAssessment]],
) -> None:
    """Known limit: a falsely low reading looks like a leak, so it causes a false alarm."""
    items = outcomes["GAUGE_LOW"]
    raised = [i for i in items if i.worst_status is not FuelStatus.NORMAL]
    assert share(items, raised) > 0.9


def test_scenario_label_does_not_affect_the_result() -> None:
    flight, samples = next(generate_dataset(3, 1, FLEET, CRITERIA))
    relabelled = FlightPlan(
        flight.flight_id,
        flight.aircraft_type,
        flight.origin,
        flight.destination,
        flight.alternate,
        flight.distance_nm,
        flight.alternate_distance_nm,
        flight.payload_kg,
        flight.planned_fob_kg,
        flight.forecast_wind_kt,
        "SOMETHING_ELSE",
    )
    assert assess(flight, samples).checkpoints == assess(relabelled, samples).checkpoints
