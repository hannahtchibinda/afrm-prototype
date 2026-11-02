from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_models import make_flight

from afrm.config import load_aircraft, load_criteria
from afrm.errors import ConvergenceError, DataError
from afrm.models import DispatchAssessment, DispatchStatus
from afrm.planning import assess_dispatch, compute_fuel_plan

GNB = load_aircraft()["GNB"]
CRITERIA = load_criteria()


def test_fuel_plan_matches_hand_calculation() -> None:
    plan = compute_fuel_plan(GNB, make_flight(), CRITERIA.planning)
    assert plan.final_reserve_kg == pytest.approx(877.52, abs=0.05)
    assert plan.alternate_kg == pytest.approx(684.93, abs=0.05)
    assert plan.trip_kg == pytest.approx(5300.68, abs=0.05)
    assert plan.contingency_kg == pytest.approx(265.03, abs=0.05)
    assert plan.required_kg == pytest.approx(7378.17, abs=0.1)


def test_contingency_floor_applies_on_short_sector() -> None:
    plan = compute_fuel_plan(GNB, make_flight(distance_nm=50.0), CRITERIA.planning)
    assert plan.contingency_kg > 0.05 * plan.trip_kg


def test_iteration_failure_is_reported() -> None:
    bad = replace(CRITERIA.planning, contingency_fraction=500.0)
    with pytest.raises(ConvergenceError):
        compute_fuel_plan(GNB, make_flight(distance_nm=3000.0), bad)


@given(
    distance=st.floats(200, 2600),
    payload=st.floats(0, 20000),
    wind=st.floats(-100, 100),
)
@settings(max_examples=200, deadline=None)
def test_longer_flights_always_need_more_fuel(distance: float, payload: float, wind: float) -> None:
    flight = make_flight(distance_nm=distance, payload_kg=payload, forecast_wind_kt=wind)
    plan = compute_fuel_plan(GNB, flight, CRITERIA.planning)
    longer = compute_fuel_plan(GNB, replace(flight, distance_nm=distance + 50), CRITERIA.planning)
    assert plan.iterations < 10
    assert longer.required_kg > plan.required_kg


def test_range_closes_at_exactly_required_fuel() -> None:
    flight = make_flight(forecast_wind_kt=-20.0)
    plan = compute_fuel_plan(GNB, flight, CRITERIA.planning)
    result = assess_dispatch(GNB, replace(flight, planned_fob_kg=plan.required_kg), CRITERIA)
    assert result.range_margin_nm == pytest.approx(0.0, abs=0.5)
    assert result.tankering_penalty_kg == pytest.approx(0.0, abs=0.1)


def at_margin(extra_kg: float) -> DispatchAssessment:
    flight = make_flight()
    required = compute_fuel_plan(GNB, flight, CRITERIA.planning).required_kg
    return assess_dispatch(GNB, replace(flight, planned_fob_kg=required + extra_kg), CRITERIA)


def test_dispatch_go() -> None:
    assert at_margin(500).status is DispatchStatus.GO


def test_dispatch_marginal() -> None:
    assert at_margin(50).status is DispatchStatus.MARGINAL


def test_dispatch_no_go_below_required() -> None:
    result = at_margin(-1)
    assert result.status is DispatchStatus.NO_GO
    assert "fuel below minimum required" in result.reasons


def test_extra_fuel_costs_fuel_to_carry() -> None:
    result = at_margin(1000)
    assert 0 < result.tankering_penalty_kg < 200
    assert result.usable_margin_kg == pytest.approx(1000 - result.tankering_penalty_kg)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"planned_fob_kg": 19001.0}, "fuel exceeds tank capacity"),
        ({"payload_kg": 21000.0, "planned_fob_kg": 8000.0}, "zero-fuel weight exceeds MZFW"),
        (
            {"payload_kg": 20000.0, "planned_fob_kg": 18000.0, "distance_nm": 1500.0},
            "take-off weight exceeds MTOW",
        ),
        (
            {"payload_kg": 20000.0, "planned_fob_kg": 16000.0, "distance_nm": 500.0},
            "landing weight exceeds MLW",
        ),
    ],
)
def test_dispatch_limits(overrides: dict[str, float], reason: str) -> None:
    result = assess_dispatch(GNB, make_flight(**overrides), CRITERIA)
    assert result.status is DispatchStatus.NO_GO
    assert reason in result.reasons


def test_profile_mismatch_is_rejected() -> None:
    with pytest.raises(DataError):
        assess_dispatch(GNB, make_flight(aircraft_type="GWB"), CRITERIA)
