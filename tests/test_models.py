import math

import pytest

from afrm.errors import DataError
from afrm.models import FlightPlan, FuelPlan, FuelStatus


def make_flight(**overrides: object) -> FlightPlan:
    values: dict[str, object] = {
        "flight_id": "T001",
        "aircraft_type": "GNB",
        "origin": "AAA",
        "destination": "BBB",
        "alternate": "CCC",
        "distance_nm": 1000.0,
        "alternate_distance_nm": 150.0,
        "payload_kg": 15000.0,
        "planned_fob_kg": 10000.0,
        "forecast_wind_kt": 0.0,
    }
    values.update(overrides)
    return FlightPlan(**values)  # type: ignore[arg-type]


def test_valid_flight_is_accepted() -> None:
    assert make_flight().distance_nm == 1000.0


@pytest.mark.parametrize(
    "bad",
    [
        {"distance_nm": -1.0},
        {"distance_nm": math.nan},
        {"payload_kg": -1.0},
        {"forecast_wind_kt": math.inf},
        {"planned_fob_kg": 0.0},
    ],
)
def test_invalid_flights_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(DataError):
        make_flight(**bad)


def test_fuel_plan_adds_up() -> None:
    plan = FuelPlan(100.0, 5000.0, 250.0, 700.0, 900.0, 3)
    assert plan.required_kg == 6950.0
    assert plan.diversion_minimum_kg == 1600.0


def test_status_is_ordered_by_severity() -> None:
    assert max(FuelStatus.NORMAL, FuelStatus.EMERGENCY) is FuelStatus.EMERGENCY
