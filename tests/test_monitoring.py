import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_models import make_flight

from afrm.config import load_aircraft, load_criteria
from afrm.models import Checkpoint, FuelPlan, FuelStatus
from afrm.monitoring import FlightMonitor, classify
from afrm.performance import cruise_exponent, fuel_from_start_weight
from afrm.planning import compute_fuel_plan

GNB = load_aircraft()["GNB"]
CRITERIA = load_criteria()
FLIGHT = make_flight()
REAL_PLAN = compute_fuel_plan(GNB, FLIGHT, CRITERIA.planning)
START_FOB = REAL_PLAN.required_kg - REAL_PLAN.taxi_kg
ZERO_FUEL = 57000.0
TEST_PLAN = FuelPlan(250, 5000, 250, 700, 900, 3)


def monitor() -> FlightMonitor:
    return FlightMonitor(GNB, FLIGHT, REAL_PLAN, CRITERIA.inflight)


def flown(extra_per_segment: float = 0.0, segments: int = 8) -> list[Checkpoint]:
    """A perfect flight that burns exactly what the model predicts, plus any extra."""
    samples = []
    fob = START_FOB
    for seq in range(segments + 1):
        samples.append(Checkpoint("T001", seq, seq * 0.25, 1000.0 * seq / segments, fob, 0.0))
        air = 1000.0 / segments + (GNB.climb_allowance_nm if seq == 0 else 0)
        exponent = cruise_exponent(air, GNB.cruise_tas_kt, GNB.cruise_ld, GNB.cruise_tsfc_per_h)
        fob -= extra_per_segment + fuel_from_start_weight(ZERO_FUEL + fob, exponent)
    return samples


@pytest.mark.parametrize(
    ("efob", "expected"),
    [
        (1601, FuelStatus.NORMAL),
        (1600, FuelStatus.NORMAL),
        (1599, FuelStatus.CAUTION),
        (1200, FuelStatus.CAUTION),
        (1199, FuelStatus.MINIMUM_FUEL),
        (900, FuelStatus.MINIMUM_FUEL),
        (899, FuelStatus.EMERGENCY),
        (-50, FuelStatus.EMERGENCY),
    ],
)
def test_classification_boundaries(efob: float, expected: FuelStatus) -> None:
    assert classify(efob, TEST_PLAN, min_fuel_buffer_kg=300) is expected


@given(st.floats(-1000, 20000), st.floats(0.1, 1000))
def test_less_fuel_is_never_less_severe(efob: float, reduction: float) -> None:
    assert classify(efob - reduction, TEST_PLAN, 300) >= classify(efob, TEST_PLAN, 300)


def test_perfect_flight_stays_normal_with_steady_estimate() -> None:
    m = monitor()
    results = [m.update(s) for s in flown()]
    assert all(r.status is FuelStatus.NORMAL for r in results)
    assert not any(r.discrepancy for r in results)
    efobs = [r.efob_destination_kg for r in results]
    assert max(efobs) - min(efobs) < 1.0
    unburned = REAL_PLAN.contingency_kg + REAL_PLAN.alternate_kg + REAL_PLAN.final_reserve_kg
    assert efobs[0] == pytest.approx(unburned, abs=1.0)


def test_range_margin_is_positive_with_reserves_on_board() -> None:
    result = monitor().update(Checkpoint("T001", 0, 0.0, 0.0, START_FOB, 0.0))
    assert result.range_margin_nm > 0
    assert result.range_to_final_reserve_nm > 1000


def test_leak_raises_discrepancy_and_escalates() -> None:
    m = monitor()
    results = [m.update(s) for s in flown(extra_per_segment=350.0)]
    assert any(r.discrepancy for r in results)
    assert max(r.status for r in results) >= FuelStatus.CAUTION
    assert max(r.burn_ratio for r in results) > 1.2


def test_better_than_model_burn_earns_no_credit() -> None:
    m = monitor()
    results = [m.update(s) for s in flown(extra_per_segment=-50.0)]
    assert all(r.data_valid and r.burn_ratio == 1.0 for r in results)


def test_small_excess_within_noise_is_ignored() -> None:
    m = monitor()
    assert all(m.update(s).burn_ratio == 1.0 for s in flown(extra_per_segment=10.0))


@pytest.mark.parametrize(
    "bad",
    [
        {"fob_kg": math.nan},
        {"fob_kg": math.inf},
        {"fob_kg": -1.0},
        {"fob_kg": 50000.0},
        {"wind_kt": math.nan},
        {"wind_kt": -500.0},
        {"distance_flown_nm": 1500.0},
        {"distance_flown_nm": -1.0},
    ],
)
def test_bad_reading_is_rejected(bad: dict[str, float]) -> None:
    sample = replace(Checkpoint("T001", 0, 0.0, 0.0, 7000.0, 0.0), **bad)
    result = monitor().update(sample)
    assert not result.data_valid
    assert result.efob_destination_kg is None


def test_fuel_rise_and_backward_position_are_rejected() -> None:
    m = monitor()
    m.update(Checkpoint("T001", 0, 0.0, 0.0, 7000.0, 0.0))
    assert m.update(Checkpoint("T001", 1, 0.2, 100.0, 5950.0, 0.0)).data_valid
    assert not m.update(Checkpoint("T001", 2, 0.4, 200.0, 9000.0, 0.0)).data_valid
    assert not m.update(Checkpoint("T001", 3, 0.6, 50.0, 5900.0, 0.0)).data_valid


def test_spike_hidden_by_segment_burn_is_rejected() -> None:
    m = monitor()
    m.update(Checkpoint("T001", 0, 0.0, 0.0, 7000.0, 0.0))
    # 500 NM plus climb burns about 3,000 kg, so 6,900 kg is way far above prediction.
    assert not m.update(Checkpoint("T001", 1, 1.1, 500.0, 6900.0, 0.0)).data_valid


def test_sudden_fall_is_believed_as_possible_leak() -> None:
    m = monitor()
    m.update(Checkpoint("T001", 0, 0.0, 0.0, 7000.0, 0.0))
    assert m.update(Checkpoint("T001", 1, 0.2, 100.0, 3000.0, 0.0)).data_valid


def test_empty_tank_reading_is_accepted() -> None:
    m = monitor()
    m.update(Checkpoint("T001", 0, 0.0, 0.0, 300.0, 0.0))
    result = m.update(Checkpoint("T001", 1, 1.1, 500.0, 0.0, 0.0))
    assert result.data_valid
    assert result.status is FuelStatus.EMERGENCY
    assert result.range_to_final_reserve_nm == 0


def test_repeated_bad_data_raises_status_to_caution() -> None:
    m = monitor()
    m.update(Checkpoint("T001", 0, 0.0, 0.0, 7000.0, 0.0))
    first = m.update(Checkpoint("T001", 1, 0.2, 100.0, math.nan, 0.0))
    second = m.update(Checkpoint("T001", 2, 0.4, 200.0, math.nan, 0.0))
    assert first.status is FuelStatus.NORMAL
    assert second.status is FuelStatus.CAUTION
    assert m.update(Checkpoint("T001", 3, 0.6, 300.0, 4950.0, 0.0)).data_valid


def test_bad_data_never_lowers_status() -> None:
    m = monitor()
    assert m.update(Checkpoint("T001", 0, 0.0, 0.0, 1500.0, 0.0)).status is FuelStatus.EMERGENCY
    held = m.update(Checkpoint("T001", 1, 0.2, 100.0, math.nan, 0.0))
    assert held.status is FuelStatus.EMERGENCY
