import pytest
from hypothesis import given
from hypothesis import strategies as st

from afrm.errors import DataError
from afrm.performance import (
    air_distance_nm,
    air_range_nm,
    cruise_exponent,
    fuel_from_end_weight,
    fuel_from_start_weight,
    ground_from_air_nm,
)


def test_breguet_matches_hand_calculation() -> None:
    # k = 1000 * 0.6 / (450 * 17) = 0.0784314; e^k - 1 = 0.081589; * 60000 = 4895.3 kg
    exponent = cruise_exponent(1000, 450, 17, 0.6)
    assert exponent == pytest.approx(0.0784314, abs=1e-6)
    assert fuel_from_end_weight(60000, exponent) == pytest.approx(4895.3, abs=0.1)


def test_wind_changes_air_distance() -> None:
    assert air_distance_nm(900, 450, -50) == pytest.approx(1012.5)
    assert air_distance_nm(900, 450, 50) == pytest.approx(810.0)


def test_wind_at_or_beyond_airspeed_is_rejected() -> None:
    with pytest.raises(DataError):
        air_distance_nm(100, 450, -450)


@given(end_weight=st.floats(40000, 80000), exponent=st.floats(0.001, 0.5))
def test_forward_and_backward_forms_agree(end_weight: float, exponent: float) -> None:
    fuel = fuel_from_end_weight(end_weight, exponent)
    assert fuel_from_start_weight(end_weight + fuel, exponent) == pytest.approx(fuel)


@given(fuel=st.floats(100, 15000), start=st.floats(50000, 79000))
def test_range_reverses_fuel_burn(fuel: float, start: float) -> None:
    air = air_range_nm(start, fuel, 450, 17, 0.6)
    burned = fuel_from_start_weight(start, cruise_exponent(air, 450, 17, 0.6))
    assert burned == pytest.approx(fuel)


def test_range_edge_cases() -> None:
    assert air_range_nm(60000, 0, 450, 17, 0.6) == 0
    assert air_range_nm(60000, -5, 450, 17, 0.6) == 0
    with pytest.raises(DataError):
        air_range_nm(60000, 60000, 450, 17, 0.6)
    assert ground_from_air_nm(900, 450, 50) == pytest.approx(1000)
