from pathlib import Path

import pytest

from afrm.config import load_aircraft, load_criteria
from afrm.errors import DataError

DEFAULT_CRITERIA = Path(__file__).parents[1] / "src" / "afrm" / "defaults" / "criteria.toml"


def test_defaults_load() -> None:
    assert set(load_aircraft()) == {"GRJ", "GNB", "GWB"}
    assert load_criteria().planning.final_reserve_min == 30.0


def test_criteria_can_change_without_code(tmp_path: Path) -> None:
    text = DEFAULT_CRITERIA.read_text("utf-8")
    custom = tmp_path / "criteria.toml"
    custom.write_text(text.replace("final_reserve_min = 30.0", "final_reserve_min = 45.0"), "utf-8")
    assert load_criteria(custom).planning.final_reserve_min == 45.0


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("schema_version = 1", "schema_version = 99"),
        ("contingency_fraction = 0.05", "contingency_fraction = -0.05"),
        ("final_reserve_min = 30.0", "final_reserve_min = 0.0"),
        ("discrepancy_ratio = 1.08", "discrepancy_ratio = 0.9"),
        ("burn_min_segments = 2", "burn_min_segments = 9"),
        ("marginal_fraction = 0.02", "unknown_key = 0.02"),
    ],
)
def test_unsafe_criteria_are_rejected(tmp_path: Path, old: str, new: str) -> None:
    text = DEFAULT_CRITERIA.read_text("utf-8")
    assert old in text
    custom = tmp_path / "criteria.toml"
    custom.write_text(text.replace(old, new), "utf-8")
    with pytest.raises(DataError):
        load_criteria(custom)


def test_bad_aircraft_files_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "aircraft.toml"
    path.write_text("[X]\nunknown = 1\n", "utf-8")
    with pytest.raises(DataError):
        load_aircraft(path)
    path.write_text("not toml [", "utf-8")
    with pytest.raises(DataError):
        load_aircraft(path)
