"""End-to-end checks. The command line, the data files and bad input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_models import make_flight

from afrm.cli import main
from afrm.config import load_aircraft, load_criteria
from afrm.dataio import load_flights, read_telemetry, save_flights
from afrm.errors import DataError
from afrm.models import FuelStatus
from afrm.pipeline import assess_all, assess_flight
from afrm.planning import compute_fuel_plan

FLEET = load_aircraft()
CRITERIA = load_criteria()
CRITERIA_FILE = Path(__file__).parents[1] / "src/afrm/defaults/criteria.toml"
HEADER = "flight_id,seq,elapsed_h,distance_flown_nm,fob_kg,wind_kt\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_files_are_identical_for_the_same_seed(tmp_path: Path) -> None:
    for name in ("one", "two"):
        args = ["generate", "--seed", "7", "--flights", "30", "--out-dir", str(tmp_path / name)]
        assert main(args) == 0
    for filename in ("flights.json", "telemetry.csv"):
        assert sha(tmp_path / "one" / filename) == sha(tmp_path / "two" / filename)


def test_command_line_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "--seed", "3", "--flights", "20", "--out-dir", str(tmp_path)]) == 0
    out = tmp_path / "out.jsonl"
    code = main(
        [
            "assess",
            "--flights",
            str(tmp_path / "flights.json"),
            "--telemetry",
            str(tmp_path / "telemetry.csv"),
            "--json",
            str(out),
            "--detail",
            "SIM00001",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "NOT FOR OPERATIONAL USE" in printed
    assert "Flights assessed: 20" in printed
    assert "SIM00001" in printed
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 20
    assert {"flight_id", "dispatch", "worst_status", "checkpoints"} <= records[0].keys()
    assert records[0]["dispatch"]["plan"]["required_kg"] > 0


def test_changing_criteria_file_changes_result_without_code_change(tmp_path: Path) -> None:
    text = CRITERIA_FILE.read_text("utf-8")
    assert "final_reserve_min = 30.0" in text
    custom = tmp_path / "criteria.toml"
    custom.write_text(text.replace("final_reserve_min = 30.0", "final_reserve_min = 45.0"), "utf-8")
    base = compute_fuel_plan(FLEET["GNB"], make_flight(), CRITERIA.planning)
    raised = compute_fuel_plan(FLEET["GNB"], make_flight(), load_criteria(custom).planning)
    assert raised.final_reserve_kg == pytest.approx(base.final_reserve_kg * 1.5, rel=0.01)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"a": 1}',
        '[{"flight_id": "X"}]',
        '[{"flight_id":"X","aircraft_type":"GNB","origin":"A","destination":"B","alternate":"C",'
        '"distance_nm":"far","alternate_distance_nm":1,"payload_kg":1,"planned_fob_kg":1,'
        '"forecast_wind_kt":0}]',
    ],
)
def test_bad_flight_file_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "f.json"
    path.write_text(content, "utf-8")
    with pytest.raises(DataError):
        load_flights(path)


def test_duplicate_flights_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    save_flights(path, [make_flight(), make_flight()])
    with pytest.raises(DataError):
        load_flights(path)


@pytest.mark.parametrize(
    "content",
    [
        "wrong,columns\n1,2\n",
        HEADER + "T,notint,0,0,1,0\n",
        HEADER + "A,0,0,0,1,0\nB,0,0,0,1,0\nA,1,0,0,1,0\n",
    ],
)
def test_bad_telemetry_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "t.csv"
    path.write_text(content, "utf-8")
    with pytest.raises(DataError):
        for _, group in read_telemetry(path):
            list(group)


def test_nan_reading_reaches_the_monitor_as_bad_data(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text(HEADER + "T001,0,0,0,7000,0\nT001,1,0.2,100,nan,0\n", "utf-8")
    result = list(assess_all([make_flight()], read_telemetry(path), FLEET, CRITERIA))
    assert result[0].checkpoints[1].data_valid is False


def test_unknown_flight_and_aircraft_are_errors(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text(HEADER + "NOPE,0,0,0,7000,0\n", "utf-8")
    with pytest.raises(DataError):
        list(assess_all([make_flight()], read_telemetry(path), FLEET, CRITERIA))
    with pytest.raises(DataError):
        assess_flight(make_flight(aircraft_type="ZZZ"), (), FLEET, CRITERIA)


def test_flight_without_telemetry_gets_dispatch_only() -> None:
    result = list(assess_all([make_flight()], [], FLEET, CRITERIA))
    assert result[0].checkpoints == ()
    assert result[0].worst_status is FuelStatus.NORMAL


def test_command_line_reports_bad_input_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "assess",
            "--flights",
            str(tmp_path / "absent.json"),
            "--telemetry",
            str(tmp_path / "absent.csv"),
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err
