"""Read and write the data files that form AFRM's integration boundary.

JSON and CSV are used, parsed by the standard library so no pickle, eval or network.
Files are size-limited, and each record is validated as it is read.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from afrm.errors import DataError
from afrm.models import Checkpoint, CheckpointAssessment, FlightAssessment, FlightPlan

MAX_INPUT_BYTES = 512 * 1024 * 1024
TELEMETRY_COLUMNS = ("flight_id", "seq", "elapsed_h", "distance_flown_nm", "fob_kg", "wind_kt")
_REQUIRED_FLIGHT_FIELDS = frozenset(
    {
        "flight_id",
        "aircraft_type",
        "origin",
        "destination",
        "alternate",
        "distance_nm",
        "alternate_distance_nm",
        "payload_kg",
        "planned_fob_kg",
        "forecast_wind_kt",
    }
)
_OPTIONAL_FLIGHT_FIELDS = frozenset({"scenario"})


def _check_size(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    if size > MAX_INPUT_BYTES:
        raise DataError(f"{path} is {size} bytes, above the {MAX_INPUT_BYTES} byte limit")


def _flight_from_record(record: dict[str, Any]) -> FlightPlan:
    missing = _REQUIRED_FLIGHT_FIELDS - record.keys()
    unknown = record.keys() - _REQUIRED_FLIGHT_FIELDS - _OPTIONAL_FLIGHT_FIELDS
    if missing or unknown:
        raise DataError(f"Bad flight record: missing {sorted(missing)}, unknown {sorted(unknown)}")
    try:
        return FlightPlan(**record)
    except TypeError as exc:
        raise DataError(f"Bad value in flight {record.get('flight_id')}: {exc}") from exc


def load_flights(path: Path) -> list[FlightPlan]:
    _check_size(path)
    try:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(records, list):
        raise DataError(f"{path}: expected a JSON list of flights")
    flights = [_flight_from_record(record) for record in records]
    identifiers = [flight.flight_id for flight in flights]
    if len(set(identifiers)) != len(identifiers):
        raise DataError(f"{path}: flight_id values are not unique")
    return flights


def save_flights(path: Path, flights: Iterable[FlightPlan]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump([asdict(flight) for flight in flights], stream, indent=1)
        stream.write("\n")


def write_telemetry_rows(stream: TextIO, rows: Iterable[Checkpoint]) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(
        (r.flight_id, r.seq, r.elapsed_h, r.distance_flown_nm, r.fob_kg, r.wind_kt) for r in rows
    )


def _checkpoint_from_row(row: dict[str, str], line: int) -> Checkpoint:
    try:
        return Checkpoint(
            flight_id=row["flight_id"],
            seq=int(row["seq"]),
            elapsed_h=float(row["elapsed_h"]),
            distance_flown_nm=float(row["distance_flown_nm"]),
            fob_kg=float(row["fob_kg"]),
            wind_kt=float(row["wind_kt"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"Telemetry line {line}: {exc}") from exc


def read_telemetry(path: Path) -> Iterator[tuple[str, Iterator[Checkpoint]]]:
    """Stream telemetry grouped by flight. Rows of one flight must be adjacent; this is what
    allows memory use to stay flat however long the file is."""
    _check_size(path)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != TELEMETRY_COLUMNS:
                raise DataError(f"{path}: expected columns {TELEMETRY_COLUMNS}")
            checkpoints = (_checkpoint_from_row(row, reader.line_num) for row in reader)
            seen: set[str] = set()
            for flight_id, group in itertools.groupby(checkpoints, key=lambda c: c.flight_id):
                if flight_id in seen:
                    raise DataError(f"{path}: rows for {flight_id} are not contiguous")
                seen.add(flight_id)
                yield flight_id, group
    except OSError as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc


def checkpoint_to_record(item: CheckpointAssessment) -> dict[str, Any]:
    record = asdict(item)
    record["status"] = item.status.name
    return record


def assessment_to_record(item: FlightAssessment) -> dict[str, Any]:
    """Stable JSON shape for downstream systems. Enum names are used, not numeric values."""
    dispatch = asdict(item.dispatch)
    dispatch["status"] = item.dispatch.status.value
    dispatch["reasons"] = list(item.dispatch.reasons)
    dispatch["plan"]["required_kg"] = item.dispatch.plan.required_kg
    return {
        "flight_id": item.flight.flight_id,
        "aircraft_type": item.flight.aircraft_type,
        "scenario": item.flight.scenario,
        "dispatch": dispatch,
        "worst_status": item.worst_status.name,
        "checkpoints": [checkpoint_to_record(c) for c in item.checkpoints],
    }
