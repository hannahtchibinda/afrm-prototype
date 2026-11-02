"""Command-line interface. Exit codes: 0 is success, 2 is invalid input or usage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from afrm import __version__
from afrm.config import load_aircraft, load_criteria
from afrm.dataio import (
    TELEMETRY_COLUMNS,
    assessment_to_record,
    load_flights,
    read_telemetry,
    save_flights,
    write_telemetry_rows,
)
from afrm.errors import AfrmError
from afrm.models import FlightAssessment
from afrm.pipeline import assess_all
from afrm.simulation import generate_dataset

_DISCLAIMER = "AFRM prototype: simulated data only. NOT FOR OPERATIONAL USE"


def _generate(args: argparse.Namespace) -> int:
    fleet = load_aircraft(args.aircraft)
    criteria = load_criteria(args.criteria)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flights = []
    with (args.out_dir / "telemetry.csv").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(",".join(TELEMETRY_COLUMNS) + "\n")
        for flight, rows in generate_dataset(args.seed, args.flights, fleet, criteria):
            flights.append(flight)
            write_telemetry_rows(stream, rows)
    save_flights(args.out_dir / "flights.json", flights)
    print(f"Wrote {len(flights)} flights to {args.out_dir} (seed {args.seed})")
    return 0


def _print_detail(item: FlightAssessment) -> None:
    dispatch, plan, flight = item.dispatch, item.dispatch.plan, item.flight
    print(f"\n{flight.flight_id} {flight.aircraft_type} {flight.origin}-{flight.destination}")
    print(
        f"  Distance {flight.distance_nm:.0f} NM, forecast wind {flight.forecast_wind_kt:+.0f} kt"
    )
    print(
        f"  Taxi {plan.taxi_kg:.0f} | Trip {plan.trip_kg:.0f} | Contingency "
        f"{plan.contingency_kg:.0f} | Alternate {plan.alternate_kg:.0f} | "
        f"Final reserve {plan.final_reserve_kg:.0f} kg"
    )
    print(f"  Required {plan.required_kg:.0f} kg, planned {flight.planned_fob_kg:.0f} kg")
    print(
        f"  Dispatch {dispatch.status.value}: usable margin {dispatch.usable_margin_kg:.0f} kg, "
        f"range margin {dispatch.range_margin_nm:.0f} NM {list(dispatch.reasons) or ''}"
    )
    print("  Seq  EFOB dest kg  Burn ratio  Range margin NM  Status")
    for point in item.checkpoints:
        efob = "-" if point.efob_destination_kg is None else f"{point.efob_destination_kg:.0f}"
        margin = "-" if point.range_margin_nm is None else f"{point.range_margin_nm:.0f}"
        flags = ("" if point.data_valid else " BAD-DATA") + (" DISCREPANCY" * point.discrepancy)
        print(
            f"  {point.seq:>3}  {efob:>12}  {point.burn_ratio:>10.2f}  {margin:>15}  "
            f"{point.status.name}{flags}"
        )


def _assess(args: argparse.Namespace) -> int:
    fleet = load_aircraft(args.aircraft)
    criteria = load_criteria(args.criteria)
    flights = load_flights(args.flights)
    wanted = set(args.detail or ())
    dispatch_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    output = None
    print(_DISCLAIMER)
    try:
        if args.json:
            output = args.json.open("w", encoding="utf-8", newline="\n")
        for item in assess_all(flights, read_telemetry(args.telemetry), fleet, criteria):
            dispatch_counts[item.dispatch.status.value] += 1
            status_counts[item.worst_status.name] += 1
            if output:
                # JSON Lines: one record at a time, so results never accumulate in memory.
                output.write(json.dumps(assessment_to_record(item)) + "\n")
            if item.flight.flight_id in wanted:
                _print_detail(item)
    finally:
        if output:
            output.close()
    print(f"\nFlights assessed: {sum(dispatch_counts.values())}")
    print(f"Dispatch: {dict(sorted(dispatch_counts.items()))}")
    print(f"Worst in-flight status: {dict(sorted(status_counts.items()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afrm", description=_DISCLAIMER)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--aircraft", type=Path, help="Aircraft TOML (default: built-in)")
    parser.add_argument("--criteria", type=Path, help="Criteria TOML (default: built-in)")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate simulated flights and telemetry")
    generate.add_argument("--seed", type=int, default=2026)
    generate.add_argument("--flights", type=int, default=200)
    generate.add_argument("--out-dir", type=Path, default=Path("data"))
    generate.set_defaults(handler=_generate)

    assess = commands.add_parser("assess", help="Assess flights against the criteria")
    assess.add_argument("--flights", type=Path, default=Path("data/flights.json"))
    assess.add_argument("--telemetry", type=Path, default=Path("data/telemetry.csv"))
    assess.add_argument("--json", type=Path, help="Write results as JSON Lines")
    assess.add_argument("--detail", nargs="*", help="Flight IDs to print in full")
    assess.set_defaults(handler=_assess)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except AfrmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
