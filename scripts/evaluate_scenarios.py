"""Measure algorithm behaviour per scenario over a large simulated population.

Usage: python scripts/evaluate_scenarios.py [--seed 60221] [--flights 10000]
Writes docs/evidence/scenario_evaluation.json. The default seed is held out as it wasn't used by
the tests, the committed dataset or during debugging, and thresholds were fixed before it was run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from afrm.config import load_aircraft, load_criteria
from afrm.models import FuelStatus
from afrm.pipeline import assess_flight
from afrm.simulation import generate_dataset

OUT = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "scenario_evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=60221)
    parser.add_argument("--flights", type=int, default=10000)
    args = parser.parse_args()

    fleet = load_aircraft()
    criteria = load_criteria()
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for flight, samples in generate_dataset(args.seed, args.flights, fleet, criteria):
        item = assess_flight(flight, samples, fleet, criteria)
        row = rows[flight.scenario]
        row["flights"] += 1
        row[f"dispatch_{item.dispatch.status.value}"] += 1
        row[f"worst_{item.worst_status.name}"] += 1
        row["any_discrepancy"] += any(c.discrepancy for c in item.checkpoints)
        row["any_rejected_reading"] += any(not c.data_valid for c in item.checkpoints)
        row["any_alert"] += item.worst_status is not FuelStatus.NORMAL or any(
            c.discrepancy for c in item.checkpoints
        )
    result = {"seed": args.seed, "flights": args.flights, "scenarios": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", "utf-8", newline="\n")
    for name, row in sorted(rows.items()):
        print(name, dict(sorted(row.items())))


if __name__ == "__main__":
    main()
