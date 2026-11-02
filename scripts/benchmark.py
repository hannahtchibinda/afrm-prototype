"""Measure the resources AFRM uses. Writes docs/evidence/benchmark.json.

Each measurement runs in a fresh subprocess so that one cannot warm the caches of another one.
Time and memory are measured separately because tracemalloc itself slows the execution.
Energy isn't measured: it is estimated afterwards from CPU time and a power assumption.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import tempfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from afrm.models import Checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence" / "benchmark.json"
REPEATS = 5

_RUN = """
import sys, time, tracemalloc
from pathlib import Path
from afrm.config import load_aircraft, load_criteria
from afrm.dataio import load_flights, read_telemetry
from afrm.pipeline import assess_all
folder, mode = Path(sys.argv[1]), sys.argv[2]
fleet, criteria = load_aircraft(), load_criteria()
if mode == "memory":
    tracemalloc.start()
flights = load_flights(folder / "flights.json")
telemetry = read_telemetry(folder / "telemetry.csv")
if len(sys.argv) > 3:  # Eager comparison: hold every reading in memory first.
    telemetry = [(key, list(group)) for key, group in telemetry]
wall, cpu = time.perf_counter(), time.process_time()
count = sum(1 for _ in assess_all(flights, telemetry, fleet, criteria))
wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
peak = tracemalloc.get_traced_memory()[1] if mode == "memory" else 0
print(count, wall, cpu, peak)
"""


def _run(folder: str, mode: str, *flags: str, eager: bool = False) -> tuple[float, float, int]:
    command = [sys.executable, *flags, "-c", _RUN, folder, mode]
    if eager:
        command.append("eager")
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    _, wall, cpu, peak = output.split()
    return float(wall), float(cpu), int(peak)


def _time(folder: str, *flags: str) -> dict[str, float]:
    samples = [_run(folder, "time", *flags) for _ in range(REPEATS)]
    return {
        "wall_s_median": statistics.median(s[0] for s in samples),
        "cpu_s_median": statistics.median(s[1] for s in samples),
        "cpu_s_stdev": statistics.stdev(s[1] for s in samples),
    }


@dataclass
class _WithoutSlots:
    """Same fields as the real Checkpoint, but without slots, for comparison."""

    flight_id: str
    seq: int
    elapsed_h: float
    distance_flown_nm: float
    fob_kg: float
    wind_kt: float


def _bytes_per_record(cls: type) -> float:
    """Records with distinct float values, so that no values are shared between them."""
    count = 100_000
    tracemalloc.start()
    held = [cls("X", i, i + 0.1, i + 0.2, i + 0.3, i + 0.4) for i in range(count)]
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del held
    return peak / count


def main() -> None:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "repeats": REPEATS,
    }
    scaling = {}
    with tempfile.TemporaryDirectory() as tmp:
        for count in (1000, 10000, 50000):
            folder = f"{tmp}/{count}"
            command = [sys.executable, "-m", "afrm.cli", "generate", "--seed", "5"]
            subprocess.run(
                [*command, "--flights", str(count), "--out-dir", folder],
                check=True,
                capture_output=True,
            )
            telemetry_bytes = Path(folder, "telemetry.csv").stat().st_size
            timing = _time(folder)
            peak = _run(folder, "memory")[2]
            scaling[count] = {
                **timing,
                "peak_mib": peak / 2**20,
                "telemetry_mib": telemetry_bytes / 2**20,
                "us_per_flight": timing["cpu_s_median"] / count * 1e6,
            }
            if count == 10000:
                result["eager_peak_mib_10000"] = _run(folder, "memory", eager=True)[2] / 2**20
                result["streaming_peak_mib_10000"] = peak / 2**20
                result["runtime_modes_10000"] = {
                    "default": timing,
                    "dev_mode (-X dev)": _time(folder, "-X", "dev"),
                    "optimised (-O)": _time(folder, "-O"),
                }
    result["scaling"] = scaling
    result["bytes_per_record"] = {
        "checkpoint_with_slots": _bytes_per_record(Checkpoint),
        "without_slots": _bytes_per_record(_WithoutSlots),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n", "utf-8", newline="\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
