"""Run every quality check and record the results.
Usage: python scripts/verify.py
Exit code is non-zero if any check fails. The log then goes to docs/evidence/verify.txt.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GATES: list[tuple[str, list[str]]] = [
    ("Lint (ruff)", [PY, "-m", "ruff", "check", "src", "tests", "scripts"]),
    ("Format (ruff)", [PY, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
    ("Types (mypy --strict)", [PY, "-m", "mypy"]),
    ("Security lint (bandit)", [PY, "-m", "bandit", "-q", "-r", "src", "-c", "pyproject.toml"]),
    (
        "Dependency audit (pip-audit)",
        [PY, "-m", "pip_audit", "-r", "requirements-dev.lock", "--require-hashes", "--disable-pip"],
    ),
    (
        "Tests + coverage (dev mode)",
        [PY, "-X", "dev", "-m", "pytest", "-q", "--cov", "--cov-report=term-missing"],
    ),
    ("Wheel build", [PY, "-m", "build", "--wheel", "--outdir", "dist"]),
]


def _same_flights(fresh: Path, committed: Path) -> bool:
    """Operating systems can differ in the last digit of a calculation, so numbers get a
    tiny tolerance. Everything else must match exactly."""
    new = json.loads(fresh.read_text("utf-8"))
    old = json.loads(committed.read_text("utf-8"))
    if len(new) != len(old):
        return False
    for new_flight, old_flight in zip(new, old, strict=True):
        if new_flight.keys() != old_flight.keys():
            return False
        for key, value in new_flight.items():
            other = old_flight[key]
            if isinstance(value, float):
                if not math.isclose(value, other, rel_tol=1e-9):
                    return False
            elif value != other:
                return False
    return True


def _data_reproducible() -> tuple[bool, str]:
    """Regenerating with the documented seed must match the committed files."""
    with tempfile.TemporaryDirectory() as tmp:
        command = [PY, "-m", "afrm.cli", "generate", "--seed", "2026", "--flights", "200"]
        subprocess.run([*command, "--out-dir", tmp], check=True, capture_output=True)
        flights_ok = _same_flights(Path(tmp) / "flights.json", ROOT / "data" / "flights.json")

        fresh = (Path(tmp) / "telemetry.csv").read_text("utf-8")
        telemetry_ok = fresh == (ROOT / "data" / "telemetry.csv").read_text("utf-8")
    detail = f"flights.json matches: {flights_ok}\ntelemetry.csv matches: {telemetry_ok}\n"
    return flights_ok and telemetry_ok, detail


def main() -> int:
    log = [f"Python {platform.python_version()} on {platform.platform()}\n"]
    results = []
    for name, command in GATES:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        passed = run.returncode == 0
        results.append((name, passed))
        log.append(f"\n=== {name}: {'PASS' if passed else 'FAIL'} ===\n{run.stdout}{run.stderr}")
    passed, detail = _data_reproducible()
    results.append(("Data reproducibility", passed))
    log.append(f"\n=== Data reproducibility: {'PASS' if passed else 'FAIL'} ===\n{detail}")

    summary = "".join(f"{'PASS' if ok else 'FAIL'}  {name}\n" for name, ok in results)
    out = ROOT / "docs" / "evidence" / "verify.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary + "".join(log), encoding="utf-8", newline="\n")
    print(summary, end="")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
