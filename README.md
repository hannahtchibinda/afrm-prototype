# AFRM: Aircraft Fuel and Range Management prototype

Plans the amount of fuel a flight needs, assesses whether the flight can go ahead, 
and then monitors its fuel and range against defined criteria as telemetry arrives.

> **Not for operational use.** All flight data is simulated, and calculations are simplified in context.
>  every assumption is located at [docs/criteria.md](docs/criteria.md) .

## Requirements

- Python 3.13 or later. CI runs on Python 3.13 and 3.14, on Windows and Linux.
- No runtime dependencies. Development tools are pinned by hash in `requirements-dev.lock`.

## Setup

Windows Command Prompt:

```cmd
py -3.14 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps -e .
```

On Windows, PowerShell can block `Activate.ps1` under its execution policy. You can avoid this by using Command Prompt

## Use

```cmd
afrm generate --seed 2026 --flights 200
afrm assess --json results.jsonl --detail SIM00001
```

`data/` already holds the dataset for seed 2026. Use `--criteria` or `--aircraft` (before the
subcommand) to replace the built-in TOML files. Exit code 0 means success; 2 means invalid input.

## Input and output

| File | Format | Contents |
|---|---|---|
| `flights.json` | JSON list | Route, distances, payload, planned fuel, forecast wind |
| `telemetry.csv` | CSV, rows of one flight adjacent | `flight_id,seq,elapsed_h,distance_flown_nm,fob_kg,wind_kt` |
| `results.jsonl` | JSON Lines | One record per flight: dispatch result and each checkpoint's status |

Units are kg, NM, kt and hours. Positive wind is a tailwind.

## Structure

| Module | Responsibility |
|---|---|
| `performance.py` | Pure Breguet fuel and range functions |
| `planning.py` | Fuel plan (fixed-point iteration) and dispatch status |
| `monitoring.py` | In-flight status, burn ratio, bad-data handling |
| `pipeline.py` | Joins plans with telemetry and runs each flight |
| `simulation.py` | Seeded generator with its own fuel behaviour, separate from the algorithm's model |
| `dataio.py`, `config.py` | Validated file boundaries |
| `cli.py` | Command-line interface |

Diagrams are in `docs/figures/`.

## Verification

```cmd
python scripts\verify.py
python scripts\evaluate_scenarios.py
python scripts\benchmark.py
```

These run all the quality checks, measure behaviour across 10,000 simulated flights on a held-out
seed, and measure time and memory. Results are kept in `docs/evidence/`. `docs/requirements.md` traces
each requirement to its tests. CI repeats the checks on every push and weekly.

## Known limits

- A falsely low gauge reading cannot be told apart from a leak, and usually raises a false alarm.
- The burn discrepancy flag also fires on about 9 % of healthy simulated flights.
- Dispatch scenarios are built using the same planner that assesses them, so their 100 % results show
  consistency, not accuracy.
- Aircraft figures are generic round numbers and nothing has been checked against real flight data.

