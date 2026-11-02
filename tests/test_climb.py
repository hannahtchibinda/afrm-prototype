from afrm.config import load_aircraft, load_criteria
from afrm.models import Checkpoint, FlightPlan
from afrm.monitoring import FlightMonitor
from afrm.performance import cruise_exponent, fuel_from_start_weight
from afrm.planning import compute_fuel_plan

GNB = load_aircraft()["GNB"]
CRITERIA = load_criteria()
FLIGHT = FlightPlan("T001", "GNB", "A", "B", "C", 1000.0, 150.0, 15000.0, 10000.0, 0.0)
REAL_PLAN = compute_fuel_plan(GNB, FLIGHT, CRITERIA.planning)
ZERO_FUEL = 57000.0


def monitor() -> FlightMonitor:
    return FlightMonitor(GNB, FLIGHT, REAL_PLAN, CRITERIA.inflight)


def test_climb_is_shared_across_short_segments() -> None:
    m = monitor()
    fob = 7000.0
    results = []
    for seq in range(7):
        distance = seq * 25.0
        results.append(m.update(Checkpoint("T001", seq, seq * 0.1, distance, fob, 0.0)))
        air = 50.0 if distance < 100 else 25.0  # The first 100 NM count double.
        exponent = cruise_exponent(air, GNB.cruise_tas_kt, GNB.cruise_ld, GNB.cruise_tsfc_per_h)
        fob -= fuel_from_start_weight(ZERO_FUEL + fob, exponent)
    assert all(r.data_valid and r.burn_ratio == 1.0 for r in results)
    efobs = [r.efob_destination_kg for r in results]
    assert max(efobs) - min(efobs) < 1.0
