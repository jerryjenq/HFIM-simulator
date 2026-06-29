import sqlite3
import tempfile
import unittest
from pathlib import Path

from hfim_simulator.store import SimulationStore


class SimulationStoreTest(unittest.TestCase):
    def test_timepoints_upsert_only_changes_modified_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SimulationStore(Path(tmp) / "sim.sqlite")
            run_id = store.create_run("overflow", "2026-06-25T20:00:00Z", {"scenario": "overflow"})

            rows = [
                {"time_min": 0, "drug": "fosfomycin", "central": 1, "extra": 2},
                {"time_min": 1, "drug": "fosfomycin", "central": 1.1, "extra": 1.9},
            ]
            first = store.upsert_timepoints(run_id, rows)
            second = store.upsert_timepoints(run_id, rows)
            changed = store.upsert_timepoints(
                run_id,
                [
                    {"time_min": 1, "drug": "fosfomycin", "central": 1.2, "extra": 1.9},
                    {"time_min": 2, "drug": "fosfomycin", "central": 1.3, "extra": 1.8},
                ],
            )

            self.assertEqual(first, {"inserted": 2, "updated": 0, "unchanged": 0})
            self.assertEqual(second, {"inserted": 0, "updated": 0, "unchanged": 2})
            self.assertEqual(changed, {"inserted": 1, "updated": 1, "unchanged": 0})

            fetched = sqlite3.connect(store.path).execute(
                "SELECT central_mg_l FROM concentration_timepoints WHERE run_id = ? AND time_min = 1",
                (run_id,),
            ).fetchone()
            self.assertEqual(fetched[0], 1.2)

    def test_refresh_runs_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SimulationStore(Path(tmp) / "sim.sqlite")
            run_id = store.create_run("q24_replacement", "2026-06-25T20:00:00Z", {"scenario": "q24_replacement"})
            store.finish_run(run_id, "success", "2026-06-25T20:01:00Z", "ok")

            row = sqlite3.connect(store.path).execute(
                "SELECT scenario, status, notes FROM simulation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            self.assertEqual(row, ("q24_replacement", "success", "ok"))


if __name__ == "__main__":
    unittest.main()
