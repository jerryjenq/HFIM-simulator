from __future__ import annotations

import json
from pathlib import Path
import sqlite3


class SimulationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def init(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS simulation_runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scenario TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  status TEXT NOT NULL DEFAULT 'running',
                  parameters_json TEXT NOT NULL,
                  notes TEXT
                );
                CREATE TABLE IF NOT EXISTS concentration_timepoints (
                  run_id INTEGER NOT NULL,
                  time_min REAL NOT NULL,
                  drug TEXT NOT NULL,
                  central_mg_l REAL NOT NULL,
                  extra_mg_l REAL NOT NULL,
                  central_volume_ml REAL,
                  extra_volume_ml REAL,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (run_id, time_min, drug),
                  FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
                );
                CREATE TABLE IF NOT EXISTS preparation_rows (
                  run_id INTEGER NOT NULL,
                  drug TEXT NOT NULL,
                  component TEXT NOT NULL,
                  amount_mg REAL,
                  daily_amount_mg REAL,
                  note TEXT,
                  PRIMARY KEY (run_id, drug, component),
                  FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
                );
                """
            )

    def create_run(self, scenario: str, started_at: str, parameters: dict) -> int:
        with self._connect() as con:
            cursor = con.execute(
                """
                INSERT INTO simulation_runs (scenario, started_at, parameters_json)
                VALUES (?, ?, ?)
                """,
                (scenario, started_at, json.dumps(parameters, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, finished_at: str, notes: str = "") -> None:
        with self._connect() as con:
            con.execute(
                """
                UPDATE simulation_runs
                SET status = ?, finished_at = ?, notes = ?
                WHERE id = ?
                """,
                (status, finished_at, notes, run_id),
            )

    def upsert_timepoints(self, run_id: int, rows: list[dict]) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        with self._connect() as con:
            for row in rows:
                existing = con.execute(
                    """
                    SELECT central_mg_l, extra_mg_l, central_volume_ml, extra_volume_ml
                    FROM concentration_timepoints
                    WHERE run_id = ? AND time_min = ? AND drug = ?
                    """,
                    (run_id, row["time_min"], row["drug"]),
                ).fetchone()
                values = (
                    run_id,
                    row["time_min"],
                    row["drug"],
                    row["central"],
                    row["extra"],
                    row.get("central_volume_ml"),
                    row.get("extra_volume_ml"),
                )
                if existing is None:
                    con.execute(
                        """
                        INSERT INTO concentration_timepoints (
                          run_id, time_min, drug, central_mg_l, extra_mg_l, central_volume_ml, extra_volume_ml
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
                    counts["inserted"] += 1
                elif _same(existing, values[3:]):
                    counts["unchanged"] += 1
                else:
                    con.execute(
                        """
                        UPDATE concentration_timepoints
                        SET central_mg_l = ?,
                            extra_mg_l = ?,
                            central_volume_ml = ?,
                            extra_volume_ml = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE run_id = ? AND time_min = ? AND drug = ?
                        """,
                        (
                            row["central"],
                            row["extra"],
                            row.get("central_volume_ml"),
                            row.get("extra_volume_ml"),
                            run_id,
                            row["time_min"],
                            row["drug"],
                        ),
                    )
                    counts["updated"] += 1
        return counts

    def upsert_preparation_rows(self, run_id: int, rows: list[dict]) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        with self._connect() as con:
            for row in rows:
                existing = con.execute(
                    """
                    SELECT amount_mg, daily_amount_mg, note
                    FROM preparation_rows
                    WHERE run_id = ? AND drug = ? AND component = ?
                    """,
                    (run_id, row["drug"], row["component"]),
                ).fetchone()
                next_values = (row.get("amount_mg"), row.get("daily_amount_mg"), row.get("note"))
                if existing is None:
                    con.execute(
                        """
                        INSERT INTO preparation_rows (
                          run_id, drug, component, amount_mg, daily_amount_mg, note
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (run_id, row["drug"], row["component"], *next_values),
                    )
                    counts["inserted"] += 1
                elif _same(existing, next_values):
                    counts["unchanged"] += 1
                else:
                    con.execute(
                        """
                        UPDATE preparation_rows
                        SET amount_mg = ?, daily_amount_mg = ?, note = ?
                        WHERE run_id = ? AND drug = ? AND component = ?
                        """,
                        (*next_values, run_id, row["drug"], row["component"]),
                    )
                    counts["updated"] += 1
        return counts

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA busy_timeout = 30000")
        return con


def _same(existing, incoming) -> bool:
    return tuple(existing) == tuple(incoming)
