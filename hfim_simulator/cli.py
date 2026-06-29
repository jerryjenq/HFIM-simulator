from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .pk import DrugConfig, FosfomycinConfig, SystemConfig, flow_for_half_life, simulate_hfim, solve_css_cmax_replacement
from .store import SimulationStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HFIM three-drug PK simulator.")
    parser.add_argument("--scenario", choices=["q24_replacement", "overflow"], default="q24_replacement")
    parser.add_argument("--duration-h", type=float, default=168)
    parser.add_argument("--dt-min", type=float, default=1)
    parser.add_argument("--extra-volume-ml", type=float, default=241)
    parser.add_argument(
        "--q-extra-to-central-ml-min",
        type=float,
        default=None,
        help="Physical extra-to-central transfer rate. In q24 replacement mode, default is 0.167 mL/min and is not auto-scaled by extra volume.",
    )
    parser.add_argument("--fos-target-css-mg-l", type=float, default=150)
    parser.add_argument("--fos-target-cmax-mg-l", type=float, default=250)
    parser.add_argument("--fos-dose-volume-ml", type=float, default=6)
    parser.add_argument("--fos-infusion-duration-h", type=float, default=1)
    parser.add_argument("--fos-dosing-frequency-h", type=float, default=6)
    parser.add_argument("--extra-replacement-interval-h", type=float, default=24)
    parser.add_argument("--imipenem-target-mg-l", type=float, default=9)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--db", default="data/hfim-simulations.sqlite")
    args = parser.parse_args()
    _validate_args(args)

    q_extra_to_central = (
        args.q_extra_to_central_ml_min
        if args.q_extra_to_central_ml_min is not None
        else 0.167 if args.scenario == "q24_replacement" else SystemConfig().q_extra_to_central_ml_min
    )
    shared_central_half_life_h = min(3.0, 1.25, 1.25)
    base_system = SystemConfig(extra_volume_ml=args.extra_volume_ml)
    shared_central_outflow = flow_for_half_life(base_system.central_volume_ml, shared_central_half_life_h)
    q_central_diluent = max(0.0, shared_central_outflow - q_extra_to_central)
    system = SystemConfig(
        extra_volume_ml=args.extra_volume_ml,
        q_extra_to_central_ml_min=q_extra_to_central,
        q_extra_diluent_ml_min=0 if args.scenario == "q24_replacement" else SystemConfig().q_extra_diluent_ml_min,
        q_central_diluent_ml_min=q_central_diluent,
    )
    fos_duration_min = int(round(args.fos_infusion_duration_h * 60))
    fos_interval_min = int(round(args.fos_dosing_frequency_h * 60))
    if args.scenario == "q24_replacement":
        solver = solve_css_cmax_replacement(
            system=system,
            drug_name="fosfomycin",
            target_css_mg_l=args.fos_target_css_mg_l,
            target_cmax_mg_l=args.fos_target_cmax_mg_l,
            central_infusion_ml_min=args.fos_dose_volume_ml / fos_duration_min,
            infusion_duration_min=fos_duration_min,
            dosing_interval_min=fos_interval_min,
            replacement_interval_h=args.extra_replacement_interval_h,
            duration_h=args.duration_h,
            dt_min=args.dt_min,
        )
        fos = FosfomycinConfig(
            central_stock_mg_ml=solver.central_stock_mg_ml,
            extra_stock_mg_ml=solver.extra_replacement_concentration_mg_ml,
            central_infusion_ml_min=args.fos_dose_volume_ml / fos_duration_min,
            extra_infusion_ml_min=0,
            infusion_duration_min=fos_duration_min,
            dosing_interval_min=fos_interval_min,
            preload_extra_mg=0,
            reservoir_replacement_interval_h=args.extra_replacement_interval_h,
        )
    else:
        solver = None
        fos = FosfomycinConfig(
            central_infusion_ml_min=args.fos_dose_volume_ml / fos_duration_min,
            infusion_duration_min=fos_duration_min,
            dosing_interval_min=fos_interval_min,
        )
    drugs = [
        DrugConfig(
            "imipenem",
            args.imipenem_target_mg_l,
            1.25,
            loading_target_concentration_mg_l=args.imipenem_target_mg_l * 2,
            loading_duration_h=0.5,
        ),
        DrugConfig(
            "relebactam",
            args.imipenem_target_mg_l * 2 / 3,
            1.25,
            loading_target_concentration_mg_l=args.imipenem_target_mg_l * 4 / 3,
            loading_duration_h=0.5,
        ),
    ]
    result = simulate_hfim(args.scenario, system, fos, drugs, args.duration_h, args.dt_min)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.scenario}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    csv_path = output_dir / f"{prefix}-timecourse.csv"
    summary_path = output_dir / f"{prefix}-summary.json"
    _write_csv(csv_path, result.rows)
    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf8")

    store = SimulationStore(args.db)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = store.create_run(args.scenario, started_at, vars(args))
    counts = store.upsert_timepoints(run_id, [
        {
            "time_min": row["time_min"],
            "drug": row["drug"],
            "central": row["central_mg_l"],
            "extra": row["extra_mg_l"],
            "central_volume_ml": row["central_volume_ml"],
            "extra_volume_ml": row["extra_volume_ml"],
        }
        for row in result.rows
    ])
    prep_counts = store.upsert_preparation_rows(run_id, result.summary["drug_preparation"])
    store.finish_run(
        run_id,
        "success",
        datetime.now(timezone.utc).isoformat(),
        f"timepoints={counts}; preparation={prep_counts}",
    )

    print(f"Scenario: {args.scenario}")
    print(f"Run ID: {run_id}")
    print(f"Time-course CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"SQLite DB: {args.db}")
    print(f"FOS central AUC0-24: {result.summary['fosfomycin']['central_auc_0_24_mg_h_l']:.2f} mg*h/L")
    print(f"FOS central Cavg: {result.summary['fosfomycin']['central_cavg_0_24_mg_l']:.2f} mg/L")
    print(f"FOS central Cmax: {result.summary['fosfomycin']['central_cmax_mg_l']:.2f} mg/L")
    print(f"FOS central Cmin after 24 h: {result.summary['fosfomycin']['central_cmin_after_24h_mg_l']:.2f} mg/L")
    if solver is not None:
        print(f"FOS solved central stock: {solver.central_stock_mg_ml:.6f} mg/mL")
        print(f"FOS solved extra replacement concentration: {solver.extra_replacement_concentration_mg_ml:.6f} mg/mL")
        print(f"Solver status: {solver.message}")
    print("Drug preparation:")
    for row in result.summary["drug_preparation"]:
        amount = row["amount_mg"]
        daily = row["daily_amount_mg"]
        daily_text = "" if daily is None else f", daily {daily:.3f} mg"
        print(f"- {row['drug']} {row['component']}: {amount:.3f} mg{daily_text}")

def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "time_min",
        "time_h",
        "drug",
        "central_mg_l",
        "extra_mg_l",
        "central_volume_ml",
        "extra_volume_ml",
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace) -> None:
    if args.duration_h < 24:
        raise SystemExit("duration-h must be at least 24 because the simulator reports AUC0-24 and Cavg0-24.")
    if args.fos_infusion_duration_h <= 0:
        raise SystemExit("fos-infusion-duration-h must be positive.")


if __name__ == "__main__":
    main()
