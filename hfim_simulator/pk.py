from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SystemConfig:
    central_bottle_ml: float = 100
    cartridge_ml: float = 70
    extra_volume_ml: float = 241
    q_extra_to_central_ml_min: float = 0.921
    q_extra_diluent_ml_min: float = 0.921
    q_central_diluent_ml_min: float = 0.65

    @property
    def central_volume_ml(self) -> float:
        return self.central_bottle_ml + self.cartridge_ml

    @property
    def q_waste_ml_min(self) -> float:
        return self.q_extra_to_central_ml_min + self.q_central_diluent_ml_min


@dataclass(frozen=True)
class FosfomycinConfig:
    central_stock_mg_ml: float = 5.897897
    extra_stock_mg_ml: float = 8.351422
    central_infusion_ml_min: float = 0.1
    extra_infusion_ml_min: float = 0.1
    infusion_duration_min: int = 60
    dosing_interval_min: int = 360
    preload_extra_mg: float | None = None
    drug_name: str = "fosfomycin"
    reservoir_replacement_interval_h: float = 24.0

    @property
    def central_dose_mg(self) -> float:
        return self.central_stock_mg_ml * self.central_infusion_ml_min * self.infusion_duration_min

    @property
    def extra_dose_mg(self) -> float:
        return self.extra_stock_mg_ml * self.extra_infusion_ml_min * self.infusion_duration_min


@dataclass(frozen=True)
class DrugConfig:
    name: str
    target_concentration_mg_l: float
    half_life_h: float
    dosing_mode: str = "loading dose + continuous infusion"
    loading_target_concentration_mg_l: float | None = None
    loading_duration_h: float = 0.0
    loading_volume_ml: float = 5.0
    intermittent_interval_h: float = 6.0
    intermittent_duration_h: float = 1.0


@dataclass(frozen=True)
class ContinuousInfusionRegimen:
    target_concentration_mg_l: float
    half_life_h: float
    central_volume_ml: float
    elimination_flow_ml_min: float
    loading_target_concentration_mg_l: float
    loading_duration_h: float
    loading_volume_ml: float
    intermittent_interval_h: float
    intermittent_duration_h: float
    loading_dose_mg: float
    loading_concentration_mg_ml: float
    loading_infusion_rate_ml_h: float
    loading_infusion_rate_ml_min: float
    intermittent_dose_mg: float
    infusion_rate_mg_h: float
    daily_amount_mg: float


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    rows: list[dict]
    summary: dict


@dataclass(frozen=True)
class CssCmaxSolverResult:
    central_stock_mg_ml: float
    extra_replacement_concentration_mg_ml: float
    target_css_mg_l: float
    target_auc_0_24_mg_h_l: float
    target_cmax_mg_l: float
    predicted_auc_0_24_mg_h_l: float
    predicted_cavg_mg_l: float
    predicted_cmax_mg_l: float
    predicted_cmin_mg_l: float
    central_only_auc_0_24_mg_h_l: float
    extra_only_auc_0_24_mg_h_l: float
    cmax_error_mg_l: float
    feasible: bool
    message: str


def compute_continuous_infusion(
    target_concentration_mg_l: float,
    half_life_h: float,
    central_volume_ml: float,
    loading_target_concentration_mg_l: float | None = None,
    loading_duration_h: float = 0.0,
    loading_volume_ml: float = 5.0,
    intermittent_interval_h: float = 6.0,
    intermittent_duration_h: float = 1.0,
    shared_elimination_flow_ml_min: float | None = None,
) -> ContinuousInfusionRegimen:
    if shared_elimination_flow_ml_min is None:
        half_life_min = half_life_h * 60
        elimination_flow_ml_min = math.log(2) * central_volume_ml / half_life_min
    else:
        elimination_flow_ml_min = shared_elimination_flow_ml_min
        half_life_h = half_life_for_flow(central_volume_ml, shared_elimination_flow_ml_min)
    target_mg_ml = target_concentration_mg_l / 1000
    loading_target = loading_target_concentration_mg_l if loading_target_concentration_mg_l is not None else target_concentration_mg_l
    loading_target_mg_ml = loading_target / 1000
    infusion_rate_mg_min = target_mg_ml * elimination_flow_ml_min
    infusion_rate_mg_h = infusion_rate_mg_min * 60
    loading_dose_mg = loading_target_mg_ml * central_volume_ml
    loading_concentration_mg_ml = loading_dose_mg / loading_volume_ml if loading_volume_ml > 0 else 0.0
    loading_infusion_rate_ml_h = loading_volume_ml / loading_duration_h if loading_duration_h > 0 else 0.0
    loading_infusion_rate_ml_min = loading_volume_ml / (loading_duration_h * 60) if loading_duration_h > 0 else 0.0
    intermittent_dose_mg = target_mg_ml * central_volume_ml
    return ContinuousInfusionRegimen(
        target_concentration_mg_l=target_concentration_mg_l,
        half_life_h=half_life_h,
        central_volume_ml=central_volume_ml,
        elimination_flow_ml_min=elimination_flow_ml_min,
        loading_target_concentration_mg_l=loading_target,
        loading_duration_h=loading_duration_h,
        loading_volume_ml=loading_volume_ml,
        intermittent_interval_h=intermittent_interval_h,
        intermittent_duration_h=intermittent_duration_h,
        loading_dose_mg=loading_dose_mg,
        loading_concentration_mg_ml=loading_concentration_mg_ml,
        loading_infusion_rate_ml_h=loading_infusion_rate_ml_h,
        loading_infusion_rate_ml_min=loading_infusion_rate_ml_min,
        intermittent_dose_mg=intermittent_dose_mg,
        infusion_rate_mg_h=infusion_rate_mg_h,
        daily_amount_mg=infusion_rate_mg_h * 24,
    )


def flow_for_half_life(volume_ml: float, half_life_h: float) -> float:
    if volume_ml <= 0 or half_life_h <= 0:
        raise ValueError("volume_ml and half_life_h must be positive")
    return math.log(2) * volume_ml / (half_life_h * 60)


def half_life_for_flow(volume_ml: float, flow_ml_min: float) -> float:
    if volume_ml <= 0 or flow_ml_min <= 0:
        raise ValueError("volume_ml and flow_ml_min must be positive")
    return math.log(2) * volume_ml / flow_ml_min / 60


def simulate_hfim(
    scenario: str,
    system: SystemConfig,
    fos: FosfomycinConfig,
    drugs: list[DrugConfig],
    duration_h: float = 168,
    dt_min: float = 1,
) -> SimulationResult:
    if scenario not in {"q24_replacement", "overflow"}:
        raise ValueError("scenario must be 'q24_replacement' or 'overflow'")
    if dt_min <= 0:
        raise ValueError("dt_min must be positive")

    steps = int(round(duration_h * 60 / dt_min))
    vc = system.central_volume_ml
    ve = system.extra_volume_ml
    a_central = 0.0
    a_extra = 0.0
    extra_q6h_dose_count = 0
    overflow_loss_mg = 0.0

    continuous = {
        drug.name: compute_continuous_infusion(
            drug.target_concentration_mg_l,
            drug.half_life_h,
            system.central_volume_ml,
            loading_target_concentration_mg_l=drug.loading_target_concentration_mg_l,
            loading_duration_h=drug.loading_duration_h,
            loading_volume_ml=drug.loading_volume_ml,
            intermittent_interval_h=drug.intermittent_interval_h,
            intermittent_duration_h=drug.intermittent_duration_h,
            shared_elimination_flow_ml_min=system.q_waste_ml_min,
        )
        for drug in drugs
    }
    drug_configs = {drug.name: drug for drug in drugs}
    drug_amounts = {
        name: _initial_amount_for_mode(regimen, drug_configs[name].dosing_mode)
        for name, regimen in continuous.items()
    }

    rows = []
    for step in range(steps + 1):
        time_min = step * dt_min
        if scenario == "q24_replacement" and _is_replacement_time(time_min, fos.reservoir_replacement_interval_h):
            a_extra = fos.extra_stock_mg_ml * ve
        c_central_mg_l = a_central / vc * 1000
        c_extra_mg_l = a_extra / ve * 1000
        row = {
            "time_min": time_min,
            "time_h": time_min / 60,
            "drug": fos.drug_name,
            "central_mg_l": c_central_mg_l,
            "extra_mg_l": c_extra_mg_l,
            "central_volume_ml": vc,
            "extra_volume_ml": ve,
        }
        rows.append(row)
        for name, amount in drug_amounts.items():
            rows.append({
                "time_min": time_min,
                "time_h": time_min / 60,
                "drug": name,
                "central_mg_l": amount / vc * 1000,
                "extra_mg_l": 0.0,
                "central_volume_ml": vc,
                "extra_volume_ml": ve,
            })

        if step == steps:
            break

        q_fos_central = _q6h_rate(time_min, fos.central_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min)
        q_fos_extra = 0.0
        if scenario == "overflow":
            q_fos_extra = _q6h_rate(time_min, fos.extra_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min)
            if q_fos_extra > 0 and _is_dose_start(time_min, fos.dosing_interval_min):
                extra_q6h_dose_count += 1
        c_central_mg_ml = a_central / vc
        c_extra_mg_ml = a_extra / ve
        central_input_mg_min = q_fos_central * fos.central_stock_mg_ml + system.q_extra_to_central_ml_min * c_extra_mg_ml
        central_output_mg_min = (system.q_waste_ml_min + q_fos_central) * c_central_mg_ml
        extra_input_mg_min = q_fos_extra * fos.extra_stock_mg_ml
        extra_to_central_mg_min = system.q_extra_to_central_ml_min * c_extra_mg_ml
        extra_overflow_mg_min = q_fos_extra * c_extra_mg_ml if scenario == "overflow" else 0.0
        extra_delta_mg_min = (
            0.0
            if scenario == "q24_replacement"
            else extra_input_mg_min - extra_to_central_mg_min - extra_overflow_mg_min
        )

        a_central = max(0.0, a_central + (central_input_mg_min - central_output_mg_min) * dt_min)
        a_extra = max(0.0, a_extra + extra_delta_mg_min * dt_min)
        overflow_loss_mg += extra_overflow_mg_min * dt_min

        for name, regimen in continuous.items():
            amount = drug_amounts[name]
            concentration_mg_ml = amount / vc
            input_mg_min = _input_rate_for_mode(time_min, regimen, drug_configs[name].dosing_mode)
            output_mg_min = system.q_waste_ml_min * concentration_mg_ml
            drug_amounts[name] = max(0.0, amount + (input_mg_min - output_mg_min) * dt_min)

    summary = {
        fos.drug_name: _summarize_intermit(rows, fos.drug_name, overflow_loss_mg, extra_q6h_dose_count),
        "drug_preparation": _preparation_table(system, fos, continuous, drug_configs, scenario, duration_h),
    }
    for name, regimen in continuous.items():
        diluent_formulation = _central_diluent_formulation(system, regimen, drug_configs[name].dosing_mode)
        summary[name] = {
            "target_concentration_mg_l": regimen.target_concentration_mg_l,
            "loading_target_concentration_mg_l": regimen.loading_target_concentration_mg_l,
            "loading_duration_h": regimen.loading_duration_h,
            "loading_volume_ml": regimen.loading_volume_ml,
            "loading_dose_mg": regimen.loading_dose_mg,
            "loading_concentration_mg_ml": regimen.loading_concentration_mg_ml,
            "loading_infusion_rate_ml_h": regimen.loading_infusion_rate_ml_h,
            "loading_infusion_rate_ml_min": regimen.loading_infusion_rate_ml_min,
            "intermittent_dose_mg": regimen.intermittent_dose_mg,
            "intermittent_interval_h": regimen.intermittent_interval_h,
            "intermittent_duration_h": regimen.intermittent_duration_h,
            "infusion_rate_mg_h": regimen.infusion_rate_mg_h,
            "daily_amount_mg": regimen.daily_amount_mg,
            "elimination_flow_ml_min": regimen.elimination_flow_ml_min,
            "dosing_mode": drug_configs[name].dosing_mode,
            **diluent_formulation,
        }
    return SimulationResult(scenario=scenario, rows=rows, summary=summary)


def solve_css_cmax_replacement(
    system: SystemConfig,
    drug_name: str,
    target_css_mg_l: float,
    target_cmax_mg_l: float,
    central_infusion_ml_min: float,
    infusion_duration_min: int,
    dosing_interval_min: int,
    replacement_interval_h: float = 24.0,
    duration_h: float = 168,
    dt_min: float = 1,
) -> CssCmaxSolverResult:
    target_auc = target_css_mg_l * 24
    central_basis = FosfomycinConfig(
        drug_name=drug_name,
        central_stock_mg_ml=1.0,
        extra_stock_mg_ml=0.0,
        central_infusion_ml_min=central_infusion_ml_min,
        extra_infusion_ml_min=0.0,
        infusion_duration_min=infusion_duration_min,
        dosing_interval_min=dosing_interval_min,
        preload_extra_mg=0.0,
        reservoir_replacement_interval_h=replacement_interval_h,
    )
    extra_basis = FosfomycinConfig(
        drug_name=drug_name,
        central_stock_mg_ml=0.0,
        extra_stock_mg_ml=1.0,
        central_infusion_ml_min=central_infusion_ml_min,
        extra_infusion_ml_min=0.0,
        infusion_duration_min=infusion_duration_min,
        dosing_interval_min=dosing_interval_min,
        preload_extra_mg=0.0,
        reservoir_replacement_interval_h=replacement_interval_h,
    )
    central_result = simulate_hfim("q24_replacement", system, central_basis, [], duration_h=max(24, duration_h), dt_min=dt_min)
    extra_result = simulate_hfim("q24_replacement", system, extra_basis, [], duration_h=max(24, duration_h), dt_min=dt_min)
    central_rows = [row for row in central_result.rows if row["drug"] == drug_name]
    extra_rows = [row for row in extra_result.rows if row["drug"] == drug_name]
    central_profile = [row["central_mg_l"] for row in central_rows]
    extra_profile = [row["central_mg_l"] for row in extra_rows]
    cmin_indices = [
        index
        for index, row in enumerate(central_rows)
        if row["time_h"] >= 24
    ] or [
        index
        for index, row in enumerate(central_rows)
        if row["time_h"] > 0
    ]
    central_auc = central_result.summary[drug_name]["central_auc_0_24_mg_h_l"]
    extra_auc = extra_result.summary[drug_name]["central_auc_0_24_mg_h_l"]
    if target_auc <= 0 or target_cmax_mg_l <= 0 or central_auc <= 0:
        return CssCmaxSolverResult(
            0.0,
            0.0,
            target_css_mg_l,
            target_auc,
            target_cmax_mg_l,
            0.0,
            0.0,
            0.0,
            0.0,
            central_auc,
            extra_auc,
            target_cmax_mg_l,
            False,
            "Targets must be positive and the central dosing basis must produce exposure.",
        )

    central_stock_for_auc = target_auc / central_auc
    if extra_auc <= 0:
        combined = [central_stock_for_auc * c_value for c_value in central_profile]
        cmax = max(combined)
        cmin = min(combined[index] for index in cmin_indices) if cmin_indices else min(combined)
        cmax_error = cmax - target_cmax_mg_l
        feasible = abs(cmax_error) <= max(5.0, target_cmax_mg_l * 0.05)
        message = (
            "Solver used a central-only solution because the extra compartment does not contribute exposure."
            if feasible
            else "Solver used a central-only solution because the extra compartment does not contribute exposure, but Cmax is not close to the target."
        )
        return CssCmaxSolverResult(
            central_stock_mg_ml=central_stock_for_auc,
            extra_replacement_concentration_mg_ml=0.0,
            target_css_mg_l=target_css_mg_l,
            target_auc_0_24_mg_h_l=target_auc,
            target_cmax_mg_l=target_cmax_mg_l,
            predicted_auc_0_24_mg_h_l=target_auc,
            predicted_cavg_mg_l=target_css_mg_l,
            predicted_cmax_mg_l=cmax,
            predicted_cmin_mg_l=cmin,
            central_only_auc_0_24_mg_h_l=central_auc,
            extra_only_auc_0_24_mg_h_l=extra_auc,
            cmax_error_mg_l=cmax_error,
            feasible=feasible,
            message=message,
        )

    central_upper = max(central_stock_for_auc * 1.2, 1.0)
    best = None
    grid_points = 501
    for index in range(grid_points):
        central_stock = central_upper * index / (grid_points - 1)
        remaining_auc = target_auc - central_stock * central_auc
        if remaining_auc < -1e-9:
            continue
        if extra_auc > 0:
            extra_stock = max(0.0, remaining_auc / extra_auc)
        elif remaining_auc <= 1e-9:
            extra_stock = 0.0
        else:
            continue
        combined = [
            central_stock * c_value + extra_stock * e_value
            for c_value, e_value in zip(central_profile, extra_profile)
        ]
        cmax = max(combined)
        cmin = min(combined[index] for index in cmin_indices) if cmin_indices else min(combined)
        auc = central_stock * central_auc + extra_stock * extra_auc
        cavg = auc / 24
        score = abs(cmax - target_cmax_mg_l) + abs(cavg - target_css_mg_l) * 0.25
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "central_stock": central_stock,
                "extra_stock": extra_stock,
                "auc": auc,
                "cavg": cavg,
                "cmax": cmax,
                "cmin": cmin,
            }

    if best is None:
        return CssCmaxSolverResult(
            0.0,
            0.0,
            target_css_mg_l,
            target_auc,
            target_cmax_mg_l,
            0.0,
            0.0,
            0.0,
            0.0,
            central_auc,
            extra_auc,
            target_cmax_mg_l,
            False,
            "No non-negative central/extra concentration combination could reach the target AUC.",
        )

    cmax_error = best["cmax"] - target_cmax_mg_l
    feasible = abs(cmax_error) <= max(5.0, target_cmax_mg_l * 0.05)
    message = (
        "Solver found a profile close to both Css/Cavg and Cmax targets."
        if feasible
        else "Solver reached the Css/Cavg target but could not closely match Cmax with the current flow, duration, and interval settings."
    )
    return CssCmaxSolverResult(
        central_stock_mg_ml=best["central_stock"],
        extra_replacement_concentration_mg_ml=best["extra_stock"],
        target_css_mg_l=target_css_mg_l,
        target_auc_0_24_mg_h_l=target_auc,
        target_cmax_mg_l=target_cmax_mg_l,
        predicted_auc_0_24_mg_h_l=best["auc"],
        predicted_cavg_mg_l=best["cavg"],
        predicted_cmax_mg_l=best["cmax"],
        predicted_cmin_mg_l=best["cmin"],
        central_only_auc_0_24_mg_h_l=central_auc,
        extra_only_auc_0_24_mg_h_l=extra_auc,
        cmax_error_mg_l=cmax_error,
        feasible=feasible,
        message=message,
    )


def _q6h_rate(time_min: float, rate_ml_min: float, duration_min: int, interval_min: int) -> float:
    return rate_ml_min if time_min % interval_min < duration_min else 0.0


def _is_dose_start(time_min: float, interval_min: int) -> bool:
    return abs(time_min % interval_min) < 1e-9


def _is_replacement_time(time_min: float, interval_h: float) -> bool:
    interval_min = interval_h * 60
    return interval_min > 0 and abs(time_min % interval_min) < 1e-9


def _initial_amount_for_mode(regimen: ContinuousInfusionRegimen, dosing_mode: str) -> float:
    has_loading = dosing_mode in {
        "loading dose + continuous infusion",
        "loading dose + intermittent infusion",
        "loading dose only",
    }
    if has_loading and regimen.loading_duration_h <= 0:
        return regimen.loading_dose_mg
    return 0.0


def _input_rate_for_mode(time_min: float, regimen: ContinuousInfusionRegimen, dosing_mode: str) -> float:
    return _loading_rate_for_mode(time_min, regimen, dosing_mode) + _maintenance_rate_for_mode(time_min, regimen, dosing_mode)


def _loading_rate_for_mode(time_min: float, regimen: ContinuousInfusionRegimen, dosing_mode: str) -> float:
    has_loading = dosing_mode in {
        "loading dose + continuous infusion",
        "loading dose + intermittent infusion",
        "loading dose only",
    }
    if not has_loading or regimen.loading_duration_h <= 0:
        return 0.0
    loading_duration_min = regimen.loading_duration_h * 60
    if time_min < loading_duration_min:
        return regimen.loading_dose_mg / loading_duration_min
    return 0.0


def _maintenance_rate_for_mode(time_min: float, regimen: ContinuousInfusionRegimen, dosing_mode: str) -> float:
    if dosing_mode in {"loading dose + continuous infusion", "continuous infusion only"}:
        return regimen.infusion_rate_mg_h / 60
    if dosing_mode in {"loading dose + intermittent infusion", "intermittent infusion only"}:
        duration_min = regimen.intermittent_duration_h * 60
        interval_min = regimen.intermittent_interval_h * 60
        if duration_min > 0 and interval_min > 0 and time_min % interval_min < duration_min:
            return regimen.intermittent_dose_mg / duration_min
    return 0.0


def _central_diluent_formulation(
    system: SystemConfig,
    regimen: ContinuousInfusionRegimen,
    dosing_mode: str,
) -> dict:
    is_continuous = dosing_mode in {"loading dose + continuous infusion", "continuous infusion only"}
    daily_volume_ml = system.q_central_diluent_ml_min * 24 * 60
    if not is_continuous:
        return {
            "central_diluent_concentration_mg_ml": None,
            "central_diluent_volume_per_24h_ml": daily_volume_ml,
            "central_diluent_drug_per_24h_mg": 0.0,
            "central_diluent_feasible": True,
        }
    if system.q_central_diluent_ml_min <= 0:
        return {
            "central_diluent_concentration_mg_ml": None,
            "central_diluent_volume_per_24h_ml": daily_volume_ml,
            "central_diluent_drug_per_24h_mg": regimen.daily_amount_mg,
            "central_diluent_feasible": False,
        }
    concentration_mg_ml = (regimen.infusion_rate_mg_h / 60) / system.q_central_diluent_ml_min
    return {
        "central_diluent_concentration_mg_ml": concentration_mg_ml,
        "central_diluent_volume_per_24h_ml": daily_volume_ml,
        "central_diluent_drug_per_24h_mg": regimen.daily_amount_mg,
        "central_diluent_feasible": True,
    }


def _summarize_intermit(rows: list[dict], drug_name: str, overflow_loss_mg: float, extra_q6h_dose_count: int) -> dict:
    drug_rows = [row for row in rows if row["drug"] == drug_name]
    central = [row["central_mg_l"] for row in drug_rows]
    extra = [row["extra_mg_l"] for row in drug_rows]
    return {
        "central_cmax_mg_l": max(central),
        "central_cmin_mg_l": min(central),
        "central_cmin_after_24h_mg_l": min(
            row["central_mg_l"]
            for row in drug_rows
            if row["time_h"] >= 24
        ) if any(row["time_h"] >= 24 for row in drug_rows) else min(central),
        "central_cavg_0_24_mg_l": _auc(drug_rows, "central_mg_l", 24) / 24,
        "extra_cmax_mg_l": max(extra),
        "extra_cmin_mg_l": min(extra),
        "central_auc_0_24_mg_h_l": _auc(drug_rows, "central_mg_l", 24),
        "extra_auc_0_24_mg_h_l": _auc(drug_rows, "extra_mg_l", 24),
        "central_auc_full_mg_h_l": _auc(drug_rows, "central_mg_l", None),
        "extra_auc_full_mg_h_l": _auc(drug_rows, "extra_mg_l", None),
        "final_central_volume_ml": drug_rows[-1]["central_volume_ml"],
        "final_extra_volume_ml": drug_rows[-1]["extra_volume_ml"],
        "overflow_loss_mg": overflow_loss_mg,
        "extra_q6h_dose_count": extra_q6h_dose_count,
    }


def _auc(rows: list[dict], column: str, until_h: float | None) -> float:
    total = 0.0
    usable = [row for row in rows if until_h is None or row["time_h"] <= until_h]
    for prev, curr in zip(usable, usable[1:]):
        dt_h = curr["time_h"] - prev["time_h"]
        total += (prev[column] + curr[column]) * 0.5 * dt_h
    return total


def _preparation_table(
    system: SystemConfig,
    fos: FosfomycinConfig,
    continuous: dict[str, ContinuousInfusionRegimen],
    drug_configs: dict[str, DrugConfig],
    scenario: str,
    duration_h: float,
) -> list[dict]:
    table = [
        {
            "drug": fos.drug_name,
            "component": f"central q{fos.dosing_interval_min / 60:g}h infusion",
            "amount_mg": fos.central_dose_mg,
            "daily_amount_mg": fos.central_dose_mg * 24 / (fos.dosing_interval_min / 60),
            "note": f"{fos.central_infusion_ml_min * fos.infusion_duration_min:g} mL over {fos.infusion_duration_min / 60:g} h",
        }
    ]
    if scenario == "overflow":
        table.append({
            "drug": fos.drug_name,
            "component": f"extra q{fos.dosing_interval_min / 60:g}h infusion",
            "amount_mg": fos.extra_dose_mg,
            "daily_amount_mg": fos.extra_dose_mg * 24 / (fos.dosing_interval_min / 60),
            "note": "extra volume is controlled by overflow",
        })
    elif scenario == "q24_replacement":
        replacement_volume_ml = system.extra_volume_ml
        replacement_amount_mg = fos.extra_stock_mg_ml * replacement_volume_ml
        transfer_volume_ml = system.q_extra_to_central_ml_min * fos.reservoir_replacement_interval_h * 60
        transfer_amount_mg = fos.extra_stock_mg_ml * transfer_volume_ml
        replacements = max(1, math.ceil(duration_h / fos.reservoir_replacement_interval_h))
        table.append({
            "drug": fos.drug_name,
            "component": f"extra q{fos.reservoir_replacement_interval_h:g}h fixed-concentration solution",
            "amount_mg": replacement_amount_mg,
            "daily_amount_mg": replacement_amount_mg * 24 / fos.reservoir_replacement_interval_h,
            "note": (
                f"pressure/filter-controlled q24h replacement; "
                f"prepare {replacement_volume_ml:g} mL at {fos.extra_stock_mg_ml:g} mg/mL per interval; "
                f"Qextra transfer is modeled as drug leaving this same fill "
                f"({transfer_volume_ml:.1f} mL, {transfer_amount_mg:.1f} mg per interval), not as extra prepared solution; "
                f"+10% volume {replacement_volume_ml * 1.10:.1f} mL; "
                f"{duration_h:g} h uses {replacements:g} intervals = {replacement_amount_mg * replacements:.1f} mg"
            ),
        })
    for name, regimen in continuous.items():
        dosing_mode = drug_configs[name].dosing_mode
        if dosing_mode in {"loading dose + continuous infusion", "loading dose + intermittent infusion", "loading dose only"}:
            table.append({
                "drug": name,
                "component": "loading dose",
                "amount_mg": regimen.loading_dose_mg,
                "daily_amount_mg": None,
                "note": (
                    f"target {regimen.loading_target_concentration_mg_l:g} mg/L; "
                    f"dissolve in {regimen.loading_volume_ml:g} mL "
                    f"({regimen.loading_concentration_mg_ml:g} mg/mL); "
                    f"infuse over {regimen.loading_duration_h:g} h "
                    f"at {regimen.loading_infusion_rate_ml_h:g} mL/h"
                ),
            })
        if dosing_mode in {"loading dose + continuous infusion", "continuous infusion only"}:
            diluent_note = "amount is mg/h; maintenance CI is mixed into the central diluent reservoir"
            concentration = _central_diluent_formulation(system, regimen, dosing_mode)["central_diluent_concentration_mg_ml"]
            if concentration is not None:
                diluent_note += f" at {concentration:g} mg/mL and replaced q24h"
            table.append({
                "drug": name,
                "component": "continuous infusion",
                "amount_mg": regimen.infusion_rate_mg_h,
                "daily_amount_mg": regimen.daily_amount_mg,
                "note": diluent_note,
            })
        if dosing_mode in {"loading dose + intermittent infusion", "intermittent infusion only"}:
            table.append({
                "drug": name,
                "component": f"intermittent q{regimen.intermittent_interval_h:g}h infusion",
                "amount_mg": regimen.intermittent_dose_mg,
                "daily_amount_mg": regimen.intermittent_dose_mg * 24 / regimen.intermittent_interval_h,
                "note": f"{regimen.intermittent_duration_h:g} h infusion",
            })
    return table
