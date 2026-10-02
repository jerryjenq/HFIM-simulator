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
    # Optional second central line ("slow line") used by the two-phase curve-shape setup. It starts
    # the moment the main infusion ends and runs for slow_duration_min in every dosing interval.
    # All three default to 0, which means "no slow line" and leaves the classic setup unchanged.
    slow_stock_mg_ml: float = 0.0
    slow_infusion_ml_min: float = 0.0
    slow_duration_min: int = 0

    @property
    def central_dose_mg(self) -> float:
        return self.central_stock_mg_ml * self.central_infusion_ml_min * self.infusion_duration_min

    @property
    def has_slow_line(self) -> bool:
        return self.slow_duration_min > 0 and self.slow_infusion_ml_min > 0 and self.slow_stock_mg_ml > 0

    @property
    def slow_dose_volume_ml(self) -> float:
        return self.slow_infusion_ml_min * self.slow_duration_min if self.has_slow_line else 0.0

    @property
    def slow_dose_mg(self) -> float:
        return self.slow_stock_mg_ml * self.slow_dose_volume_ml

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


@dataclass(frozen=True)
class TwoPhaseSolverResult:
    """Steady-state solution of the two-phase (main infusion + slow line + fixed extra feed) setup.

    All predicted_* values describe one dosing interval at steady state, not the first 24 h.
    max_deviation_pct is the largest relative gap between the achievable central profile and the
    true one-compartment profile at the drug's own (long) half-life, anywhere in the interval.
    """

    central_stock_mg_ml: float
    slow_stock_mg_ml: float
    slow_infusion_ml_min: float
    slow_duration_min: int
    extra_replacement_concentration_mg_ml: float
    target_css_mg_l: float
    reference_half_life_h: float
    predicted_cavg_mg_l: float
    predicted_cmax_mg_l: float
    predicted_cmin_mg_l: float
    reference_cmax_mg_l: float
    reference_cmin_mg_l: float
    max_deviation_pct: float
    single_phase_max_deviation_pct: float
    deviation_limit_pct: float
    feasible: bool
    message: str
    cycle_time_h: tuple
    predicted_profile_mg_l: tuple
    reference_profile_mg_l: tuple


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
    # Dose per interval that delivers the same average mg/min as continuous infusion at steady state
    # (Dose / interval = CL x Css, the standard multiple-dose superposition identity). This makes
    # repeated intermittent dosing converge to the target Css instead of drifting away from it based on
    # how the interval happens to compare with the washout half-life.
    intermittent_dose_mg = infusion_rate_mg_min * (intermittent_interval_h * 60)
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


def intermittent_peak_trough(
    cavg_mg_l: float,
    half_life_h: float,
    interval_h: float,
    infusion_duration_h: float,
) -> tuple[float, float]:
    """Steady-state Cmax/Cmin for repeated IV infusion holding Cavg fixed.

    In a central-only system Cavg is set by dose/interval alone, so once the target Cavg is fixed
    the peak-to-trough shape is fully determined by half-life vs interval vs infusion duration -
    it is not an independent target. This closed form lets the UI show what Cmax/Cmin each candidate
    interval would produce, making the interval visible as the lever that shapes the peak.
    """
    if cavg_mg_l <= 0 or half_life_h <= 0 or interval_h <= 0 or infusion_duration_h <= 0:
        return 0.0, 0.0
    duration = min(infusion_duration_h, interval_h)
    k = math.log(2) / half_life_h
    accumulation = (1 - math.exp(-k * duration)) / (1 - math.exp(-k * interval_h))
    cmax = cavg_mg_l * (interval_h / duration) * accumulation
    cmin = cmax * math.exp(-k * (interval_h - duration))
    return cmax, cmin


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
    fos: FosfomycinConfig | None,
    drugs: list[DrugConfig],
    duration_h: float = 168,
    dt_min: float = 1,
) -> SimulationResult:
    if scenario not in {"q24_replacement", "overflow", "central_only"}:
        raise ValueError("scenario must be 'q24_replacement', 'overflow', or 'central_only'")
    if dt_min <= 0:
        raise ValueError("dt_min must be positive")
    if scenario == "central_only":
        # Matched-half-life setup: no extra compartment and no setup drug at all. Every drug is
        # dosed straight into central and washes out against the shared central flow, so there is
        # no second compartment to reshape any drug's apparent half-life.
        fos = None
    elif fos is None:
        raise ValueError("fos is required unless scenario is 'central_only'")

    steps = int(round(duration_h * 60 / dt_min))
    vc = system.central_volume_ml
    ve = system.extra_volume_ml
    a_central = 0.0
    a_extra = 0.0
    extra_q6h_dose_count = 0
    overflow_loss_mg = 0.0

    setup_central_average_flow_ml_min = (
        average_intermittent_rate(
            fos.central_infusion_ml_min,
            fos.infusion_duration_min,
            fos.dosing_interval_min,
        )
        + (
            average_intermittent_rate(fos.slow_infusion_ml_min, fos.slow_duration_min, fos.dosing_interval_min)
            if fos.has_slow_line
            else 0.0
        )
        if fos is not None
        else 0.0
    )
    shared_continuous_elimination_flow_ml_min = system.q_waste_ml_min + setup_central_average_flow_ml_min

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
            shared_elimination_flow_ml_min=shared_continuous_elimination_flow_ml_min,
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
        if fos is not None and scenario == "q24_replacement" and _is_replacement_time(time_min, fos.reservoir_replacement_interval_h):
            a_extra = fos.extra_stock_mg_ml * ve
        c_central_mg_l = a_central / vc * 1000
        c_extra_mg_l = a_extra / ve * 1000
        if fos is not None:
            rows.append({
                "time_min": time_min,
                "time_h": time_min / 60,
                "drug": fos.drug_name,
                "central_mg_l": c_central_mg_l,
                "extra_mg_l": c_extra_mg_l,
                "central_volume_ml": vc,
                "extra_volume_ml": ve,
            })
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

        q_fos_central = (
            _q6h_rate(time_min, fos.central_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min)
            if fos is not None
            else 0.0
        )
        # Second central line (two-phase setup): runs right after the main infusion ends.
        q_fos_slow = _slow_line_rate(time_min, fos) if fos is not None else 0.0
        q_fos_extra = 0.0
        if fos is not None and scenario == "overflow":
            q_fos_extra = _q6h_rate(time_min, fos.extra_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min)
            if q_fos_extra > 0 and _is_dose_start(time_min, fos.dosing_interval_min):
                extra_q6h_dose_count += 1
        c_central_mg_ml = a_central / vc
        c_extra_mg_ml = a_extra / ve
        setup_central_stock_mg_ml = fos.central_stock_mg_ml if fos is not None else 0.0
        setup_extra_stock_mg_ml = fos.extra_stock_mg_ml if fos is not None else 0.0
        setup_slow_stock_mg_ml = fos.slow_stock_mg_ml if fos is not None else 0.0
        central_input_mg_min = (
            q_fos_central * setup_central_stock_mg_ml
            + q_fos_slow * setup_slow_stock_mg_ml
            + system.q_extra_to_central_ml_min * c_extra_mg_ml
        )
        central_output_mg_min = (system.q_waste_ml_min + q_fos_central + q_fos_slow) * c_central_mg_ml
        extra_input_mg_min = q_fos_extra * setup_extra_stock_mg_ml
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
            # Every drug shares the same fixed central volume, so it washes out against the same total
            # instantaneous outflow, including the setup drug's own central pump flow (q_fos_central),
            # not just the shared diluent/extra-transfer flow.
            output_mg_min = (system.q_waste_ml_min + q_fos_central + q_fos_slow) * concentration_mg_ml
            drug_amounts[name] = max(0.0, amount + (input_mg_min - output_mg_min) * dt_min)

    summary = {
        "drug_preparation": _preparation_table(system, fos, continuous, drug_configs, scenario, duration_h),
    }
    if fos is not None:
        summary[fos.drug_name] = _summarize_intermit(rows, fos.drug_name, overflow_loss_mg, extra_q6h_dose_count)
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
    scenario: str = "q24_replacement",
    extra_infusion_ml_min: float = 0.0,
    duration_h: float = 168,
    dt_min: float = 1,
) -> CssCmaxSolverResult:
    if scenario not in {"q24_replacement", "overflow"}:
        raise ValueError("scenario must be 'q24_replacement' or 'overflow'")
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
        extra_infusion_ml_min=extra_infusion_ml_min if scenario == "overflow" else 0.0,
        infusion_duration_min=infusion_duration_min,
        dosing_interval_min=dosing_interval_min,
        preload_extra_mg=0.0,
        reservoir_replacement_interval_h=replacement_interval_h,
    )
    central_result = simulate_hfim(scenario, system, central_basis, [], duration_h=max(24, duration_h), dt_min=dt_min)
    extra_result = simulate_hfim(scenario, system, extra_basis, [], duration_h=max(24, duration_h), dt_min=dt_min)
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


def one_compartment_steady_state_profile(
    cavg_mg_l: float,
    half_life_h: float,
    interval_min: float,
    infusion_min: float,
    dt_min: float = 1,
) -> list[float]:
    """True steady-state profile of a one-compartment drug given as repeated IV infusion.

    This is the curve the HFIM setup is trying to reproduce for the long-half-life drug: the dose
    per interval is whatever delivers cavg_mg_l (dose / interval = CL x Cavg), infused over
    infusion_min and eliminated at half_life_h. One value per dt_min step across one interval.
    """
    if cavg_mg_l <= 0 or half_life_h <= 0 or interval_min <= 0 or infusion_min <= 0 or dt_min <= 0:
        return []
    duration = min(infusion_min, interval_min)
    k = math.log(2) / (half_life_h * 60)
    plateau = cavg_mg_l * interval_min / duration
    cmax = plateau * (1 - math.exp(-k * duration)) / (1 - math.exp(-k * interval_min))
    cmin = cmax * math.exp(-k * (interval_min - duration))
    steps = int(round(interval_min / dt_min))
    profile = []
    for step in range(steps):
        t = step * dt_min
        if t < duration:
            profile.append(plateau * (1 - math.exp(-k * t)) + cmin * math.exp(-k * t))
        else:
            profile.append(cmax * math.exp(-k * (t - duration)))
    return profile


def _periodic_central_profile(
    input_mg_min: list[float],
    pump_flow_ml_min: list[float],
    waste_flow_ml_min: float,
    central_volume_ml: float,
    dt_min: float,
) -> list[float]:
    """Steady-state central concentration (mg/L) over one dosing interval for a repeating input.

    Uses the same explicit time step as simulate_hfim, so the solved setup reproduces exactly what
    the full simulation shows once it has settled. Each step is A_next = A x gain + input x dt, so a
    whole interval is an affine map A_end = a x A_start + b; the repeating state is A_start = b / (1 - a).
    """
    gains = [1 - (waste_flow_ml_min + flow) / central_volume_ml * dt_min for flow in pump_flow_ml_min]
    a = 1.0
    b = 0.0
    for gain, rate in zip(gains, input_mg_min):
        b = b * gain + rate * dt_min
        a *= gain
    if not 0 <= a < 1:
        raise ValueError("time step is too large for the central washout rate")
    amount = b / (1 - a)
    profile = []
    for gain, rate in zip(gains, input_mg_min):
        profile.append(amount / central_volume_ml * 1000)
        amount = amount * gain + rate * dt_min
    return profile


def _weighted_nonnegative_fit(columns: list[list[float]], target: list[float]) -> list[float] | None:
    """Least-squares mix of basis profiles that best follows target in relative terms, all weights >= 0.

    There are at most three basis profiles (main infusion, slow line, extra feed), so every subset is
    simply tried: solve the small normal equations for that subset and keep the best all-non-negative one.
    """
    count = len(columns)
    weights = [1 / (value * value) for value in target]
    best = None
    for mask in range(1, 2 ** count):
        active = [index for index in range(count) if mask & (1 << index)]
        matrix = [
            [sum(w * columns[i][n] * columns[j][n] for n, w in enumerate(weights)) for j in active]
            for i in active
        ]
        rhs = [sum(w * columns[i][n] * target[n] for n, w in enumerate(weights)) for i in active]
        solution = _solve_linear(matrix, rhs)
        if solution is None or any(value < 0 for value in solution):
            continue
        coefficients = [0.0] * count
        for index, value in zip(active, solution):
            coefficients[index] = value
        residual = sum(
            w * (sum(coefficients[i] * columns[i][n] for i in active) - target[n]) ** 2
            for n, w in enumerate(weights)
        )
        if best is None or residual < best[0]:
            best = (residual, coefficients)
    return best[1] if best else None


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-300:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(size):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[col])]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def solve_two_phase_replacement(
    system: SystemConfig,
    drug_name: str,
    target_css_mg_l: float,
    reference_half_life_h: float,
    central_infusion_ml_min: float,
    infusion_duration_min: int,
    dosing_interval_min: int,
    slow_infusion_ml_min: float | None = None,
    dt_min: float = 1,
    slow_duration_step_min: int = 30,
    deviation_limit_pct: float = 10.0,
) -> TwoPhaseSolverResult:
    """Solve the q24h-replacement setup so the central curve follows the drug's own half-life.

    Central washout is fixed by the shortest half-life on the system, so a longer-half-life drug
    falls too fast once its infusion stops. Three drug sources are combined to slow that fall:

    - the main central infusion (dose volume and duration as entered),
    - a slow second central line that starts when the main infusion ends, and
    - the fixed-concentration extra compartment feeding central at Qextra the whole time.

    The solver picks the three strengths and the slow-line duration that track the true
    one-compartment steady-state curve most closely, then scales them so steady-state Cavg (and so
    AUC per day) equals the target exactly. It reports how far the result still is from the true
    curve; beyond deviation_limit_pct the setup should not be relied on for curve shape.

    The slow line always runs from its own syringe on a second pump. slow_infusion_ml_min=None means
    that syringe holds solution at the same concentration as the main infusion and the pump rate is
    solved; a number means the pump runs at that fixed rate and a weaker concentration is solved.
    """
    steps = int(round(dosing_interval_min / dt_min)) if dt_min > 0 else 0
    reference = one_compartment_steady_state_profile(
        target_css_mg_l, reference_half_life_h, dosing_interval_min, infusion_duration_min, dt_min
    )
    if (
        steps <= 0
        or not reference
        or central_infusion_ml_min <= 0
        or infusion_duration_min <= 0
        or infusion_duration_min > dosing_interval_min
    ):
        return TwoPhaseSolverResult(
            0.0, 0.0, 0.0, 0, 0.0, target_css_mg_l, reference_half_life_h, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, deviation_limit_pct, False,
            "Target, half-life, dose volume, infusion duration and dosing interval must all be positive, "
            "and the infusion cannot be longer than the dosing interval.",
            (), (), (),
        )

    vc = system.central_volume_ml
    waste = system.q_waste_ml_min
    q_extra = system.q_extra_to_central_ml_min
    times = [step * dt_min for step in range(steps)]
    in_main = [t < infusion_duration_min for t in times]
    main_input = [1.0 if flag else 0.0 for flag in in_main]
    floor_input = [1.0] * steps

    def evaluate(slow_duration_min: int):
        slow_end = infusion_duration_min + slow_duration_min
        in_slow = [infusion_duration_min <= t < slow_end for t in times]
        slow_input = [1.0 if flag else 0.0 for flag in in_slow]
        slow_flow = slow_infusion_ml_min if slow_infusion_ml_min is not None else 0.0
        outcome = None
        # With a shared stock the slow line's own (small) pump flow depends on the answer, so the
        # fit is repeated a few times with the flow it implies; it settles after one or two passes.
        for _ in range(4 if slow_infusion_ml_min is None and slow_duration_min > 0 else 1):
            pump_flow = [
                (central_infusion_ml_min if main else 0.0) + (slow_flow if slow else 0.0)
                for main, slow in zip(in_main, in_slow)
            ]
            columns = [_periodic_central_profile(main_input, pump_flow, waste, vc, dt_min)]
            labels = ["main"]
            if slow_duration_min > 0:
                columns.append(_periodic_central_profile(slow_input, pump_flow, waste, vc, dt_min))
                labels.append("slow")
            if q_extra > 0:
                columns.append(_periodic_central_profile(floor_input, pump_flow, waste, vc, dt_min))
                labels.append("floor")
            fitted = _weighted_nonnegative_fit(columns, reference)
            if fitted is None:
                return None
            profile = [sum(c * column[n] for c, column in zip(fitted, columns)) for n in range(steps)]
            cavg = sum(profile) / steps
            if cavg <= 0:
                return None
            scale = target_css_mg_l / cavg
            rates = {label: value * scale for label, value in zip(labels, fitted)}
            profile = [value * scale for value in profile]
            main_rate = rates.get("main", 0.0)
            slow_rate = rates.get("slow", 0.0)
            outcome = (profile, main_rate, slow_rate, rates.get("floor", 0.0), slow_flow)
            if slow_infusion_ml_min is not None or slow_duration_min == 0:
                break
            if main_rate <= 0:
                return None
            # Same concentration as the main infusion: concentration = main mg/min / main mL/min.
            slow_flow = slow_rate / (main_rate / central_infusion_ml_min)
        return outcome

    def max_deviation(profile: list[float]) -> float:
        return max(abs(value / ref - 1) for value, ref in zip(profile, reference)) * 100

    candidates = [0]
    longest = int(dosing_interval_min - infusion_duration_min)
    step_size = max(1, int(slow_duration_step_min))
    candidates.extend(range(step_size, longest + 1, step_size))
    best = None
    single_phase_deviation = 0.0
    for slow_duration in candidates:
        outcome = evaluate(slow_duration)
        if outcome is None:
            continue
        deviation = max_deviation(outcome[0])
        if slow_duration == 0:
            single_phase_deviation = deviation
        if outcome[2] <= 0 and slow_duration > 0:
            continue
        if best is None or deviation < best[0] - 1e-9:
            best = (deviation, slow_duration, outcome)
    if best is None:
        return TwoPhaseSolverResult(
            0.0, 0.0, 0.0, 0, 0.0, target_css_mg_l, reference_half_life_h, 0.0, 0.0, 0.0,
            max(reference), min(reference), 0.0, 0.0, deviation_limit_pct, False,
            "No non-negative combination of central infusion, slow line and extra feed could reach the target.",
            (), (), (),
        )

    deviation, slow_duration, (profile, main_rate, slow_rate, floor_rate, slow_flow) = best
    central_stock = main_rate / central_infusion_ml_min
    if slow_duration > 0 and slow_rate > 0:
        slow_stock = central_stock if slow_infusion_ml_min is None else slow_rate / slow_infusion_ml_min
    else:
        slow_duration, slow_flow, slow_stock = 0, 0.0, 0.0
    extra_concentration = floor_rate / q_extra if q_extra > 0 else 0.0
    feasible = deviation <= deviation_limit_pct
    if feasible:
        message = (
            f"Two-phase setup follows the true {reference_half_life_h:g} h curve within "
            f"{deviation:.1f}% at every point of the dosing interval."
        )
    else:
        message = (
            f"Two-phase setup stays up to {deviation:.1f}% away from the true {reference_half_life_h:g} h curve, "
            f"above the {deviation_limit_pct:g}% limit. Daily AUC is still exact, but curve shape, trough and "
            "T>MIC are not reliable for this combination; use a Blaser-type setup for it."
        )
    return TwoPhaseSolverResult(
        central_stock_mg_ml=central_stock,
        slow_stock_mg_ml=slow_stock,
        slow_infusion_ml_min=slow_flow,
        slow_duration_min=slow_duration,
        extra_replacement_concentration_mg_ml=extra_concentration,
        target_css_mg_l=target_css_mg_l,
        reference_half_life_h=reference_half_life_h,
        predicted_cavg_mg_l=sum(profile) / steps,
        predicted_cmax_mg_l=max(profile),
        predicted_cmin_mg_l=min(profile),
        reference_cmax_mg_l=max(reference),
        reference_cmin_mg_l=min(reference),
        max_deviation_pct=deviation,
        single_phase_max_deviation_pct=single_phase_deviation,
        deviation_limit_pct=deviation_limit_pct,
        feasible=feasible,
        message=message,
        cycle_time_h=tuple(t / 60 for t in times),
        predicted_profile_mg_l=tuple(profile),
        reference_profile_mg_l=tuple(reference),
    )


def _q6h_rate(time_min: float, rate_ml_min: float, duration_min: int, interval_min: int) -> float:
    return rate_ml_min if time_min % interval_min < duration_min else 0.0


def _slow_line_rate(time_min: float, fos: FosfomycinConfig) -> float:
    if not fos.has_slow_line:
        return 0.0
    position = time_min % fos.dosing_interval_min
    start = fos.infusion_duration_min
    return fos.slow_infusion_ml_min if start <= position < start + fos.slow_duration_min else 0.0


def average_intermittent_rate(rate_ml_min: float, duration_min: int, interval_min: int) -> float:
    if rate_ml_min <= 0 or duration_min <= 0 or interval_min <= 0:
        return 0.0
    duty_cycle = min(duration_min, interval_min) / interval_min
    return rate_ml_min * duty_cycle


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
    last_end_h = drug_rows[-1]["time_h"]
    last_start_h = max(0.0, last_end_h - 24)
    last_rows = [row for row in drug_rows if row["time_h"] >= last_start_h - 1e-9]
    return {
        "central_cmax_mg_l": max(central),
        # Minimum over the whole series, dominated by the pre-dose baseline (0 at t=0) - not a
        # steady-state trough. Use central_cmin_after_24h_mg_l for the clinically meaningful trough.
        "central_cmin_overall_mg_l": min(central),
        "central_cmin_after_24h_mg_l": min(
            row["central_mg_l"]
            for row in drug_rows
            if row["time_h"] >= 24
        ) if any(row["time_h"] >= 24 for row in drug_rows) else min(central),
        "central_cavg_0_24_mg_l": _auc(drug_rows, "central_mg_l", 24) / 24,
        "extra_cmax_mg_l": max(extra),
        # Same caveat as central_cmin_overall_mg_l - whole-series minimum, not a steady-state trough.
        "extra_cmin_overall_mg_l": min(extra),
        "central_auc_0_24_mg_h_l": _auc(drug_rows, "central_mg_l", 24),
        # Last 24 h of the run: the settled daily exposure once the system has stopped filling up.
        # Equals the 0-24 h figures when the run is only 24 h long.
        "central_auc_last_24h_mg_h_l": _auc_window(drug_rows, "central_mg_l", last_start_h, last_end_h),
        "central_cmax_last_24h_mg_l": max(row["central_mg_l"] for row in last_rows),
        "central_cmin_last_24h_mg_l": min(row["central_mg_l"] for row in last_rows),
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


def _auc_window(rows: list[dict], column: str, start_h: float, end_h: float) -> float:
    total = 0.0
    usable = [row for row in rows if start_h - 1e-9 <= row["time_h"] <= end_h + 1e-9]
    for prev, curr in zip(usable, usable[1:]):
        total += (prev[column] + curr[column]) * 0.5 * (curr["time_h"] - prev["time_h"])
    return total


def _preparation_table(
    system: SystemConfig,
    fos: FosfomycinConfig | None,
    continuous: dict[str, ContinuousInfusionRegimen],
    drug_configs: dict[str, DrugConfig],
    scenario: str,
    duration_h: float,
) -> list[dict]:
    table: list[dict] = []
    if fos is None:
        # central_only: no setup drug, so emit no setup-drug central/extra rows at all rather than
        # zero-amount placeholder rows that would show up as meaningless bench instructions.
        return table + _central_drug_preparation_rows(system, continuous, drug_configs)
    table.append(
        {
            "drug": fos.drug_name,
            "component": f"central q{fos.dosing_interval_min / 60:g}h infusion",
            "amount_mg": fos.central_dose_mg,
            "daily_amount_mg": fos.central_dose_mg * 24 / (fos.dosing_interval_min / 60),
            "note": f"{fos.central_infusion_ml_min * fos.infusion_duration_min:g} mL over {fos.infusion_duration_min / 60:g} h",
        }
    )
    if fos.has_slow_line:
        slow_start_h = fos.infusion_duration_min / 60
        slow_end_h = slow_start_h + fos.slow_duration_min / 60
        table.append({
            "drug": fos.drug_name,
            "component": f"central q{fos.dosing_interval_min / 60:g}h slow line (h {slow_start_h:g}-{slow_end_h:g})",
            "amount_mg": fos.slow_dose_mg,
            "daily_amount_mg": fos.slow_dose_mg * 24 / (fos.dosing_interval_min / 60),
            "note": (
                f"{fos.slow_dose_volume_ml:.3f} mL over {fos.slow_duration_min / 60:g} h "
                f"at {fos.slow_stock_mg_ml:g} mg/mL and {fos.slow_infusion_ml_min * 60:.3f} mL/h; "
                f"pump 2 with its own syringe, starts when the main infusion ends"
            ),
        })
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
    return table + _central_drug_preparation_rows(system, continuous, drug_configs)


def _central_drug_preparation_rows(
    system: SystemConfig,
    continuous: dict[str, ContinuousInfusionRegimen],
    drug_configs: dict[str, DrugConfig],
) -> list[dict]:
    table = []
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
