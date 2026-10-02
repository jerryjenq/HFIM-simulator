from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import math
from pathlib import Path
import textwrap

from .agent import ask_setup_agent, build_agent_context
from .pk import (
    DrugConfig,
    FosfomycinConfig,
    SystemConfig,
    average_intermittent_rate,
    flow_for_half_life,
    half_life_for_flow,
    intermittent_peak_trough,
    simulate_hfim,
    solve_blaser_setup,
    solve_css_cmax_replacement,
    solve_two_phase_replacement,
)
from .store import SimulationStore


def _render_global_styles(st) -> None:
    st.markdown(
        """
        <style>
        /* st.metric's default value font is fixed-size and clips long numbers with an invisible
        ellipsis instead of wrapping. This app shows 6-decimal mg/mL concentrations that operators
        weigh reagents against, so a clipped digit is a real dosing risk, not just a cosmetic issue. */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            white-space: normal;
            overflow-wrap: break-word;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="HFIM PK Simulator", layout="wide")
    _render_global_styles(st)
    st.navigation([
        st.Page(_page_two_half_life, title="2 half life", url_path="two-half-life", default=True),
        st.Page(_page_one_half_life, title="1 half life", url_path="one-half-life"),
        st.Page(_page_blaser, title="2 half life - Blaser", url_path="blaser"),
    ]).run()


def _page_two_half_life() -> None:
    import streamlit as st

    st.title("HFIM PK Simulator")
    st.caption(
        "Two different half-lives: one drug uses the central + extra compartment setup so its apparent "
        "half-life can be longer than the shared central washout. Enter experimental conditions, simulate "
        "central and extra-compartment PK concentrations, and estimate how much drug to prepare."
    )

    st.subheader("1. Simulation setup")
    setup_cols = st.columns(4)
    active_drug_count = int(setup_cols[0].number_input(
        "Number of drugs",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="This prototype supports up to 6 drugs. One selected drug can use the central/extra setup; the others use central-only loading and maintenance dosing.",
    ))
    scenario_label = setup_cols[1].selectbox(
        "Selected-drug extra strategy",
        [
            "q24h full extra replacement",
            "Overflow: fixed extra volume with extra outflow to waste",
        ],
        help="This strategy applies to the drug selected in Section 3 for the central/extra setup. It does not automatically apply to every drug.",
    )
    scenario = _scenario_from_label(scenario_label)
    duration_h = setup_cols[2].number_input(
        "Simulation duration (h)",
        min_value=24.0,
        value=168.0,
        step=24.0,
        help="AUC0-24 and Cavg/Css require at least 24 h of simulated data.",
    )
    dt_min = setup_cols[3].number_input("Time step (min)", min_value=0.25, value=1.0, step=0.25)
    shared_central_half_life_h = _shared_central_half_life_from_widget_state(active_drug_count, st.session_state)

    with st.expander("Compartment and flow settings", expanded=True):
        st.markdown(
            "Flow is not a free parameter. The shortest active drug half-life sets the shared central-to-waste flow. "
            "Central diluent is the remaining inflow needed after extra-to-central transfer."
        )
        helper_cols = st.columns(3)
        flow_mode = helper_cols[0].selectbox(
            "Flow setup mode",
            [
                "Auto flow from target half-life (fixed volume)",
                "Manual flow entry (do not auto-adjust)",
            ],
        )
        target_system_half_life = shared_central_half_life_h
        helper_cols[1].metric("Shared central half-life", f"{target_system_half_life:.2f} h")
        helper_cols[2].caption(
            "Auto mode uses the shortest active drug half-life. Total central outflow = ln(2) x central volume / half-life. "
            "Then central diluent = total central outflow - extra-to-central transfer."
            if scenario == "q24_replacement"
            else "Auto mode uses the shortest active drug half-life for central washout and the selected half-life for extra washout. Manual mode keeps the flow fields editable."
        )

        cols = st.columns(6)
        central_bottle_ml = cols[0].number_input("Central bottle (mL)", min_value=1.0, value=100.0, step=5.0)
        cartridge_ml = cols[1].number_input("Cartridge (mL)", min_value=1.0, value=70.0, step=5.0)
        extra_volume_ml = cols[2].number_input("Extra volume (mL)", min_value=1.0, value=241.0, step=1.0)
        auto_central_flow = flow_for_half_life(central_bottle_ml + cartridge_ml, target_system_half_life)
        auto_extra_flow = flow_for_half_life(extra_volume_ml, target_system_half_life)
        auto_flow_mode = _is_auto_flow_mode(flow_mode)
        q_extra_default = _qextra_default_for_scenario(scenario, flow_mode, auto_extra_flow)
        # Kept short and the same length in both scenarios so the six flow-setting columns don't wrap
        # to different heights and misalign; the "fixed transfer" nuance lives in the help tooltip.
        qextra_label = "Extra to central (mL/min)"
        q_extra_to_central = cols[3].number_input(
            qextra_label,
            min_value=0.0,
            value=q_extra_default,
            step=0.001,
            format="%.3f",
            disabled=scenario != "q24_replacement" and auto_flow_mode,
            key=_flow_widget_key("qextra", scenario, auto_flow_mode and scenario != "q24_replacement", extra_volume_ml, target_system_half_life),
            help=(
                "In q24 replacement mode this is a physical transfer setting, not a value automatically derived from extra volume. "
                "Change it only if the real extra-to-central transfer rate changes."
                if scenario == "q24_replacement"
                else "In overflow mode this can be auto-calculated from the extra washout half-life."
            ),
        )
        q_extra_diluent = cols[4].number_input(
            "Extra diluent (mL/min)" if scenario != "q24_replacement" else "Extra diluent (not used)",
            min_value=0.0,
            value=0.0 if scenario == "q24_replacement" else q_extra_to_central if auto_flow_mode else 0.921,
            step=0.001,
            format="%.3f",
            disabled=scenario == "q24_replacement" or auto_flow_mode,
            key=_flow_widget_key("extra_diluent", scenario, auto_flow_mode, extra_volume_ml, target_system_half_life),
            help=(
                "Sets the extra-compartment diluent volume to prepare. This flow is not read by the PK "
                "calculation - the simulator assumes the extra volume stays fixed. Keep it close to extra-to-central "
                "(plus overflow demand while dosing) so the physical extra volume actually stays fixed."
            ),
        )
        q_central_diluent = cols[5].number_input(
            "Central diluent (mL/min)",
            min_value=0.0,
            value=_central_diluent_default_for_flow_mode(flow_mode, auto_central_flow, q_extra_to_central),
            step=0.001,
            format="%.3f",
            disabled=auto_flow_mode,
            key=_flow_widget_key(
                "central_diluent",
                scenario,
                auto_flow_mode,
                central_bottle_ml + cartridge_ml + q_extra_to_central,
                target_system_half_life,
            ),
        )
        recirculation_ml_min = st.number_input(
            "Cartridge recirculation (mL/min)",
            min_value=1.0,
            value=120.0,
            step=5.0,
            help=(
                "Central-to-cartridge loop flow shown on the schematic below. This is a display-only label - "
                "it does not feed into the PK calculation. Set it to match the actual recirculation pump rate "
                "so the schematic reflects the real setup."
            ),
        )
        total_central_outflow = q_extra_to_central + q_central_diluent
        achieved_central_half_life = half_life_for_flow(central_bottle_ml + cartridge_ml, total_central_outflow) if total_central_outflow > 0 else None
        achieved_extra_half_life = half_life_for_flow(extra_volume_ml, q_extra_to_central) if q_extra_to_central > 0 else None
        if auto_flow_mode and q_extra_to_central > auto_central_flow:
            st.warning(
                f"Qextra ({q_extra_to_central:.3f} mL/min) is already higher than the shared central target outflow "
                f"({auto_central_flow:.3f} mL/min). Central diluent is set to 0, so the achieved central half-life will be shorter than {target_system_half_life:.2f} h."
            )
        if scenario == "q24_replacement":
            max_single_fill_qextra = extra_volume_ml / (24 * 60)
            q24_transfer_volume = q_extra_to_central * 24 * 60
            st.info(
                f"Current setup gives central washout half-life approximately {_fmt_optional(achieved_central_half_life)} h. "
                f"Target total central outflow from the shortest active half-life is {auto_central_flow:.3f} mL/min; "
                f"central diluent is {q_central_diluent:.3f} mL/min after subtracting Qextra {q_extra_to_central:.3f} mL/min. "
                f"The extra-to-central transfer is fixed and is not auto-scaled by extra volume. "
                f"At this rate, 24 h transfer volume is {q24_transfer_volume:.1f} mL. "
                f"In pressure/filter-controlled replacement, this liquid leaves the same extra fill; it is not an additional reserve solution. "
                f"One {extra_volume_ml:.1f} mL extra fill can physically support up to {max_single_fill_qextra:.3f} mL/min for 24 h before running dry. "
                f"For total central outflow: ln(2) x {central_bottle_ml + cartridge_ml:g} mL / ({target_system_half_life:g} h x 60) = {auto_central_flow:.3f} mL/min."
            )
        else:
            st.info(
                f"Current setup gives central washout half-life approximately {_fmt_optional(achieved_central_half_life)} h and extra washout half-life approximately {_fmt_optional(achieved_extra_half_life)} h. "
                f"For the extra compartment: ln(2) x {extra_volume_ml:g} mL / ({target_system_half_life:g} h x 60) = {auto_extra_flow:.3f} mL/min. "
                f"For total central outflow: ln(2) x {central_bottle_ml + cartridge_ml:g} mL / ({target_system_half_life:g} h x 60) = {auto_central_flow:.3f} mL/min."
            )
        extra_diluent_warning = _extra_diluent_balance_warning(scenario, q_extra_to_central, q_extra_diluent, extra_volume_ml)
        if extra_diluent_warning:
            st.warning(extra_diluent_warning)

    st.subheader("2. Drug targets and injection settings")
    st.caption("Choose the number of drugs, then set whether each drug should target AUC0-24, Cmax, or maintained Css. Dosing settings are split into loading dose, maintenance dosing, and dosing frequency.")
    drug_inputs = _drug_input_panel(st, active_drug_count)

    st.subheader("3. Css/Cmax solver and replacement setup" if scenario == "q24_replacement" else "3. Intermittent / extra-compartment drug setup")
    st.caption(
        "This section solves central stock and q24h extra replacement concentration from target Css/Cavg and Cmax."
        if scenario == "q24_replacement"
        else "This section solves central and extra stock concentrations for the overflow central/extra setup, then lets you manually override the suggested stock values."
    )
    st.info(_extra_setup_help_text(scenario))
    setup_drug_names = list(drug_inputs.keys())
    default_setup_index = next(
        (index for index, name in enumerate(setup_drug_names) if drug_inputs[name]["maintenance"] == "intermittent infusion"),
        0,
    )
    setup_drug_name = st.selectbox("Drug for this central/extra setup", setup_drug_names, index=default_setup_index)
    st.markdown(_optimization_guidance_text(setup_drug_name, scenario))
    setup_drug_values = drug_inputs[setup_drug_name]
    if setup_drug_values:
        two_phase = False
        if scenario == "q24_replacement":
            two_phase = st.checkbox(
                f"Two-phase central dosing: add a slow second central line so {setup_drug_name} follows its own "
                f"{setup_drug_values['half_life_h']:g} h half-life",
                value=True,
                help=(
                    "Central washout is set by the shortest half-life, so a longer-half-life drug falls too fast after "
                    "each infusion. With this on, a second central line runs slowly right after the main infusion and the "
                    "extra-compartment concentration is lowered to match, so the whole curve tracks the true half-life "
                    "instead of only matching AUC and Cmax. Turn it off for the original single-infusion AUC + Cmax solver."
                ),
            )
        target_cols = st.columns(3)
        target_css_mg_l = target_cols[0].number_input("Target Css / Cavg (mg/L)", min_value=0.0, value=setup_drug_values["target_concentration_mg_l"], step=5.0)
        target_cmax_mg_l = target_cols[1].number_input(
            "Target Cmax (mg/L)",
            min_value=0.0,
            value=250.0,
            step=5.0,
            disabled=two_phase,
            help=(
                "Not used in two-phase mode: Cmax follows from the half-life, dosing interval and infusion duration."
                if two_phase
                else None
            ),
        )
        if scenario == "q24_replacement":
            reservoir_replacement_interval_h = target_cols[2].number_input("Extra replacement interval (h)", min_value=0.1, value=24.0, step=1.0)
        else:
            reservoir_replacement_interval_h = 24.0
            target_cols[2].metric("Extra strategy", "overflow q-dose")

        setup_cols = st.columns(4)
        fos_central_volume = setup_cols[0].number_input("Central dose volume (mL)", min_value=0.0, value=6.0, step=0.5)
        fos_duration_h = setup_cols[1].number_input("Infusion duration (h)", min_value=0.01, value=1.0, step=0.25)
        fos_frequency_h = setup_cols[2].number_input("Dosing frequency (h)", min_value=0.1, value=float(setup_drug_values["dosing_frequency_h"] or 6.0), step=1.0)
        fos_duration = int(round(fos_duration_h * 60))
        fos_interval = int(round(fos_frequency_h * 60))
        q24_system = SystemConfig(
            central_bottle_ml=central_bottle_ml,
            cartridge_ml=cartridge_ml,
            extra_volume_ml=extra_volume_ml,
            q_extra_to_central_ml_min=q_extra_to_central,
            q_extra_diluent_ml_min=q_extra_diluent,
            q_central_diluent_ml_min=q_central_diluent,
        )
        fos_central_rate = fos_central_volume / fos_duration
        if scenario == "overflow":
            overflow_setup_cols = st.columns(2)
            fos_extra_volume = overflow_setup_cols[0].number_input("Extra dose volume (mL)", min_value=0.0, value=6.0, step=0.5)
            fos_extra_rate = fos_extra_volume / fos_duration
            overflow_setup_cols[1].metric("Extra pump rate", f"{fos_extra_rate:.3f} mL/min")
        else:
            fos_extra_volume = 0.0
            fos_extra_rate = 0.0

        solver_result = solve_css_cmax_replacement(
            q24_system,
            setup_drug_name,
            target_css_mg_l,
            target_cmax_mg_l,
            fos_central_rate,
            fos_duration,
            fos_interval,
            replacement_interval_h=reservoir_replacement_interval_h,
            scenario=scenario,
            extra_infusion_ml_min=fos_extra_rate,
            duration_h=duration_h,
            dt_min=dt_min,
        )
        two_phase_result = None
        if two_phase:
            slow_cols = st.columns(2)
            slow_source = slow_cols[0].selectbox(
                "Slow line source",
                [_SLOW_LINE_SAME_STOCK, _SLOW_LINE_SEPARATE_STOCK],
                help=(
                    "The slow line always has its own syringe on a second pump. Same concentration: fill that syringe "
                    "with solution at the main-infusion concentration and run it at the solved low rate. "
                    "Weaker concentration: keep the pump at a rate you choose and prepare a more dilute solution instead."
                ),
            )
            slow_line_fixed_rate_ml_min = None
            if slow_source == _SLOW_LINE_SEPARATE_STOCK:
                slow_line_fixed_rate_ml_min = slow_cols[1].number_input(
                    "Slow line pump rate (mL/h)", min_value=0.01, value=6.0, step=0.5
                ) / 60
            two_phase_result = solve_two_phase_replacement(
                q24_system,
                setup_drug_name,
                target_css_mg_l,
                setup_drug_values["half_life_h"],
                fos_central_rate,
                fos_duration,
                fos_interval,
                slow_infusion_ml_min=slow_line_fixed_rate_ml_min,
                dt_min=dt_min,
                slow_duration_step_min=30 if fos_interval <= 720 else 60,
            )
        two_phase_active = two_phase_result is not None and two_phase_result.central_stock_mg_ml > 0
        fos_central_stock = two_phase_result.central_stock_mg_ml if two_phase_active else solver_result.central_stock_mg_ml
        fos_slow_stock = two_phase_result.slow_stock_mg_ml if two_phase_active else 0.0
        fos_slow_rate = two_phase_result.slow_infusion_ml_min if two_phase_active else 0.0
        fos_slow_duration = two_phase_result.slow_duration_min if two_phase_active else 0
        setup_cols[3].metric("Solved central stock", f"{fos_central_stock:.6f} mg/mL")

        if scenario == "q24_replacement":
            preload_extra_mg = 0.0
            extra_transfer_volume_ml = q_extra_to_central * reservoir_replacement_interval_h * 60
            if two_phase_active:
                fos_extra_stock = two_phase_result.extra_replacement_concentration_mg_ml
                _render_two_phase_solver_panel(st, two_phase_result)
            else:
                if two_phase_result is not None:
                    st.warning(two_phase_result.message + " Showing the single-infusion AUC + Cmax solver instead.")
                fos_extra_stock = solver_result.extra_replacement_concentration_mg_ml
                solver_cols = st.columns(4)
                solver_cols[0].metric("Solved extra replacement", f"{fos_extra_stock:.6f} mg/mL")
                solver_cols[1].metric("Predicted Cavg", f"{solver_result.predicted_cavg_mg_l:.1f} mg/L")
                solver_cols[2].metric("Predicted Cmax", f"{solver_result.predicted_cmax_mg_l:.1f} mg/L", f"{solver_result.cmax_error_mg_l:+.1f}")
                solver_cols[3].metric("Predicted Cmin after 24h", f"{solver_result.predicted_cmin_mg_l:.1f} mg/L")
                if solver_result.feasible:
                    st.success(solver_result.message)
                else:
                    st.warning(solver_result.message)
            if extra_transfer_volume_ml > extra_volume_ml:
                max_flow_from_single_fill = extra_volume_ml / (reservoir_replacement_interval_h * 60)
                st.error(
                    f"Physical feasibility warning: {q_extra_to_central:.3f} mL/min for q{reservoir_replacement_interval_h:g}h "
                    f"moves {extra_transfer_volume_ml:.1f} mL from extra to central. "
                    f"A {extra_volume_ml:.1f} mL pressure/filter-controlled fill would run dry before the interval ends. "
                    f"To use only one {extra_volume_ml:.1f} mL fill, Qextra must be <= {max_flow_from_single_fill:.3f} mL/min. "
                    f"Lower Qextra, increase extra volume, or shorten the replacement interval."
                )
        else:
            solver_cols = st.columns(4)
            solver_cols[0].metric("Solved extra stock", f"{solver_result.extra_replacement_concentration_mg_ml:.6f} mg/mL")
            solver_cols[1].metric("Predicted Cavg", f"{solver_result.predicted_cavg_mg_l:.1f} mg/L")
            solver_cols[2].metric("Predicted Cmax", f"{solver_result.predicted_cmax_mg_l:.1f} mg/L", f"{solver_result.cmax_error_mg_l:+.1f}")
            solver_cols[3].metric("Predicted Cmin after 24h", f"{solver_result.predicted_cmin_mg_l:.1f} mg/L")
            if solver_result.feasible:
                st.success(solver_result.message)
            else:
                st.warning(solver_result.message)
            manual_cols = st.columns(3)
            fos_central_stock = manual_cols[0].number_input(
                "Central stock (mg/mL)",
                min_value=0.0,
                value=fos_central_stock,
                step=0.1,
                format="%.6f",
                help=(
                    "Defaults to the overflow Css/Cmax solver result. The final simulated Cavg/Cmax in "
                    "Section 5 updates if you manually override this stock concentration."
                ),
            )
            fos_extra_stock = manual_cols[1].number_input(
                "Extra stock (mg/mL)",
                min_value=0.0,
                value=solver_result.extra_replacement_concentration_mg_ml,
                step=0.1,
                format="%.6f",
                help="Defaults to the overflow Css/Cmax solver result for the selected extra q-dose volume.",
            )
            preload_default = fos_extra_stock * fos_extra_volume
            preload_extra_mg = manual_cols[2].number_input("Extra preload amount (mg)", min_value=0.0, value=preload_default, step=1.0)
        if scenario == "q24_replacement":
            slow_line_caption = (
                f"Slow line (pump 2, own syringe at {fos_slow_stock:.6f} mg/mL): {fos_slow_rate * 60:.3f} mL/h "
                f"for {fos_slow_duration / 60:g} h right after each main infusion "
                f"= {fos_slow_stock * fos_slow_rate * fos_slow_duration:.3f} mg in {fos_slow_rate * fos_slow_duration:.3f} mL. "
                if two_phase_active and fos_slow_duration > 0
                else ""
            )
            st.caption(
                f"Calculated central pump rate: {fos_central_rate:.3f} mL/min. "
                f"Each central dose = {fos_central_stock * fos_central_volume:.3f} mg. "
                + slow_line_caption
                + f"Prepare {extra_volume_ml:.1f} mL of extra replacement solution at {fos_extra_stock:.6f} mg/mL "
                f"per q{reservoir_replacement_interval_h:g}h interval before overfill."
            )
        else:
            st.caption(
                f"Calculated pump rates: central {fos_central_rate:.3f} mL/min, extra {fos_extra_rate:.3f} mL/min. "
                f"Each central dose = {fos_central_stock * fos_central_volume:.3f} mg; each extra dose = {fos_extra_stock * fos_extra_volume:.3f} mg."
            )
    else:
        fos_central_stock = fos_extra_stock = fos_central_rate = fos_extra_rate = preload_extra_mg = 0.0
        fos_duration = 60
        fos_interval = 360
        reservoir_replacement_interval_h = 24.0
        target_css_mg_l = 150.0
        target_cmax_mg_l = 250.0
        solver_result = None
        two_phase_result = None
        two_phase_active = False
        fos_slow_stock = fos_slow_rate = 0.0
        fos_slow_duration = 0

    system = SystemConfig(
        central_bottle_ml=central_bottle_ml,
        cartridge_ml=cartridge_ml,
        extra_volume_ml=extra_volume_ml,
        q_extra_to_central_ml_min=q_extra_to_central,
        q_extra_diluent_ml_min=q_extra_diluent,
        q_central_diluent_ml_min=q_central_diluent,
    )
    fos = FosfomycinConfig(
        drug_name=setup_drug_name,
        central_stock_mg_ml=fos_central_stock,
        extra_stock_mg_ml=fos_extra_stock,
        central_infusion_ml_min=fos_central_rate,
        extra_infusion_ml_min=fos_extra_rate,
        infusion_duration_min=fos_duration,
        dosing_interval_min=fos_interval,
        preload_extra_mg=preload_extra_mg,
        reservoir_replacement_interval_h=reservoir_replacement_interval_h,
        slow_stock_mg_ml=fos_slow_stock,
        slow_infusion_ml_min=fos_slow_rate,
        slow_duration_min=fos_slow_duration,
    )

    st.subheader("4. Editable setup and injection overview")
    drugs = []
    for name, values in drug_inputs.items():
        if name != setup_drug_name:
            drugs.append(DrugConfig(
                name,
                target_concentration_mg_l=values["target_concentration_mg_l"],
                half_life_h=values["half_life_h"],
                dosing_mode=values["dosing_mode"],
                loading_target_concentration_mg_l=values["loading_target_concentration_mg_l"],
                loading_duration_h=values["loading_duration_h"],
                loading_volume_ml=values["loading_volume_ml"],
                intermittent_interval_h=values["dosing_frequency_h"] or 6.0,
                intermittent_duration_h=values["maintenance_duration_h"],
            ))
    result = simulate_hfim(scenario, system, fos, drugs, duration_h=duration_h, dt_min=dt_min)

    # One integrated apparatus diagram: vessels, flows and dosing instructions on a single canvas,
    # so there is no separate recipe panel to cross-reference against the plumbing.
    apparatus_fig = _plot_two_half_life_apparatus(_two_half_life_apparatus_view(
        system, fos, drug_inputs, result.summary, scenario, duration_h, recirculation_ml_min, target_system_half_life,
    ))
    st.image(_figure_export_bytes(apparatus_fig, "png", dpi=300), width="stretch")
    st.caption(
        "Volumes on the waste and diluent bottles are totals for the whole run. Concentrations are shown "
        "in µg/mL. The central compartment is magnetically stirred."
    )
    _render_schematic_export_buttons(st, apparatus_fig, "apparatus2", "hfim-apparatus-2-half-life")
    st.dataframe(
        _setup_overview_rows(central_bottle_ml, cartridge_ml, extra_volume_ml, q_extra_to_central, q_extra_diluent, q_central_diluent, scenario, fos),
        width="stretch",
        hide_index=True,
        column_config={
            "Part": st.column_config.Column(width="small"),
            "Current value": st.column_config.Column(width="small"),
            "How to use": st.column_config.Column(width="large"),
        },
    )
    st.markdown("**System solution volumes**")
    st.dataframe(_solution_volume_rows(q_central_diluent, q_extra_diluent, scenario, duration_h), width="stretch", hide_index=True)
    st.markdown("**Drug injection plan**")
    st.dataframe(
        _injection_plan_rows(drug_inputs, fos, scenario, setup_drug_name),
        width="stretch",
        hide_index=True,
        column_config={
            "Drug": st.column_config.Column(width="small"),
            "Target": st.column_config.Column(width="medium"),
            "Half-life": st.column_config.Column(width="small"),
            "Dosing plan": st.column_config.Column(width="large"),
        },
    )

    run_and_save = st.button("Run and save to SQLite")
    setup_summary = result.summary[setup_drug_name]
    setup_target_auc = target_css_mg_l * 24

    st.subheader("5. Result overview")
    cols = st.columns(5)
    cols[0].metric(f"{setup_drug_name} AUC0-24", f"{setup_summary['central_auc_0_24_mg_h_l']:.1f}", f"target {setup_target_auc:g}" if setup_target_auc else None)
    cols[1].metric("Central Cavg/Css", f"{setup_summary['central_cavg_0_24_mg_l']:.1f} mg/L", f"target {target_css_mg_l:g}")
    cols[2].metric(
        f"{setup_drug_name} Cmax central",
        f"{setup_summary['central_cmax_mg_l']:.1f} mg/L",
        # Two-phase mode has no Cmax target: the peak follows from the half-life and dose timing.
        f"true curve {two_phase_result.reference_cmax_mg_l:.1f}" if two_phase_active else f"target {target_cmax_mg_l:g}",
    )
    cmin_value = setup_summary["central_cmin_after_24h_mg_l"]
    cols[3].metric(f"{setup_drug_name} Cmin after 24h", f"{cmin_value:.1f} mg/L")
    if scenario == "q24_replacement":
        cols[4].metric("Extra replacement", f"{fos.extra_stock_mg_ml:.6f} mg/mL")
    else:
        cols[4].metric(f"{setup_drug_name} overflow loss", f"{setup_summary['overflow_loss_mg']:.2f} mg")

    if two_phase_active:
        steady_cols = st.columns(4)
        steady_cols[0].metric(
            "Steady-state AUC per 24 h",
            f"{setup_summary['central_auc_last_24h_mg_h_l']:.1f}",
            f"target {setup_target_auc:g}" if setup_target_auc else None,
        )
        steady_cols[1].metric(
            "Steady-state Cmax",
            f"{setup_summary['central_cmax_last_24h_mg_l']:.1f} mg/L",
            f"true curve {two_phase_result.reference_cmax_mg_l:.1f}",
            delta_color="off",
        )
        steady_cols[2].metric(
            "Steady-state Cmin",
            f"{setup_summary['central_cmin_last_24h_mg_l']:.1f} mg/L",
            f"true curve {two_phase_result.reference_cmin_mg_l:.1f}",
            delta_color="off",
        )
        steady_cols[3].metric("Max deviation from true curve", f"{two_phase_result.max_deviation_pct:.1f}%")
        st.caption(
            "Two-phase mode targets the settled daily exposure, taken here from the last 24 h of the run. "
            "AUC0-24 in the row above covers the first day, while the system is still filling up, so it sits below the target."
            + ("" if duration_h >= 48 else " Run at least 48 h to see the settled values.")
        )

    rows = result.rows
    st.subheader("6. PK concentration")
    if setup_drug_values:
        reference_curve = None
        if two_phase_active:
            reference_curve = _true_one_compartment_curve(
                target_css_mg_l,
                two_phase_result.reference_half_life_h,
                fos_interval,
                fos_duration,
                duration_h,
                dt_min,
            )
        st.pyplot(_plot_static(
            rows,
            [setup_drug_name],
            f"{setup_drug_name} central and extra concentration",
            include_extra=True,
            reference=reference_curve,
        ))
        if two_phase_active:
            st.caption(
                f"Dotted line: the true one-compartment curve for a {two_phase_result.reference_half_life_h:g} h half-life "
                "with the same dose timing and the same steady-state AUC. The extra line is the fixed fill concentration."
            )
    central_drugs = [drug.name for drug in drugs]
    if central_drugs:
        st.pyplot(_plot_static(rows, central_drugs, "Central concentration for loading/infusion drugs", include_extra=False))

    st.subheader("7. Preparation and weighing plan")
    _render_preparation_styles(st)
    prep_rows = _format_preparation_rows(result.summary["drug_preparation"])
    setup_prep, extra_replacement_prep, other_prep = _prep_rows_for_display(prep_rows, setup_drug_name, scenario)
    destination_cards = _preparation_destination_cards(prep_rows, result.summary, system, fos, duration_h)
    card_cols = st.columns(3)
    for index, card in enumerate(destination_cards):
        _render_preparation_card(card_cols[index], card)
    review_rows = _preparation_review_rows(prep_rows, result.summary, system, fos, duration_h)
    st.markdown("**Final preparation review**")
    st.caption("Use this table as the bench checklist: each row tells you which drug goes into which dosing part, with required amount, 10% extra when applicable, and the amount to weigh.")
    st.dataframe(
        review_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Drug": st.column_config.Column(width="small"),
            "Add into": st.column_config.Column(width="medium"),
            "Dosing part": st.column_config.Column(width="medium"),
            "Frequency": st.column_config.Column(width="medium"),
            "10% extra": st.column_config.Column(width="small"),
            "Note": st.column_config.Column(width="large"),
        },
    )

    st.markdown("**Calculation details**")
    if setup_prep:
        st.markdown(f"**{setup_drug_name} central dosing**")
        st.dataframe(
            setup_prep,
            width="stretch",
            hide_index=True,
            column_config={"Note": st.column_config.Column(width="large")},
        )
    if extra_replacement_prep:
        st.markdown(f"**{setup_drug_name} extra q24h replacement solution**")
        st.caption(
            "This is separate from the central q6h infusion. In pressure/filter-controlled q24h replacement, "
            "you prepare the extra-compartment fill only; Qextra is drug leaving that same fill during the interval."
        )
        replacement_summary = _replacement_solution_summary(system, fos, duration_h)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Replacement concentration", f"{replacement_summary['concentration_mg_ml']:.6f} mg/mL")
        metric_cols[1].metric(
            f"Prepared volume per q{fos.reservoir_replacement_interval_h:g}h",
            f"{replacement_summary['prepared_volume_per_interval_ml']:.1f} mL",
        )
        metric_cols[2].metric(
            f"Drug per q{fos.reservoir_replacement_interval_h:g}h",
            f"{replacement_summary['prepared_drug_per_interval_mg']:.3f} mg",
        )
        metric_cols[3].metric(
            f"{duration_h:g} h total +10%",
            f"{replacement_summary['total_drug_with_overfill_mg']:.1f} mg",
        )
        extra_solution_rows = _replacement_solution_rows(system, fos, duration_h)
        st.dataframe(
            extra_solution_rows,
            width="stretch",
            hide_index=True,
            column_config={"How calculated": st.column_config.Column(width="large")},
        )
    central_diluent_ci_rows = _central_diluent_reservoir_rows(result.summary, duration_h)
    if central_diluent_ci_rows:
        central_diluent_recipe = _central_diluent_reservoir_summary(result.summary, duration_h)
        st.markdown("**Central diluent q24h shared reservoir recipe**")
        st.caption(
            "Prepare one shared Diluent Central reservoir every 24 h. Loading dose is given directly into central; "
            "continuous-infusion maintenance drugs are mixed into this same central diluent volume and replaced q24h for stability."
        )
        recipe_cols = st.columns(4)
        recipe_cols[0].metric("Required volume q24h", central_diluent_recipe["volume_q24h"])
        recipe_cols[1].metric("10% extra volume q24h", central_diluent_recipe["extra_volume_q24h_10_percent"])
        recipe_cols[2].metric("Total to prepare q24h", central_diluent_recipe["prepared_volume_q24h"])
        recipe_cols[3].metric(f"Total to prepare {duration_h:g} h", central_diluent_recipe["prepared_volume_total"])
        st.caption(f"Number of q24h reservoirs = {central_diluent_recipe['replacements']}. The 10% extra is shown separately from the total prepared volume.")
        st.dataframe(
            central_diluent_ci_rows,
            width="stretch",
            hide_index=True,
            column_config={"Note": st.column_config.Column(width="large")},
        )
    if other_prep:
        st.markdown(f"**{_prep_group_title(other_prep)}**")
        st.dataframe(
            other_prep,
            width="stretch",
            hide_index=True,
            column_config={"Note": st.column_config.Column(width="large")},
        )

    st.subheader("8. What this means")
    st.markdown(_interpretation_text(scenario, setup_drug_name, setup_summary, setup_target_auc, result.summary, [drug.name for drug in drugs]))

    st.subheader("9. Equations")
    st.markdown(_equation_text(
        system,
        fos,
        setup_summary,
        setup_target_auc,
        scenario,
        duration_h,
        q_central_diluent,
        q_extra_diluent,
        target_system_half_life,
    ))

    st.subheader("10. HFIM Setup Assistant")
    st.caption("Ask the assistant about loading-dose targets, maintenance dosing, extra-compartment dilution, overflow loss, or whether the current HFIM setup is internally consistent.")
    agent_context = build_agent_context(
        system={
            "central_volume_ml": system.central_volume_ml,
            "extra_volume_ml": system.extra_volume_ml,
            "q_extra_to_central_ml_min": system.q_extra_to_central_ml_min,
            "q_extra_diluent_ml_min": system.q_extra_diluent_ml_min,
            "q_central_diluent_ml_min": system.q_central_diluent_ml_min,
            "scenario": scenario,
            "target_css_mg_l": target_css_mg_l,
            "target_cmax_mg_l": target_cmax_mg_l,
        },
        setup_drug_name=setup_drug_name,
        drug_inputs=drug_inputs,
        summary={name: value for name, value in result.summary.items() if name != "drug_preparation"},
    )
    _setup_assistant_panel(st, agent_context)

    if run_and_save:
        store = SimulationStore(Path("data") / "hfim-simulations.sqlite")
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = store.create_run(scenario, started_at, {
            "scenario": scenario,
            "duration_h": duration_h,
            "dt_min": dt_min,
            "extra_volume_ml": extra_volume_ml,
            "target_css_mg_l": target_css_mg_l,
            "target_cmax_mg_l": target_cmax_mg_l,
            "drugs": drug_inputs,
        })
        counts = store.upsert_timepoints(run_id, [
            {
                "time_min": row["time_min"],
                "drug": row["drug"],
                "central": row["central_mg_l"],
                "extra": row["extra_mg_l"],
                "central_volume_ml": row["central_volume_ml"],
                "extra_volume_ml": row["extra_volume_ml"],
            }
            for row in rows
        ])
        prep_counts = store.upsert_preparation_rows(run_id, result.summary["drug_preparation"])
        store.finish_run(run_id, "success", datetime.now(timezone.utc).isoformat(), f"timepoints={counts}; prep={prep_counts}")
        st.success(f"Saved run {run_id} to data/hfim-simulations.sqlite")


def _drug_input_panel(st, active_drug_count: int) -> dict[str, dict]:
    selected = {}
    for index in range(active_drug_count):
        default = _drug_default(index)
        with st.container(border=True):
            st.markdown(f"**Drug {index + 1}**")
            cols = st.columns(4)
            raw_name = cols[0].text_input("Drug name", value=default["name"], key=f"drug_name_{index}")
            name = _normalized_unique_drug_name(raw_name, index, set(selected))
            if raw_name.strip().lower() != name:
                st.caption(f"Internal simulation name: {name}")
            target_type = cols[1].selectbox(
                "Simulation target",
                ["Maintain concentration", "AUC0-24 exposure", "Cmax after loading dose"],
                index=["Maintain concentration", "AUC0-24 exposure", "Cmax after loading dose"].index(default["target_type"]),
                key=f"target_type_{index}",
            )
            target_label = {
                "Maintain concentration": "Target Css (mg/L)",
                "AUC0-24 exposure": "Target AUC0-24 (mg*h/L)",
                "Cmax after loading dose": "Target Cmax (mg/L)",
            }[target_type]
            target_value = cols[2].number_input(target_label, min_value=0.0, value=default["target_value"], step=1.0, key=f"target_value_{index}")
            half_life = cols[3].number_input("Half-life (h)", min_value=0.01, value=default["half_life"], step=0.05, key=f"half_life_{index}")
            st.caption(
                f"{name or 'This drug'}: the shortest active half-life sets the shared central-to-waste flow. "
                "If this value becomes the shortest half-life, the whole central system flow and maintenance drug amounts increase."
            )
            if target_type == "Cmax after loading dose":
                st.caption(
                    f"{name or 'This drug'}: loading dose is enabled by default and its loading target defaults "
                    "to this Cmax value directly (not a multiplier). Sustained accumulation over repeat maintenance "
                    "dosing is not solved automatically here - use the Section 3 solver for full Css/Cmax optimization."
                )

            dosing_cols = st.columns(3)
            if name == "fosfomycin":
                loading_dose = dosing_cols[0].checkbox("Loading dose", value=False, disabled=True, key=f"loading_dose_{index}")
                maintenance = dosing_cols[1].selectbox(
                    "Maintenance dosing",
                    ["intermittent infusion"],
                    key=f"maintenance_{index}",
                )
                dosing_frequency_h = dosing_cols[2].number_input(
                    "Dosing frequency (h)",
                    min_value=0.1,
                    value=default["dosing_frequency_h"],
                    step=1.0,
                    key=f"dosing_frequency_{index}",
                )
                dosing_mode = "q6h central + extra infusion"
            else:
                loading_dose = dosing_cols[0].checkbox(
                    "Loading dose",
                    value=_loading_dose_default_for_target_type(default["loading_dose"], target_type),
                    key=f"loading_dose_{index}",
                )
                maintenance = dosing_cols[1].selectbox(
                    "Maintenance dosing",
                    ["continuous infusion", "intermittent infusion", "no maintenance"],
                    index=["continuous infusion", "intermittent infusion", "no maintenance"].index(default["maintenance"]),
                    key=f"maintenance_{index}",
                )
                dosing_frequency_h = dosing_cols[2].number_input(
                    "Dosing frequency (h)",
                    min_value=0.0,
                    value=default["dosing_frequency_h"] if maintenance == "intermittent infusion" else 0.0,
                    step=1.0,
                    disabled=maintenance != "intermittent infusion",
                    help="Only intermittent infusion uses a q-hour dosing interval.",
                    key=f"dosing_frequency_{index}",
                )
                dosing_mode = _dosing_mode_from_controls(loading_dose, maintenance)
            loading_target = 0.0
            loading_duration_h = 0.0
            maintenance_duration_h = default["maintenance_duration_h"]
            detail_cols = st.columns(4)
            if loading_dose:
                loading_target = detail_cols[0].number_input(
                    "Loading target (mg/L)",
                    min_value=0.0,
                    value=_loading_target_default_mg_l(target_value, target_type, default["loading_target_multiplier"]),
                    step=1.0,
                    key=f"loading_target_{index}",
                )
                loading_duration_h = detail_cols[1].number_input(
                    "Loading infusion duration (h)",
                    min_value=0.0,
                    value=default["loading_duration_h"],
                    step=0.25,
                    key=f"loading_duration_{index}",
                )
                loading_volume_ml = detail_cols[2].number_input(
                    "Loading dose volume (mL)",
                    min_value=0.01,
                    value=default["loading_volume_ml"],
                    step=0.5,
                    help="Volume used to dissolve the loading dose. This determines stock concentration and pump rate, not the target mg dose.",
                    key=f"loading_volume_{index}",
                )
            else:
                loading_volume_ml = default["loading_volume_ml"]
            if maintenance == "intermittent infusion":
                maintenance_duration_h = detail_cols[3].number_input(
                    "Maintenance infusion duration (h)",
                    min_value=0.01,
                    value=default["maintenance_duration_h"],
                    step=0.25,
                    key=f"maintenance_duration_{index}",
                )
            selected[name] = {
                "target_type": target_type,
                "target_value": target_value,
                "target_concentration_mg_l": _target_to_concentration(target_type, target_value),
                "half_life_h": half_life,
                "dosing_mode": dosing_mode,
                "loading_dose": loading_dose,
                "maintenance": maintenance,
                "dosing_frequency_h": dosing_frequency_h,
                "loading_target_concentration_mg_l": loading_target if loading_dose else None,
                "loading_duration_h": loading_duration_h,
                "loading_volume_ml": loading_volume_ml,
                "maintenance_duration_h": maintenance_duration_h,
            }
    return selected


def _normalized_unique_drug_name(raw_name: str, index: int, existing_names: set[str]) -> str:
    base = raw_name.strip().lower() or f"drug{index + 1}"
    candidate = base
    suffix = 2
    while candidate in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _scenario_from_label(label: str) -> str:
    if label.startswith("q24h"):
        return "q24_replacement"
    return "overflow"


def _is_auto_flow_mode(flow_mode: str) -> bool:
    return flow_mode.startswith("Auto")


def _central_diluent_default_for_flow_mode(flow_mode: str, auto_central_flow: float, q_extra_to_central: float = 0.0) -> float:
    if not _is_auto_flow_mode(flow_mode):
        return 0.65
    return max(0.0, auto_central_flow - q_extra_to_central)


def _shared_central_half_life_from_widget_state(active_drug_count: int, state) -> float:
    half_lives = []
    for index in range(active_drug_count):
        default = _drug_default(index)
        value = _state_get(state, f"half_life_{index}", default["half_life"])
        try:
            half_life = float(value)
        except (TypeError, ValueError):
            half_life = float(default["half_life"])
        if half_life > 0:
            half_lives.append(half_life)
    return min(half_lives) if half_lives else 1.0


def _state_get(state, key: str, default):
    getter = getattr(state, "get", None)
    if getter is not None:
        return getter(key, default)
    try:
        return state[key]
    except (KeyError, TypeError):
        return default


def _extra_diluent_balance_warning(
    scenario: str,
    q_extra_to_central_ml_min: float,
    q_extra_diluent_ml_min: float,
    extra_volume_ml: float,
) -> str | None:
    if scenario != "overflow":
        return None
    imbalance = q_extra_diluent_ml_min - q_extra_to_central_ml_min
    if abs(imbalance) <= 1e-6:
        return None
    direction = "less than" if imbalance < 0 else "more than"
    drift = "drain" if imbalance < 0 else "overfill"
    return (
        f"Extra diluent ({q_extra_diluent_ml_min:.3f} mL/min) is {direction} extra-to-central "
        f"({q_extra_to_central_ml_min:.3f} mL/min) by {abs(imbalance):.3f} mL/min. This flow is not used in "
        f"the PK calculation - the simulator assumes the extra volume stays fixed at {extra_volume_ml:g} mL. "
        f"In the real system this mismatch will {drift} the extra compartment over time; match extra diluent "
        "to extra-to-central (plus overflow demand while dosing) to keep the physical volume stable."
    )


def _qextra_default_for_scenario(scenario: str, flow_mode: str, auto_extra_flow: float) -> float:
    if scenario == "q24_replacement":
        return 0.167
    return auto_extra_flow if _is_auto_flow_mode(flow_mode) else 0.921


def _flow_widget_key(prefix: str, scenario: str, auto_flow_mode: bool, volume_or_space: float, target_half_life_h: float) -> str:
    if not auto_flow_mode:
        return f"{prefix}_{scenario}_manual"
    return f"{prefix}_{scenario}_auto_{volume_or_space:.3f}_{target_half_life_h:.3f}"


_SLOW_LINE_SAME_STOCK = "Own syringe at the same concentration as the main infusion (lower pump rate)"
_SLOW_LINE_SEPARATE_STOCK = "Own syringe at a weaker concentration, at a pump rate I set"


def _render_two_phase_solver_panel(st, solution) -> None:
    top = st.columns(4)
    top[0].metric("Solved extra replacement", f"{solution.extra_replacement_concentration_mg_ml:.6f} mg/mL")
    if solution.slow_duration_min > 0:
        top[1].metric(
            "Slow line",
            f"{solution.slow_infusion_ml_min * 60:.3f} mL/h",
            f"for {solution.slow_duration_min / 60:g} h after each infusion",
            delta_color="off",
        )
        top[2].metric("Slow line syringe", f"{solution.slow_stock_mg_ml:.6f} mg/mL")
    else:
        top[1].metric("Slow line", "not needed")
        top[2].metric("Slow line syringe", "-")
    top[3].metric(
        "Max deviation from true curve",
        f"{solution.max_deviation_pct:.1f}%",
        f"limit {solution.deviation_limit_pct:g}%",
        delta_color="off",
    )
    bottom = st.columns(4)
    bottom[0].metric("Steady-state Cavg", f"{solution.predicted_cavg_mg_l:.1f} mg/L")
    bottom[1].metric(
        "Steady-state Cmax",
        f"{solution.predicted_cmax_mg_l:.1f} mg/L",
        f"true curve {solution.reference_cmax_mg_l:.1f}",
        delta_color="off",
    )
    bottom[2].metric(
        "Steady-state Cmin",
        f"{solution.predicted_cmin_mg_l:.1f} mg/L",
        f"true curve {solution.reference_cmin_mg_l:.1f}",
        delta_color="off",
    )
    bottom[3].metric("Half-life being followed", f"{solution.reference_half_life_h:g} h")
    if solution.feasible:
        st.success(solution.message)
    else:
        st.warning(solution.message)
    st.caption(
        "Max deviation is the largest gap, anywhere in one dosing interval at steady state, between this setup's "
        "central concentration and the true one-compartment curve at the drug's own half-life. "
        f"Without the slow line the same setup would be {solution.single_phase_max_deviation_pct:.1f}% off. "
        "T>MIC can still differ by more than this figure when the MIC sits on a flat part of the curve."
    )


def _true_one_compartment_curve(
    cavg_mg_l: float,
    half_life_h: float,
    interval_min: float,
    infusion_min: float,
    duration_h: float,
    dt_min: float,
) -> tuple[list[float], list[float], str]:
    """Concentration from t=0 of a true one-compartment drug on the same dose timing (for the overlay)."""
    k = math.log(2) / (half_life_h * 60)
    duration = min(infusion_min, interval_min)
    plateau = cavg_mg_l * interval_min / duration
    decay = math.exp(-k * dt_min)
    times, values = [], []
    concentration = 0.0
    for step in range(int(round(duration_h * 60 / dt_min)) + 1):
        time_min = step * dt_min
        times.append(time_min / 60)
        values.append(concentration)
        infusing = time_min % interval_min < duration
        concentration = concentration * decay + (plateau * (1 - decay) if infusing else 0.0)
    return times, values, f"true {half_life_h:g} h one-compartment curve"


def _extra_setup_help_text(scenario: str) -> str:
    if scenario == "q24_replacement":
        return (
            "q24h replacement mode assumes the extra compartment is filled with drug at t=0 and fully replaced every 24 h. "
            "In pressure/filter-controlled mode, Qextra is liquid leaving that same fill and is not prepared as a second solution."
        )
    if scenario == "overflow":
        return (
            "Overflow mode uses intermittent extra-compartment drug infusion. Extra volume is kept fixed by an overflow outflow line, "
            "so some drug can be lost to extra waste during dosing."
        )
    return "Overflow mode uses intermittent extra-compartment drug infusion with a separate overflow outflow line."


def _optimization_guidance_text(setup_drug_name: str, scenario: str) -> str:
    if scenario == "q24_replacement":
        return (
            f"**Optimization recommendation for {setup_drug_name}:** keep the physical HFIM settings fixed first, then let the solver adjust "
            f"the {setup_drug_name} central stock and q24h extra replacement concentration. Use this strategy for the drug whose central profile "
            "needs both AUC/Cavg control and peak-shape control. For imipenem/relebactam, loading dose plus maintenance infusion is usually the cleaner setup."
        )
    return (
        f"**Optimization recommendation for {setup_drug_name}:** use overflow only when you want intermittent extra dosing without changing the pump during each dose. "
        "It is easier operationally, but drug can leave through the extra overflow line."
    )


def _drug_default(index: int) -> dict:
    defaults = [
        {
            "name": "fosfomycin",
            "target_type": "AUC0-24 exposure",
            "target_value": 3600.0,
            "half_life": 3.0,
            "loading_dose": False,
            "maintenance": "intermittent infusion",
            "dosing_frequency_h": 6.0,
            "loading_target_multiplier": 1.0,
            "loading_duration_h": 0.0,
            "loading_volume_ml": 5.0,
            "maintenance_duration_h": 1.0,
        },
        {
            "name": "imipenem",
            "target_type": "Maintain concentration",
            "target_value": 9.0,
            "half_life": 1.25,
            "loading_dose": True,
            "maintenance": "continuous infusion",
            "dosing_frequency_h": 0.0,
            "loading_target_multiplier": 2.0,
            "loading_duration_h": 0.5,
            "loading_volume_ml": 5.0,
            "maintenance_duration_h": 1.0,
        },
        {
            "name": "relebactam",
            "target_type": "Maintain concentration",
            "target_value": 6.0,
            "half_life": 1.25,
            "loading_dose": True,
            "maintenance": "continuous infusion",
            "dosing_frequency_h": 0.0,
            "loading_target_multiplier": 2.0,
            "loading_duration_h": 0.5,
            "loading_volume_ml": 5.0,
            "maintenance_duration_h": 1.0,
        },
    ]
    if index < len(defaults):
        return defaults[index]
    return {
        "name": f"drug{index + 1}",
        "target_type": "Maintain concentration",
        "target_value": 1.0,
        "half_life": 1.0,
        "loading_dose": False,
        "maintenance": "no maintenance",
        "dosing_frequency_h": 0.0,
        "loading_target_multiplier": 2.0,
        "loading_duration_h": 0.5,
        "loading_volume_ml": 5.0,
        "maintenance_duration_h": 1.0,
    }


def _dosing_mode_from_controls(loading_dose: bool, maintenance: str) -> str:
    has_continuous = maintenance == "continuous infusion"
    has_intermit = maintenance == "intermittent infusion"
    if loading_dose and has_continuous:
        return "loading dose + continuous infusion"
    if loading_dose and has_intermit:
        return "loading dose + intermittent infusion"
    if has_continuous:
        return "continuous infusion only"
    if has_intermit:
        return "intermittent infusion only"
    if loading_dose:
        return "loading dose only"
    return "no dose"


def _target_to_concentration(target_type: str, target_value: float) -> float:
    if target_type == "AUC0-24 exposure":
        return target_value / 24
    return target_value


def _loading_dose_default_for_target_type(base_default: bool, target_type: str) -> bool:
    return base_default or target_type == "Cmax after loading dose"


def _loading_target_default_mg_l(target_value: float, target_type: str, multiplier: float) -> float:
    if target_type == "Cmax after loading dose":
        return target_value
    return target_value * multiplier


def _setup_overview_rows(
    central_bottle_ml: float,
    cartridge_ml: float,
    extra_volume_ml: float,
    q_extra_to_central: float,
    q_extra_diluent: float,
    q_central_diluent: float,
    scenario: str,
    fos: FosfomycinConfig,
) -> list[dict]:
    central_volume = central_bottle_ml + cartridge_ml
    total_central_outflow = q_extra_to_central + q_central_diluent
    setup_average_flow = average_intermittent_rate(
        fos.central_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min
    )
    ci_effective_outflow = total_central_outflow + setup_average_flow
    central_half_life = half_life_for_flow(central_volume, total_central_outflow) if total_central_outflow > 0 else None
    extra_half_life = half_life_for_flow(extra_volume_ml, q_extra_to_central) if q_extra_to_central > 0 else None
    if scenario == "q24_replacement":
        max_single_fill_qextra = extra_volume_ml / (fos.reservoir_replacement_interval_h * 60)
        extra_volume_use = "q24 replacement volume; changing this does not automatically change Qextra"
        extra_transfer_use = (
            "independent physical transfer rate from the fixed-concentration extra compartment into central; "
            f"one-fill limit at this volume is {max_single_fill_qextra:.3f} mL/min"
        )
    else:
        extra_volume_use = "larger extra volume slows extra washout if flow is unchanged"
        extra_transfer_use = f"sets extra washout half-life ≈ {_fmt_optional(extra_half_life)} h"
    rows = [
        {"Part": "Central effective volume", "Current value": f"{central_volume:g} mL", "How to use": "central bottle + cartridge; larger volume needs higher flow for same half-life"},
        {"Part": "Central diluent", "Current value": f"{q_central_diluent:g} mL/min", "How to use": "shared q24h reservoir flow; CI drug concentration is calculated from this pump rate"},
        {"Part": "Setup central pump average", "Current value": f"{setup_average_flow:g} mL/min", "How to use": "time-weighted average of the selected drug central q-dose pump; included in CI drug washout"},
        {"Part": "CI effective outflow", "Current value": f"{ci_effective_outflow:g} mL/min", "How to use": "Qextra + central diluent + setup central pump average; used to calculate imipenem/relebactam CI mg/h"},
        {"Part": "Extra volume", "Current value": f"{extra_volume_ml:g} mL", "How to use": extra_volume_use},
        {"Part": "Extra to central", "Current value": f"{q_extra_to_central:g} mL/min", "How to use": extra_transfer_use},
        {"Part": "Baseline waste", "Current value": f"{total_central_outflow:g} mL/min", "How to use": f"baseline central output from Qextra + central diluent; shared central half-life ≈ {_fmt_optional(central_half_life)} h"},
    ]
    if scenario == "q24_replacement":
        rows.append({
            "Part": f"Extra q{fos.reservoir_replacement_interval_h:g}h replacement",
            "Current value": f"{fos.extra_stock_mg_ml:g} mg/mL",
            "How to use": "fixed concentration; check transfer-demand volume before assuming one compartment fill is enough",
        })
    else:
        rows.insert(4, {"Part": "Extra diluent", "Current value": f"{q_extra_diluent:g} mL/min", "How to use": "usually match this to extra-to-central flow to keep the baseline extra volume fixed"})
    if scenario == "overflow":
        rows.append({
            "Part": "Extra overflow outflow",
            "Current value": f"{fos.extra_infusion_ml_min:g} mL/min while extra dosing is on",
            "How to use": "keeps extra volume fixed during extra drug infusion, but carries some drug to waste",
        })
    return rows


def _injection_plan_rows(drug_inputs: dict[str, dict], fos: FosfomycinConfig, scenario: str, setup_drug_name: str) -> list[dict]:
    rows = []
    for name, values in drug_inputs.items():
        if name == setup_drug_name:
            rows.append({
                "Drug": name,
                "Target": f"{values['target_type']} = {values['target_value']:g}",
                "Half-life": f"{values['half_life_h']:g} h",
                "Dosing plan": "central intermittent infusion",
                "Physical setting": (
                    f"{fos.central_infusion_ml_min * fos.infusion_duration_min:g} mL over "
                    f"{fos.infusion_duration_min / 60:g} h, q{fos.dosing_interval_min / 60:g}h"
                ),
            })
            if fos.has_slow_line:
                slow_start_h = fos.infusion_duration_min / 60
                rows.append({
                    "Drug": name,
                    "Target": "curve shape (two-phase)",
                    "Half-life": f"{values['half_life_h']:g} h",
                    "Dosing plan": "central slow line (pump 2, own syringe)",
                    "Physical setting": (
                        f"{fos.slow_dose_volume_ml:.3f} mL at {fos.slow_infusion_ml_min * 60:.3f} mL/h, "
                        f"h {slow_start_h:g}-{slow_start_h + fos.slow_duration_min / 60:g} of each q{fos.dosing_interval_min / 60:g}h"
                    ),
                })
            if scenario != "q24_replacement":
                rows.append({
                    "Drug": name,
                    "Target": "extra support",
                    "Half-life": f"{values['half_life_h']:g} h",
                    "Dosing plan": _extra_dosing_plan_label(scenario),
                    "Physical setting": (
                        f"{fos.extra_infusion_ml_min * fos.infusion_duration_min:g} mL over "
                        f"{fos.infusion_duration_min / 60:g} h, q{fos.dosing_interval_min / 60:g}h"
                    ),
                })
            else:
                rows.append({
                    "Drug": name,
                    "Target": "q24h extra replacement",
                    "Half-life": f"{values['half_life_h']:g} h",
                    "Dosing plan": f"full replacement q{fos.reservoir_replacement_interval_h:g}h",
                    "Physical setting": f"{fos.extra_stock_mg_ml:g} mg/mL in full extra volume",
                })
        else:
            loading_text = "loading dose: no; "
            if values["loading_dose"]:
                loading_volume_ml = values.get("loading_volume_ml", 0.0)
                loading_rate_ml_h = loading_volume_ml / values["loading_duration_h"] if values["loading_duration_h"] else 0.0
                loading_text = (
                    f"loading target: {values['loading_target_concentration_mg_l']:g} mg/L; "
                    f"{loading_volume_ml:g} mL over {values['loading_duration_h']:g} h "
                    f"({loading_rate_ml_h:g} mL/h); "
                )
            rows.append({
                "Drug": name,
                "Target": f"{values['target_type']} = {values['target_value']:g}",
                "Half-life": f"{values['half_life_h']:g} h",
                "Dosing plan": loading_text + f"maintenance: {values['maintenance']}",
                "Physical setting": "calculated from loading/Css target and the shared central-to-waste flow",
            })
    return rows


def _fmt_optional(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.2f}"


def _extra_dosing_plan_label(scenario: str) -> str:
    if scenario == "overflow":
        return "extra intermittent infusion with overflow"
    return "no extra dosing"


# ---------------------------------------------------------------------------
# HFIM apparatus schematic
#
# One integrated diagram per page: vessels, tubing, flow markers and dosing
# instructions all live on the same canvas, so there is no separate recipe
# panel to cross-reference. Semantic colours are shared with Section 7's cards:
#   green = volume, blue = concentration, amber = flow / dose, ink = names.
# ---------------------------------------------------------------------------

_APPARATUS = {
    "ink": "#12203a",
    "muted": "#5b6b85",
    "volume": "#0f7a52",
    "concentration": "#2456c7",
    "rate": "#c2670a",
    "inject": "#c2359b",
    "tube": "#9fd9e8",
    "tube_dark": "#5d9fb5",
    "glass": "#e8f6fb",
    "glass_edge": "#7fb3c6",
    "glass_shine": "#fbfeff",
    "cap": "#4c56b0",
    "cap_dark": "#333c8c",
    "liquid": "#22a06b",
    "liquid_fill": "#9fdce3",
    "cartridge": "#2f6fe0",
    "cartridge_dark": "#1d4ba8",
    "cartridge_light": "#8fb6f2",
    "ecs_fill": "#dbe9fb",
    "ecs_edge": "#b9d2f3",
    "ecs_text": "#2b4a7d",
    "marker": "#f5a524",
    "marker_edge": "#c2670a",
    "pump_body": "#e7ecf4",
    "pump_edge": "#5b6b85",
    "paper": "#ffffff",
    "panel": "#f7fafc",
    "panel_edge": "#dbe4ef",
}


def _apparatus_bottle(ax, x, y, w=0.92, h=1.45, fill_level=0.55, label=None, volume=None, stir_bar=False, zorder=3):
    """A lab media bottle: squared body, tapered shoulder, short neck and screw cap.

    Built from a real bottle silhouette (body -> shoulder -> neck -> cap) with a drop shadow and a
    layered liquid fill, rather than a plain rounded rectangle, so the diagram reads as apparatus.
    """
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
    from matplotlib.patheffects import withSimplePatchShadow

    c = _APPARATUS
    body_h = h * 0.68
    body_y = y - h / 2
    body_top = body_y + body_h
    shoulder_h = h * 0.11
    neck_w, neck_h = w * 0.30, h * 0.07
    cap_w, cap_h = w * 0.40, h * 0.12
    neck_top = body_top + shoulder_h + neck_h
    shadow = withSimplePatchShadow(offset=(1.7, -1.7), shadow_rgbFace="#16243c", alpha=0.14)

    body = FancyBboxPatch(
        (x - w / 2, body_y), w, body_h,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.15, edgecolor=c["glass_edge"], facecolor=c["glass"], zorder=zorder,
    )
    body.set_path_effects([shadow])
    ax.add_patch(body)

    # Layered fill: darker at the base, lighter at the surface, which reads as liquid depth.
    usable = body_h - 0.12
    liquid_h = max(usable * fill_level, 0.07)
    for frac, alpha in ((1.00, 0.55), (0.62, 0.30), (0.28, 0.28)):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.055, body_y + 0.055), w - 0.11, max(liquid_h * frac, 0.05),
            boxstyle="round,pad=0,rounding_size=0.09",
            linewidth=0, facecolor=c["liquid_fill"], alpha=alpha, zorder=zorder + 0.1,
        ))
    ax.plot(
        [x - w / 2 + 0.075, x + w / 2 - 0.075], [body_y + 0.055 + liquid_h] * 2,
        color=c["liquid"], linewidth=1.6, solid_capstyle="round", zorder=zorder + 0.2,
    )

    ax.add_patch(Polygon(
        [(x - w / 2 + 0.02, body_top), (x + w / 2 - 0.02, body_top),
         (x + neck_w / 2, body_top + shoulder_h), (x - neck_w / 2, body_top + shoulder_h)],
        closed=True, linewidth=1.1, edgecolor=c["glass_edge"], facecolor=c["glass"], zorder=zorder + 0.15,
    ))
    ax.add_patch(Rectangle(
        (x - neck_w / 2, body_top + shoulder_h - 0.01), neck_w, neck_h + 0.02,
        linewidth=1.1, edgecolor=c["glass_edge"], facecolor=c["glass"], zorder=zorder + 0.15,
    ))
    cap = FancyBboxPatch(
        (x - cap_w / 2, neck_top - 0.01), cap_w, cap_h,
        boxstyle="round,pad=0,rounding_size=0.045",
        linewidth=1.0, edgecolor=c["cap_dark"], facecolor=c["cap"], zorder=zorder + 0.3,
    )
    cap.set_path_effects([shadow])
    ax.add_patch(cap)
    for i in range(4):
        rx = x - cap_w / 2 + cap_w * (0.2 + i * 0.2)
        ax.plot([rx, rx], [neck_top + 0.015, neck_top + cap_h - 0.035], color=c["cap_dark"], linewidth=0.7, alpha=0.75, zorder=zorder + 0.4)

    ax.plot(
        [x - w / 2 + 0.14, x - w / 2 + 0.14], [body_y + 0.20, body_top - 0.18],
        color=c["glass_shine"], linewidth=2.4, alpha=0.85, solid_capstyle="round", zorder=zorder + 0.25,
    )
    if stir_bar:
        # Inside the vessel rather than as a separate box underneath, which would collide with the
        # vessel's own caption.
        ax.add_patch(FancyBboxPatch(
            (x - w * 0.19, body_y + 0.10), w * 0.38, 0.09,
            boxstyle="round,pad=0,rounding_size=0.035",
            linewidth=0.8, edgecolor="#5b6b85", facecolor="#c3ccd8", zorder=zorder + 0.5,
        ))
        ax.text(x, body_y + 0.31, "stirred", ha="center", va="bottom", fontsize=6.6, style="italic", color=c["muted"], zorder=zorder + 0.5)

    text_y = body_y - 0.14
    if label:
        ax.text(x, text_y, label, ha="center", va="top", fontsize=9.6, weight="bold", color=c["ink"], zorder=8)
        text_y -= 0.25
    if volume:
        ax.text(x, text_y, volume, ha="center", va="top", fontsize=8.8, color=c["volume"], zorder=8)
    return x, neck_top + cap_h


def _apparatus_cartridge(ax, x, y, w=3.0, h=0.52, label=None, volume=None, label_side="right", zorder=4):
    """Hollow-fibre cartridge drawn as a horizontal barrel with end ports and visible fibres."""
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    from matplotlib.patheffects import withSimplePatchShadow

    c = _APPARATUS
    # Outer shell = extracapillary space, where the bacteria sit; the fibre bundle inside is the
    # intracapillary space the drug is pumped through. Showing both is the point of the cartridge.
    shell = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.22",
        linewidth=1.3,
        edgecolor=c["cartridge_dark"],
        facecolor=c["ecs_fill"],
        zorder=zorder,
    )
    shell.set_path_effects([withSimplePatchShadow(offset=(1.8, -1.8), shadow_rgbFace="#16243c", alpha=0.16)])
    ax.add_patch(shell)
    lumen_h = h * 0.52
    ax.add_patch(FancyBboxPatch(
        (x - w * 0.40, y - lumen_h / 2),
        w * 0.80,
        lumen_h,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=0.9,
        edgecolor=c["cartridge_dark"],
        facecolor=c["cartridge"],
        zorder=zorder + 0.1,
    ))
    for i in range(16):
        fx = x - w * 0.375 + i * (w * 0.75 / 15)
        ax.plot([fx, fx], [y - lumen_h * 0.40, y + lumen_h * 0.40], color=c["cartridge_light"], linewidth=0.7, alpha=0.85, zorder=zorder + 0.2)
    ax.add_patch(Rectangle((x - w * 0.40, y + lumen_h * 0.10), w * 0.80, lumen_h * 0.12, facecolor=c["cartridge_light"], edgecolor="none", alpha=0.5, zorder=zorder + 0.25))
    for side in (-1, 1):
        ax.add_patch(Rectangle((x + side * w * 0.415 - 0.035, y - h * 0.30), 0.07, h * 0.60, facecolor=c["cartridge_light"], edgecolor=c["cartridge_dark"], linewidth=0.8, zorder=zorder + 0.15))
        ax.add_patch(Circle((x + side * (w / 2 + 0.07), y), radius=0.11, facecolor=c["cartridge_light"], edgecolor=c["cartridge_dark"], linewidth=1.0, zorder=zorder + 0.2))
    # Sampling port on the ECS: where CFU samples are actually drawn from.
    ax.add_patch(Rectangle((x - 0.055, y + h / 2), 0.11, 0.17, facecolor=c["ecs_edge"], edgecolor=c["cartridge_dark"], linewidth=0.8, zorder=zorder + 0.2))
    ax.add_patch(Circle((x, y + h / 2 + 0.20), radius=0.075, facecolor=c["inject"], edgecolor="none", zorder=zorder + 0.3))
    ax.text(x + 0.16, y + h / 2 + 0.20, "sampling port", ha="left", va="center", fontsize=6.6, style="italic", color=c["muted"], zorder=zorder + 0.3)
    if label_side == "right":
        # Beside the barrel, not above it, so a tall page title never collides with the label.
        # The ECS caption also lives out here: inside the shell the fibre bundle covers it.
        text_x = x + w / 2 + 0.30
        if label:
            ax.text(text_x, y + 0.24, label, ha="left", va="bottom", fontsize=9.6, weight="bold", color=c["ink"], zorder=8)
        if volume:
            ax.text(text_x, y + 0.04, volume, ha="left", va="bottom", fontsize=8.8, color=c["volume"], zorder=8)
        ax.text(text_x, y - 0.14, "bacteria held in ECS", ha="left", va="top", fontsize=7.4, style="italic", color=c["ecs_text"], zorder=8)
        return
    text_y = y + h / 2 + 0.20
    if label:
        ax.text(x, text_y, label, ha="center", va="bottom", fontsize=9.6, weight="bold", color=c["ink"], zorder=8)
        text_y += 0.24
    if volume:
        ax.text(x, text_y, volume, ha="center", va="bottom", fontsize=8.8, color=c["volume"], zorder=8)


_INJECTION_BAND_LINE_HEIGHT = 0.21


def _injection_band_row_height(max_lines: int) -> float:
    return 0.26 + max_lines * _INJECTION_BAND_LINE_HEIGHT + 0.22


def _apparatus_injection_band(ax, groups, x_start, y_top, width, columns=3, max_lines=8, header=None, max_rows=1):
    """Lay dosing instructions out as a row of per-drug blocks under the apparatus.

    Keeping them in their own band means the block list grows sideways with drug count instead of
    running down into the vessel captions. The header names the destination, so a block sitting
    below the waste bottle is not misread as belonging to it.
    """
    block_y = y_top
    if header:
        ax.text(x_start, y_top, header, ha="left", va="top", fontsize=9.4, weight="bold", color=_APPARATUS["inject"])
        block_y = y_top - 0.32
    capacity = columns * max_rows
    row_height = _injection_band_row_height(max_lines)
    for index, (title, lines) in enumerate(groups[:capacity]):
        row, column = divmod(index, columns)
        _apparatus_injection_block(
            ax, x_start + column * width, block_y - row * row_height, title, lines, max_lines=max_lines
        )
    if len(groups) > capacity:
        ax.text(
            x_start,
            block_y - (max_rows - 1) * row_height - 0.26 - max_lines * 0.21,
            f"+{len(groups) - capacity} more drugs (see Section 7)",
            ha="left",
            va="top",
            fontsize=7.6,
            color=_APPARATUS["muted"],
        )


TUBE_FAST = 5.2
TUBE_SLOW = 2.6


def _apparatus_tube(ax, points, zorder=2, linewidth=TUBE_SLOW):
    """Tubing drawn as an outer casing plus a lighter inner lumen.

    Line width encodes flow magnitude: the cartridge recirculation loop moves roughly two orders of
    magnitude more volume than the PK diluent/waste lines, and drawing every tube identically hides
    that. Use TUBE_FAST for the recirculation loop and TUBE_SLOW for the PK flows.

    Two points draw a straight run. More than two points are read as a quadratic Bezier chain -
    after the start point they alternate (control, end), so the count must be odd. Keeping the
    rule explicit avoids silently mis-rendering a route when a caller adds a waypoint.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    c = _APPARATUS
    if len(points) < 2:
        raise ValueError("a tube needs at least two points")
    if len(points) == 2:
        codes = [Path.MOVETO, Path.LINETO]
    elif len(points) % 2 == 1:
        codes = [Path.MOVETO] + [Path.CURVE3] * (len(points) - 1)
    else:
        raise ValueError("a curved tube needs an odd number of points: start then (control, end) pairs")
    path = Path(list(points), codes)
    for width, color in ((linewidth + 1.1, c["tube_dark"]), (linewidth, c["tube"])):
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, linewidth=width, capstyle="round", joinstyle="round", zorder=zorder))
        zorder += 0.1


def _apparatus_flow_arrow(ax, x, y, direction, size=0.13, zorder=6.5):
    """Small solid arrowhead sitting on a tube to show which way the fluid moves."""
    from matplotlib.patches import Polygon

    dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}[direction]
    px, py = -dy, dx
    tip = (x + dx * size, y + dy * size)
    left = (x - dx * size * 0.5 + px * size * 0.78, y - dy * size * 0.5 + py * size * 0.78)
    right = (x - dx * size * 0.5 - px * size * 0.78, y - dy * size * 0.5 - py * size * 0.78)
    ax.add_patch(Polygon([tip, left, right], closed=True, facecolor=_APPARATUS["rate"], edgecolor="none", zorder=zorder))


def _apparatus_pump(ax, x, y, rate=None, radius=0.30, zorder=7, label_offset=(0.0, -0.50), label_ha="center"):
    """Peristaltic pump head: housing ring with three rollers, the way HFIM circuits are drawn.

    Real hollow-fibre rigs move fluid with roller pumps, so drawing the pump rather than an abstract
    arrow box makes the diagram read as apparatus instead of a flow chart.
    """
    import math as _math
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withSimplePatchShadow

    c = _APPARATUS
    housing = Circle((x, y), radius, facecolor=c["pump_body"], edgecolor=c["pump_edge"], linewidth=1.3, zorder=zorder)
    housing.set_path_effects([withSimplePatchShadow(offset=(1.4, -1.4), shadow_rgbFace="#16243c", alpha=0.15)])
    ax.add_patch(housing)
    ax.add_patch(Circle((x, y), radius * 0.70, facecolor=c["paper"], edgecolor=c["pump_edge"], linewidth=0.8, zorder=zorder + 0.1))
    for index in range(3):
        angle = _math.radians(index * 120 + 30)
        ax.add_patch(Circle(
            (x + radius * 0.46 * _math.cos(angle), y + radius * 0.46 * _math.sin(angle)),
            radius * 0.19,
            facecolor=c["marker"],
            edgecolor=c["marker_edge"],
            linewidth=0.7,
            zorder=zorder + 0.2,
        ))
    ax.add_patch(Circle((x, y), radius * 0.13, facecolor=c["pump_edge"], edgecolor="none", zorder=zorder + 0.3))
    if rate:
        ax.text(
            x + label_offset[0],
            y + label_offset[1],
            rate,
            ha=label_ha,
            va="center",
            fontsize=8.0,
            color=c["rate"],
            zorder=zorder + 1,
            bbox={"facecolor": c["paper"], "edgecolor": "none", "pad": 1.4, "alpha": 0.92},
        )


def _apparatus_injection_port(ax, x, y, angle_deg=0.0, scale=1.0, zorder=6):
    """Septum injection port on a vessel shoulder, drawn in the injection accent colour."""
    from matplotlib.patches import Circle, FancyBboxPatch
    from matplotlib.transforms import Affine2D

    c = _APPARATUS
    transform = Affine2D().rotate_deg(angle_deg).translate(x, y) + ax.transData
    ax.add_patch(FancyBboxPatch(
        (0.0, -0.09 * scale),
        0.20 * scale,
        0.18 * scale,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=0.9,
        edgecolor=c["inject"],
        facecolor="#fbe6f4",
        transform=transform,
        zorder=zorder,
    ))
    ax.add_patch(Circle((0.20 * scale, 0.0), 0.065 * scale, facecolor=c["inject"], edgecolor="none", transform=transform, zorder=zorder + 0.1))


def _apparatus_syringe(ax, tip, angle_deg=0.0, scale=1.0, zorder=7):
    """Syringe anchored by its needle TIP, with the barrel trailing back along angle_deg.

    Anchoring on the tip means callers place the needle exactly on the injection port instead of
    positioning the barrel and hoping the needle lands somewhere sensible.
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D

    transform = Affine2D().rotate_deg(angle_deg).translate(*tip) + ax.transData
    needle = 0.30 * scale
    hub_w = 0.10 * scale
    barrel_w, barrel_h = 0.62 * scale, 0.24 * scale
    barrel_x = -(needle + hub_w + barrel_w)

    ax.plot([0.0, -needle], [0.0, 0.0], color="#78829a", linewidth=1.3, solid_capstyle="round", transform=transform, zorder=zorder)
    ax.add_patch(FancyBboxPatch(
        (-(needle + hub_w), -0.055 * scale), hub_w, 0.11 * scale,
        boxstyle="round,pad=0,rounding_size=0.02", linewidth=0.8,
        edgecolor="#78829a", facecolor="#b9c2d0", transform=transform, zorder=zorder,
    ))
    ax.add_patch(FancyBboxPatch(
        (barrel_x, -barrel_h / 2), barrel_w, barrel_h,
        boxstyle="round,pad=0,rounding_size=0.035", linewidth=0.9,
        edgecolor="#78829a", facecolor="#eef2f7", transform=transform, zorder=zorder,
    ))
    # Graduation ticks and a filled charge, so it reads as a loaded syringe rather than a blank box.
    ax.add_patch(FancyBboxPatch(
        (barrel_x + barrel_w * 0.10, -barrel_h / 2 + 0.025 * scale), barrel_w * 0.52, barrel_h - 0.05 * scale,
        boxstyle="round,pad=0,rounding_size=0.02", linewidth=0, facecolor="#d7e9f5", transform=transform, zorder=zorder + 0.1,
    ))
    for index in range(4):
        tick_x = barrel_x + barrel_w * (0.24 + index * 0.17)
        ax.plot([tick_x, tick_x], [-barrel_h * 0.20, barrel_h * 0.20], color="#9aa4b6", linewidth=0.55, transform=transform, zorder=zorder + 0.2)
    ax.add_patch(FancyBboxPatch(
        (barrel_x - 0.16 * scale, -barrel_h * 0.86), 0.07 * scale, barrel_h * 1.72,
        boxstyle="round,pad=0,rounding_size=0.02", linewidth=0.8,
        edgecolor="#78829a", facecolor="#c3ccd8", transform=transform, zorder=zorder,
    ))
    ax.plot([barrel_x, barrel_x - 0.16 * scale], [0.0, 0.0], color="#9aa4b6", linewidth=1.1, transform=transform, zorder=zorder)


def _apparatus_injection_block(ax, x, y, title, lines, ha="left", zorder=8, max_lines=9):
    """Dosing instructions rendered at the injection site, colour-coded per value type."""
    c = _APPARATUS
    ax.text(x, y, title, ha=ha, va="top", fontsize=9.0, weight="bold", color=c["ink"], zorder=zorder)
    line_y = y - 0.26
    if len(lines) > max_lines:
        hidden = len(lines) - (max_lines - 1)
        shown = list(lines[: max_lines - 1]) + [f"+{hidden} more (see Section 7)"]
    else:
        shown = list(lines)
    for text in shown:
        ax.text(x, line_y, text, ha=ha, va="top", fontsize=7.6, color=_apparatus_value_color(text), zorder=zorder)
        line_y -= 0.21
    return line_y


def _apparatus_value_color(text: str) -> str:
    c = _APPARATUS
    lowered = text.lower()
    if "µg/ml" in lowered or "ug/ml" in lowered or "mg/ml" in lowered:
        return c["concentration"]
    if "ml/min" in lowered or "ml/h" in lowered or "mg/dose" in lowered or "weigh" in lowered or "mg/q" in lowered:
        return c["rate"]
    if lowered.strip().endswith("ml") or " ml " in lowered or "volume" in lowered:
        return c["volume"]
    if "every" in lowered or "q24h" in lowered or "continuous" in lowered:
        return c["muted"]
    return c["muted"]


def _apparatus_legend(ax, x, y, w=3.3, h=1.14, zorder=5):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    c = _APPARATUS
    ax.add_patch(FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=c["panel_edge"],
        facecolor=c["paper"],
        zorder=zorder,
    ))
    items = [("Volume", c["volume"]), ("Concentration", c["concentration"]), ("Flow / dose", c["rate"]), ("Injection", c["inject"])]
    for index, (label, color) in enumerate(items):
        ix = x + 0.16 + (index % 2) * (w / 2)
        iy = y + h - 0.24 - (index // 2) * 0.27
        ax.add_patch(Rectangle((ix, iy - 0.06), 0.13, 0.13, facecolor=color, edgecolor="none", zorder=zorder + 1))
        ax.text(ix + 0.20, iy, label, ha="left", va="center", fontsize=7.4, color=c["muted"], zorder=zorder + 1)
    # Tube bore is a real encoding, not decoration, so the key has to spell it out.
    for offset, width, text in ((0.34, TUBE_FAST, "recirculation loop"), (0.13, TUBE_SLOW, "PK flows - width = flow")):
        key_y = y + offset
        ax.plot([x + 0.18, x + 0.56], [key_y, key_y], color=c["tube"], linewidth=width, solid_capstyle="round", zorder=zorder + 1)
        ax.text(x + 0.66, key_y, text, ha="left", va="center", fontsize=6.9, color=c["muted"], zorder=zorder + 1)


def _fmt_ug_per_ml(mg_per_ml: float) -> str:
    """Concentrations read better in ug/mL for the low values this app produces."""
    return f"{mg_per_ml * 1000:,.1f} µg/mL"


def _apparatus_manifold_route(ax, start, manifold_y, end, marker_direction, rate, inset=0.70, marker_t=0.5):
    """Route tubing from one bottle cap up to a shared overhead manifold, across, then down into another cap.

    Real bottles are plumbed through their caps, so tubing has to travel above the vessels rather
    than through them. Drawn as rise + straight run + descent so the flow marker always lands on the
    straight section, where it stays readable.
    """
    sx, sy = start
    ex, ey = end
    going_right = ex > sx
    lead_x = sx + (inset if going_right else -inset)
    tail_x = ex + (-inset if going_right else inset)
    _apparatus_tube(ax, [(sx, sy), (sx, manifold_y), (lead_x, manifold_y)])
    _apparatus_tube(ax, [(lead_x, manifold_y), (tail_x, manifold_y)])
    _apparatus_tube(ax, [(tail_x, manifold_y), (ex, manifold_y), (ex, ey)])
    pump_x = lead_x + (tail_x - lead_x) * marker_t
    _apparatus_pump(ax, pump_x, manifold_y, rate)
    arrow_x = pump_x + (tail_x - lead_x) * 0.26
    _apparatus_flow_arrow(ax, arrow_x, manifold_y, marker_direction)


def _new_apparatus_figure(width, height, xlim, ylim, dpi=200):
    import matplotlib.pyplot as plt

    _configure_schematic_fonts()
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(_APPARATUS["paper"])
    ax.set_facecolor(_APPARATUS["paper"])
    return fig, ax


def _plot_one_half_life_apparatus(view: dict):
    """Central-only apparatus: cartridge loop, central bottle, waste, and one central diluent.

    No extra compartment exists in this setup, so the layout is deliberately narrower than the
    two-half-life diagram rather than leaving an empty gap where the extra bottle used to be.
    """
    c = _APPARATUS
    fig, ax = _new_apparatus_figure(12.4, 8.4, (0, 12.4), (0, 8.4))

    central_x, bottle_y = 5.40, 3.95
    waste_x, diluent_x = 1.75, 9.05
    cartridge_y, manifold_y = 6.70, 5.45
    central_cap, side_cap = bottle_y + 0.80, bottle_y + 0.74

    # Cartridge recirculation loop, plumbed through the central cap on both sides.
    _apparatus_tube(ax, [(central_x - 0.26, central_cap), (central_x - 0.26, cartridge_y), (central_x - 1.52, cartridge_y)], linewidth=TUBE_FAST)
    _apparatus_tube(ax, [(central_x + 1.52, cartridge_y), (central_x + 0.26, cartridge_y), (central_x + 0.26, central_cap)], linewidth=TUBE_FAST)
    _apparatus_cartridge(ax, central_x, cartridge_y, w=2.9, h=0.62, label="Hollow fiber cartridge", volume=view["cartridge_volume"])
    _apparatus_pump(ax, central_x - 0.26, cartridge_y - 0.92, view["recirculation"], label_offset=(-0.52, 0.0), label_ha="right")
    _apparatus_flow_arrow(ax, central_x - 0.26, cartridge_y - 0.52, "up", size=0.17)

    _apparatus_manifold_route(ax, (central_x - 0.46, central_cap), manifold_y, (waste_x, side_cap), "left", view["waste_flow"])
    _apparatus_manifold_route(ax, (diluent_x, side_cap), manifold_y, (central_x + 0.46, central_cap), "left", view["central_diluent_flow"])

    _apparatus_bottle(ax, central_x, bottle_y, w=1.30, h=1.72, fill_level=0.62, label="Central compartment", volume=view["central_volume"], stir_bar=True)
    _apparatus_bottle(ax, waste_x, bottle_y, w=1.12, h=1.60, fill_level=0.24, label="Waste", volume=view["waste_total"])
    _apparatus_bottle(ax, diluent_x, bottle_y, w=1.12, h=1.60, fill_level=0.68, label="Diluent Central", volume=view["central_diluent_total"])

    # Needle tip lands on a septum port on the central bottle's shoulder, clear of all tubing.
    port_x, port_y = central_x - 0.65, bottle_y + 0.30
    _apparatus_injection_port(ax, port_x, port_y, angle_deg=180.0, scale=1.18)
    _apparatus_syringe(ax, (port_x - 0.21, port_y), angle_deg=0.0, scale=1.18)
    _apparatus_injection_band(
        ax, view["injection_groups"], 0.55, 2.35, width=4.0, columns=3, max_lines=8,
        header="Drug injection into central compartment",
    )

    ax.text(0.55, 8.22, view["title"], ha="left", va="top", fontsize=13.2, weight="bold", color=c["ink"])
    ax.text(0.55, 7.86, view["subtitle"], ha="left", va="top", fontsize=8.6, color=c["muted"])
    _apparatus_legend(ax, 8.65, 7.16, w=3.35, h=1.14)
    return fig


def _plot_two_half_life_apparatus(view: dict):
    """Central + extra apparatus, including the extra compartment and its diluent reservoir."""
    c = _APPARATUS
    # Central dosing blocks sit two per row. More than two blocks (for example a two-pump setup drug
    # plus other drugs) get a second row, and the canvas grows downward to hold it.
    central_band_rows = 2 if len(view["central_injection_groups"]) > 2 else 1
    band_extra_height = (central_band_rows - 1) * _injection_band_row_height(6)
    fig, ax = _new_apparatus_figure(13.8, 9.8 + band_extra_height, (0, 13.8), (-band_extra_height, 9.8))

    central_x, bottle_y = 5.05, 5.90
    waste_x, extra_x, diluent_extra_x = 1.60, 8.50, 11.90
    # Sits in the horizontal gap between the central and extra captions, and feeds the central
    # bottle's side rather than its base, so neither the riser nor its pump lands on a caption.
    diluent_central_x, diluent_central_y = 6.75, 3.05
    cartridge_y, manifold_y = 8.30, 7.30
    central_cap, side_cap = bottle_y + 0.80, bottle_y + 0.74
    extra_cap = bottle_y + 0.77

    _apparatus_tube(ax, [(central_x - 0.26, central_cap), (central_x - 0.26, cartridge_y), (central_x - 1.52, cartridge_y)], linewidth=TUBE_FAST)
    _apparatus_tube(ax, [(central_x + 1.52, cartridge_y), (central_x + 0.26, cartridge_y), (central_x + 0.26, central_cap)], linewidth=TUBE_FAST)
    _apparatus_cartridge(ax, central_x, cartridge_y, w=2.9, h=0.62, label="Hollow fiber cartridge", volume=view["cartridge_volume"])
    _apparatus_pump(ax, central_x - 0.26, cartridge_y - 0.92, view["recirculation"], label_offset=(-0.52, 0.0), label_ha="right")
    _apparatus_flow_arrow(ax, central_x - 0.26, cartridge_y - 0.52, "up", size=0.17)

    _apparatus_manifold_route(ax, (central_x - 0.46, central_cap), manifold_y, (waste_x, side_cap), "left", view["waste_flow"])
    _apparatus_manifold_route(ax, (extra_x, extra_cap), manifold_y, (central_x + 0.46, central_cap), "left", view["extra_to_central_flow"])

    _apparatus_tube(ax, [(diluent_central_x, diluent_central_y + 0.78), (diluent_central_x, bottle_y - 0.55), (central_x + 0.65, bottle_y - 0.55)])
    # Label goes above this pump: a side label hits "Extra compartment" and a label below lands on
    # the Diluent Central cap, since this riser is the shortest run on the diagram.
    diluent_pump_y = (diluent_central_y + bottle_y) / 2 - 0.03
    _apparatus_pump(ax, diluent_central_x, diluent_pump_y, view["central_diluent_flow"], label_offset=(0.0, 0.52), label_ha="center")
    _apparatus_flow_arrow(ax, diluent_central_x, diluent_pump_y - 0.47, "up")

    if view["show_extra_diluent"]:
        _apparatus_manifold_route(ax, (diluent_extra_x, side_cap), manifold_y, (extra_x + 0.36, extra_cap), "left", view["extra_diluent_flow"], inset=0.55)
        _apparatus_bottle(ax, diluent_extra_x, bottle_y, w=1.12, h=1.60, fill_level=0.68, label="Diluent Extra", volume=view["extra_diluent_total"])

    _apparatus_bottle(ax, central_x, bottle_y, w=1.30, h=1.72, fill_level=0.62, label="Central compartment", volume=view["central_volume"], stir_bar=True)
    _apparatus_bottle(ax, waste_x, bottle_y, w=1.12, h=1.60, fill_level=0.24, label="Waste", volume=view["waste_total"])
    _apparatus_bottle(ax, extra_x, bottle_y, w=1.18, h=1.64, fill_level=0.58, label="Extra compartment", volume=view["extra_volume"])
    _apparatus_bottle(ax, diluent_central_x, diluent_central_y, w=1.08, h=1.52, fill_level=0.68, label="Diluent Central", volume=view["central_diluent_total"])

    central_port = (central_x - 0.65, bottle_y + 0.30)
    _apparatus_injection_port(ax, *central_port, angle_deg=180.0, scale=1.18)
    _apparatus_syringe(ax, (central_port[0] - 0.21, central_port[1]), angle_deg=0.0, scale=1.18)
    _apparatus_injection_band(
        ax, view["central_injection_groups"], 0.55, 1.75, width=2.55, columns=2, max_lines=6,
        header="Injected into central", max_rows=central_band_rows,
    )

    extra_port = (extra_x + 0.59, bottle_y + 0.30)
    _apparatus_injection_port(ax, *extra_port, angle_deg=0.0, scale=1.18)
    _apparatus_syringe(ax, (extra_port[0] + 0.21, extra_port[1]), angle_deg=180.0, scale=1.18)
    _apparatus_injection_band(
        ax, view["extra_injection_groups"], 7.70, 1.75, width=2.90, columns=2, max_lines=6,
        header="Injected into extra",
    )

    ax.text(0.55, 9.62, view["title"], ha="left", va="top", fontsize=13.2, weight="bold", color=c["ink"])
    ax.text(0.55, 9.26, view["subtitle"], ha="left", va="top", fontsize=8.6, color=c["muted"])
    _apparatus_legend(ax, 10.05, 8.54, w=3.35, h=1.14)
    return fig


def _loading_dose_apparatus_lines(item: dict) -> list[str]:
    lines = [f"loading dose {item['loading_dose_mg']:.3f} mg"]
    volume = item.get("loading_volume_ml")
    concentration = item.get("loading_concentration_mg_ml")
    if volume and concentration is not None:
        lines.append(f"in {volume:g} mL = {_fmt_ug_per_ml(concentration)}")
    rate = item.get("loading_infusion_rate_ml_h")
    if rate:
        lines.append(f"at {rate:.2f} mL/h over {item.get('loading_duration_h', 0):g} h")
    return lines


def _central_drug_apparatus_groups(drug_inputs: dict, summary: dict, skip: str | None = None) -> list[tuple[str, list[str]]]:
    """Per-drug dosing instructions for every drug dosed straight into the central compartment."""
    groups = []
    for name, values in drug_inputs.items():
        if name == skip:
            continue
        item = summary.get(name)
        if not isinstance(item, dict):
            continue
        lines: list[str] = []
        if values.get("loading_dose"):
            lines.extend(_loading_dose_apparatus_lines(item))
        maintenance = values.get("maintenance")
        if maintenance == "continuous infusion":
            concentration = item.get("central_diluent_concentration_mg_ml")
            if concentration is not None:
                lines.append(f"then CI at {_fmt_ug_per_ml(concentration)}")
                lines.append("mixed in diluent central")
                lines.append(f"weigh {item['central_diluent_drug_per_24h_mg'] * 1.10:.2f} mg/q24h")
            else:
                lines.append("then CI in diluent central")
        elif maintenance == "intermittent infusion":
            interval = item.get("intermittent_interval_h", 0)
            dose_mg = item.get("intermittent_dose_mg", 0.0)
            lines.append(f"then every {interval:g} h: {dose_mg:.3f} mg")
            dose_volume = values.get("dose_volume_ml")
            if dose_volume:
                lines.append(f"in {dose_volume:g} mL = {_fmt_ug_per_ml(dose_mg / dose_volume)}")
                duration_h = item.get("intermittent_duration_h") or 0
                if duration_h:
                    lines.append(f"at {dose_volume / (duration_h * 60):.3f} mL/min over {duration_h:g} h")
            if interval:
                lines.append(f"{dose_mg * 24 / interval:.2f} mg/day")
        if lines:
            groups.append((name, lines))
    return groups


def _two_half_life_apparatus_view(
    system: SystemConfig,
    fos: FosfomycinConfig,
    drug_inputs: dict,
    summary: dict,
    scenario: str,
    duration_h: float,
    recirculation_ml_min: float,
    shared_half_life_h: float,
) -> dict:
    interval_h = fos.dosing_interval_min / 60
    dose_volume_ml = fos.central_infusion_ml_min * fos.infusion_duration_min
    setup_lines = [
        f"{fos.central_dose_mg:.3f} mg in {dose_volume_ml:g} mL",
        _fmt_ug_per_ml(fos.central_stock_mg_ml),
        f"at {fos.central_infusion_ml_min:.3f} mL/min over {fos.infusion_duration_min / 60:g} h",
        f"{fos.central_dose_mg * 24 / interval_h:.2f} mg/day",
    ]
    if fos.has_slow_line:
        # Two syringe pumps feed central in turn, so each gets its own block with the same three
        # facts in the same order: when it runs, how fast, how much. Each pump has its own syringe.
        main_end_h = fos.infusion_duration_min / 60
        slow_end_h = main_end_h + fos.slow_duration_min / 60
        central_groups = [
            (
                f"{fos.drug_name} pump 1 (main)",
                [
                    f"hour 0-{main_end_h:g} of every q{interval_h:g}h dose",
                    f"{fos.central_infusion_ml_min * 60:.2f} mL/h",
                    f"{fos.central_dose_mg:.2f} mg in {dose_volume_ml:g} mL",
                    f"syringe {fos.central_stock_mg_ml:.2f} mg/mL",
                ],
            ),
            (
                f"{fos.drug_name} pump 2 (slow)",
                [
                    f"hour {main_end_h:g}-{slow_end_h:g}, right after pump 1",
                    f"{fos.slow_infusion_ml_min * 60:.3f} mL/h",
                    f"{fos.slow_dose_mg:.2f} mg in {fos.slow_dose_volume_ml:.2f} mL",
                    f"own syringe {fos.slow_stock_mg_ml:.2f} mg/mL",
                    f"both pumps: {(fos.central_dose_mg + fos.slow_dose_mg) * 24 / interval_h:.2f} mg/day",
                ],
            ),
        ]
    else:
        central_groups = [(f"{fos.drug_name} q{interval_h:g}h", setup_lines)]
    central_groups.extend(_central_drug_apparatus_groups(drug_inputs, summary, skip=fos.drug_name))

    if scenario == "q24_replacement":
        fill_mg = fos.extra_stock_mg_ml * system.extra_volume_ml
        extra_groups = [(
            f"{fos.drug_name} extra q{fos.reservoir_replacement_interval_h:g}h",
            [
                "full compartment replacement",
                f"{system.extra_volume_ml:g} mL fill",
                _fmt_ug_per_ml(fos.extra_stock_mg_ml),
                f"{fill_mg:.3f} mg per replacement",
                f"+10%: {system.extra_volume_ml * 1.10:.1f} mL / {fill_mg * 1.10:.1f} mg",
            ],
        )]
    else:
        extra_dose_volume_ml = fos.extra_infusion_ml_min * fos.infusion_duration_min
        extra_groups = [(
            f"{fos.drug_name} extra q{interval_h:g}h",
            [
                f"{fos.extra_dose_mg:.3f} mg in {extra_dose_volume_ml:g} mL",
                _fmt_ug_per_ml(fos.extra_stock_mg_ml),
                f"at {fos.extra_infusion_ml_min:.3f} mL/min over {fos.infusion_duration_min / 60:g} h",
                "extra volume held by overflow line",
            ],
        )]

    return {
        "title": "HFIM apparatus - 2 half life (central + extra)",
        "subtitle": (
            f"{fos.drug_name} shaped by central + extra   |   shared central half-life "
            f"{shared_half_life_h:g} h   |   {duration_h:g} h run"
        ),
        "cartridge_volume": f"{system.cartridge_ml:g} mL",
        "central_volume": f"{system.central_bottle_ml:g} mL",
        "extra_volume": f"{system.extra_volume_ml:g} mL",
        "waste_total": f"{system.q_waste_ml_min * duration_h * 60:,.0f} mL",
        "central_diluent_total": f"{system.q_central_diluent_ml_min * duration_h * 60:,.0f} mL",
        "extra_diluent_total": f"{system.q_extra_diluent_ml_min * duration_h * 60:,.0f} mL",
        "recirculation": f"{recirculation_ml_min:g} mL/min",
        "waste_flow": f"{system.q_waste_ml_min:.3f} mL/min",
        "extra_to_central_flow": f"{system.q_extra_to_central_ml_min:.3f} mL/min",
        "central_diluent_flow": f"{system.q_central_diluent_ml_min:.3f} mL/min",
        "extra_diluent_flow": f"{system.q_extra_diluent_ml_min:.3f} mL/min",
        "show_extra_diluent": scenario != "q24_replacement" and system.q_extra_diluent_ml_min > 0,
        "central_injection_groups": central_groups,
        "extra_injection_groups": extra_groups,
    }


def _one_half_life_apparatus_view(
    system: SystemConfig,
    drug_inputs: dict,
    summary: dict,
    duration_h: float,
    recirculation_ml_min: float,
    shared_half_life_h: float,
) -> dict:
    return {
        "title": "HFIM apparatus - 1 half life (central only)",
        "subtitle": (
            f"Shared half-life {shared_half_life_h:g} h   |   Q = ln(2) x {system.central_volume_ml:g} mL / "
            f"({shared_half_life_h:g} h x 60) = {system.q_central_diluent_ml_min:.3f} mL/min   |   {duration_h:g} h run"
        ),
        "cartridge_volume": f"{system.cartridge_ml:g} mL",
        "central_volume": f"{system.central_bottle_ml:g} mL",
        "waste_total": f"{system.q_central_diluent_ml_min * duration_h * 60:,.0f} mL",
        "central_diluent_total": f"{system.q_central_diluent_ml_min * duration_h * 60:,.0f} mL",
        "recirculation": f"{recirculation_ml_min:g} mL/min",
        "waste_flow": f"{system.q_central_diluent_ml_min:.3f} mL/min",
        "central_diluent_flow": f"{system.q_central_diluent_ml_min:.3f} mL/min",
        "injection_groups": _central_drug_apparatus_groups(drug_inputs, summary),
    }


def _configure_schematic_fonts() -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "DejaVu Sans"


def _render_schematic_export_buttons(st, fig, key_prefix: str, file_prefix: str) -> None:
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "SVG",
        data=_figure_export_bytes(fig, "svg"),
        file_name=f"{file_prefix}.svg",
        mime="image/svg+xml",
        key=f"{key_prefix}_svg",
    )
    export_cols[1].download_button(
        "PDF",
        data=_figure_export_bytes(fig, "pdf"),
        file_name=f"{file_prefix}.pdf",
        mime="application/pdf",
        key=f"{key_prefix}_pdf",
    )
    export_cols[2].download_button(
        "PNG",
        data=_figure_export_bytes(fig, "png"),
        file_name=f"{file_prefix}.png",
        mime="image/png",
        key=f"{key_prefix}_png",
    )


def _figure_export_bytes(fig, file_format: str, dpi: int = 300) -> bytes:
    buffer = BytesIO()
    save_kwargs = {"format": file_format, "bbox_inches": "tight", "facecolor": "white"}
    if file_format == "png":
        save_kwargs["dpi"] = dpi
    fig.savefig(buffer, **save_kwargs)
    return buffer.getvalue()


def _plot_static(rows: list[dict], drugs: list[str], title: str, include_extra: bool, reference=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    if reference is not None:
        reference_times, reference_values, reference_label = reference
        ax.plot(reference_times, reference_values, label=reference_label, linewidth=1.4, linestyle=":", color="#444444", zorder=5)
    for drug in drugs:
        drug_rows = [row for row in rows if row["drug"] == drug]
        ax.plot(
            [row["time_h"] for row in drug_rows],
            [row["central_mg_l"] for row in drug_rows],
            label=f"{drug} central",
            linewidth=2,
        )
        if include_extra:
            ax.plot(
                [row["time_h"] for row in drug_rows],
                [row["extra_mg_l"] for row in drug_rows],
                label=f"{drug} extra",
                linewidth=2,
                linestyle="--",
            )
    ax.set_title(title)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _format_preparation_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        formatted.append({
            "Drug": row["drug"],
            "Component": row["component"],
            "Amount": _format_amount(row["amount_mg"], row["component"]),
            "Daily amount": "" if row["daily_amount_mg"] is None else f"{row['daily_amount_mg']:.3f} mg/day",
            "Note": row["note"],
        })
    return formatted


def _prep_rows_for_display(rows: list[dict], setup_drug_name: str, scenario: str) -> tuple[list[dict], list[dict], list[dict]]:
    setup_rows = []
    extra_replacement_rows = []
    other_rows = []
    for row in rows:
        if row["Drug"] != setup_drug_name:
            other_rows.append(row)
        elif scenario == "q24_replacement" and "fixed-concentration solution" in row["Component"]:
            extra_replacement_rows.append(row)
        else:
            setup_rows.append(row)
    return setup_rows, extra_replacement_rows, other_rows


def _preparation_destination_cards(
    prep_rows: list[dict],
    summary: dict,
    system: SystemConfig,
    fos: FosfomycinConfig,
    duration_h: float,
) -> list[dict]:
    direct_rows = [_row for _row in prep_rows if _preparation_destination(_row, summary) == "Central direct dosing"]
    diluent_rows = _central_diluent_reservoir_rows(summary, duration_h)
    extra_rows = [_row for _row in prep_rows if _preparation_destination(_row, summary) == "Extra q24h replacement"]
    central_recipe = _central_diluent_reservoir_summary(summary, duration_h)
    extra_summary = _replacement_solution_summary(system, fos, duration_h)
    interval_h = fos.reservoir_replacement_interval_h

    return [
        {
            "title": "Central direct dosing",
            "tone": "blue",
            "drug_names": _drug_names_for_rows(direct_rows),
            "primary_label": "Rows to administer",
            "primary_value": f"{len(direct_rows)}",
            "secondary_label": "Destination",
            "secondary_value": "central compartment",
            "caption": "Includes loading dose and intermittent/q6h central infusion rows administered directly into central.",
        },
        {
            "title": "Central diluent q24h reservoir",
            "tone": "teal",
            "drug_names": _drug_names_for_rows(diluent_rows),
            "primary_label": "Total to prepare q24h",
            "primary_value": central_recipe["prepared_volume_q24h"],
            "secondary_label": "Shared volume",
            "secondary_value": central_recipe["volume_q24h"],
            "caption": "Continuous-infusion drugs are mixed into one shared reservoir; do not multiply volume by drug count.",
        },
        {
            "title": f"Extra q{interval_h:g}h replacement",
            "tone": "amber",
            "drug_names": _drug_names_for_rows(extra_rows),
            "primary_label": f"Volume to prepare q{interval_h:g}h",
            "primary_value": f"{extra_summary['prepared_volume_per_interval_ml'] * 1.10:.1f} mL",
            "secondary_label": "Drug to weigh",
            "secondary_value": f"{extra_summary['prepared_drug_per_interval_mg'] * 1.10:.3f} mg",
            "caption": "Full extra-compartment replacement at the selected interval; 10% extra is included in this card.",
        },
    ]


def _preparation_review_rows(
    prep_rows: list[dict],
    summary: dict,
    system: SystemConfig,
    fos: FosfomycinConfig | None,
    duration_h: float,
) -> list[dict]:
    rows = []
    central_recipe = _central_diluent_reservoir_summary(summary, duration_h)
    # With no setup drug (central_only) there is no extra fill to describe, and no prep row can
    # carry the "Extra q24h replacement" destination, so the extra summary is simply not needed.
    extra_summary = _replacement_solution_summary(system, fos, duration_h) if fos is not None else None
    interval_h = fos.reservoir_replacement_interval_h if fos is not None else 0.0

    for row in prep_rows:
        destination = _preparation_destination(row, summary)
        drug_summary = summary.get(row["Drug"], {}) if isinstance(summary.get(row["Drug"], {}), dict) else {}

        if destination == "Central diluent q24h reservoir":
            drug_mg = drug_summary.get("central_diluent_drug_per_24h_mg", 0.0)
            concentration = drug_summary.get("central_diluent_concentration_mg_ml", 0.0)
            rows.append({
                "Drug": row["Drug"],
                "Add into": destination,
                "Dosing part": row["Component"],
                "Frequency": "q24h reservoir replacement",
                "Concentration": f"{concentration:.6f} mg/mL ({concentration * 1000:.3f} ug/mL)",
                "Required amount": f"{drug_mg:.3f} mg/q24h",
                "10% extra": f"{drug_mg * 0.10:.3f} mg",
                "Amount to weigh": f"{drug_mg * 1.10:.3f} mg/q24h",
                "Volume": f"shared {central_recipe['prepared_volume_q24h']} q24h",
                "Note": "mix into the same central diluent reservoir",
            })
        elif destination == "Extra q24h replacement":
            amount_mg = extra_summary["prepared_drug_per_interval_mg"]
            rows.append({
                "Drug": row["Drug"],
                "Add into": destination,
                "Dosing part": row["Component"],
                "Frequency": f"q{interval_h:g}h full replacement",
                "Concentration": f"{extra_summary['concentration_mg_ml']:.6f} mg/mL",
                "Required amount": f"{amount_mg:.3f} mg/q{interval_h:g}h",
                "10% extra": f"{amount_mg * 0.10:.3f} mg",
                "Amount to weigh": f"{amount_mg * 1.10:.3f} mg/q{interval_h:g}h",
                "Volume": f"{extra_summary['prepared_volume_per_interval_ml'] * 1.10:.1f} mL q{interval_h:g}h",
                "Note": "prepare the extra fill; Qextra transfer is not a second prepared solution",
            })
        elif row["Component"] == "loading dose":
            loading_amount_mg = drug_summary.get("loading_dose_mg")
            loading_volume_ml = drug_summary.get("loading_volume_ml")
            loading_concentration = drug_summary.get("loading_concentration_mg_ml")
            loading_rate_ml_h = drug_summary.get("loading_infusion_rate_ml_h")
            loading_rate_ml_min = drug_summary.get("loading_infusion_rate_ml_min")
            rows.append({
                "Drug": row["Drug"],
                "Add into": destination,
                "Dosing part": row["Component"],
                "Frequency": "loading dose",
                "Concentration": (
                    f"{loading_concentration:.6f} mg/mL ({loading_concentration * 1000:.3f} ug/mL)"
                    if loading_concentration is not None else ""
                ),
                "Required amount": f"{loading_amount_mg:.3f} mg" if loading_amount_mg is not None else row["Amount"],
                "10% extra": "not included",
                "Amount to weigh": f"{loading_amount_mg:.3f} mg" if loading_amount_mg is not None else row["Amount"],
                "Volume": f"{loading_volume_ml:g} mL" if loading_volume_ml is not None else _volume_from_note(row["Note"]),
                "Note": (
                    f"direct into central; infuse over {drug_summary.get('loading_duration_h', 0):g} h "
                    f"at {loading_rate_ml_h:.2f} mL/h ({loading_rate_ml_min:.3f} mL/min)"
                    if loading_rate_ml_h is not None and loading_rate_ml_min is not None else row["Note"]
                ),
            })
        else:
            rows.append({
                "Drug": row["Drug"],
                "Add into": destination,
                "Dosing part": row["Component"],
                "Frequency": _frequency_from_component(row["Component"], row["Daily amount"]),
                "Concentration": _concentration_from_note(row["Note"]),
                "Required amount": row["Amount"],
                "10% extra": "not included",
                "Amount to weigh": row["Daily amount"] or row["Amount"],
                "Volume": _volume_from_note(row["Note"]),
                "Note": row["Note"],
            })
    return rows


def _preparation_destination(row: dict, summary: dict) -> str:
    component = row["Component"]
    drug_summary = summary.get(row["Drug"], {})
    if "fixed-concentration solution" in component:
        return "Extra q24h replacement"
    if component == "continuous infusion" and isinstance(drug_summary, dict) and drug_summary.get("central_diluent_concentration_mg_ml") is not None:
        return "Central diluent q24h reservoir"
    return "Central direct dosing"


def _drug_names_for_rows(rows: list[dict]) -> str:
    names = []
    for row in rows:
        name = row.get("Drug", "")
        if name and name not in names:
            names.append(name)
    return ", ".join(names) if names else "none"


def _frequency_from_component(component: str, daily_amount: str) -> str:
    if component == "loading dose":
        return "loading dose"
    if component.startswith("central q"):
        return component.removeprefix("central ")
    if component.startswith("intermittent q"):
        return component.replace("intermittent ", "")
    return "per day" if daily_amount else "single dose"


def _concentration_from_note(note: str) -> str:
    if " at " in note and " mg/mL" in note:
        return note.split(" at ", 1)[1].split(" and ", 1)[0]
    return ""


def _volume_from_note(note: str) -> str:
    if " mL over " in note:
        return note.split(" over ", 1)[0]
    return ""


def _render_preparation_styles(st) -> None:
    st.markdown(
        """
        <style>
        /* Light cards matching the app's white page background and the Section 4 schematic's
        blue/teal/amber color key, instead of the dark glass-panel palette these were originally
        written for (which rendered as low-contrast gray-on-gray on this light-themed page). */
        .prep-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 16px 16px 14px 16px;
            min-height: 238px;
            background: #f8fafc;
        }
        .prep-card-blue { border-top: 4px solid #315fbd; background: #f3f7ff; }
        .prep-card-teal { border-top: 4px solid #117c73; background: #f1fbf9; }
        .prep-card-amber { border-top: 4px solid #b8650b; background: #fff9ed; }
        .prep-card-title {
            color: #172033;
            font-size: 1.03rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .prep-card-drugs {
            color: #475569;
            font-size: 0.9rem;
            min-height: 42px;
            margin-bottom: 12px;
        }
        .prep-card-label {
            color: #5a6678;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .prep-card-value {
            color: #172033;
            font-size: 1.75rem;
            line-height: 1.15;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .prep-card-caption {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_preparation_card(container, card: dict) -> None:
    container.markdown(
        f"""
        <div class="prep-card prep-card-{escape(card['tone'])}">
            <div class="prep-card-title">{escape(card['title'])}</div>
            <div class="prep-card-drugs">{escape(card['drug_names'])}</div>
            <div class="prep-card-label">{escape(card['primary_label'])}</div>
            <div class="prep-card-value">{escape(card['primary_value'])}</div>
            <div class="prep-card-label">{escape(card['secondary_label'])}</div>
            <div class="prep-card-value">{escape(card['secondary_value'])}</div>
            <div class="prep-card-caption">{escape(card['caption'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _solution_volume_rows(q_central_diluent: float, q_extra_diluent: float, scenario: str, duration_h: float) -> list[dict]:
    rows = [_solution_volume_row("Central diluent", q_central_diluent, duration_h)]
    # Only overflow runs a continuous extra feed. q24 replacement prepares a fill instead, and
    # central_only has no extra compartment at all, so neither should list an extra diluent volume.
    if scenario == "overflow":
        rows.append(_solution_volume_row("Extra diluent / feed volume", q_extra_diluent, duration_h))
    return rows


def _solution_volume_row(name: str, flow_ml_min: float, duration_h: float) -> dict:
    daily_ml = flow_ml_min * 24 * 60
    total_ml = flow_ml_min * duration_h * 60
    return {
        "Solution": name,
        "Flow": f"{flow_ml_min:g} mL/min",
        "24 h volume": f"{daily_ml:.1f} mL",
        "24 h + 10%": f"{daily_ml * 1.10:.1f} mL",
        f"{duration_h:g} h total": f"{total_ml:.1f} mL",
        f"{duration_h:g} h total + 10%": f"{total_ml * 1.10:.1f} mL",
    }


def _replacement_solution_rows(system: SystemConfig, fos: FosfomycinConfig, duration_h: float) -> list[dict]:
    summary = _replacement_solution_summary(system, fos, duration_h)
    replacements = summary["replacements"]
    volume_ml = summary["fill_volume_ml"]
    amount_mg = summary["fill_drug_mg"]
    transfer_volume_ml = summary["transfer_volume_ml"]
    transfer_amount_mg = summary["transfer_drug_mg"]
    prepared_volume_ml = summary["prepared_volume_per_interval_ml"]
    prepared_amount_mg = summary["prepared_drug_per_interval_mg"]
    return [
        {
            "Use": "Fill the extra compartment at the start of each 24 h block",
            "How calculated": f"Extra volume = {volume_ml:.1f} mL at {fos.extra_stock_mg_ml:.6f} mg/mL",
            "Drug per interval": f"{amount_mg:.3f} mg",
            "24 h +10%": f"{volume_ml * 1.10:.1f} mL; {amount_mg * 1.10:.3f} mg",
            f"{duration_h:g} h total +10%": f"{volume_ml * replacements * 1.10:.1f} mL; {amount_mg * replacements * 1.10:.3f} mg",
        },
        {
            "Use": "Drug delivered from extra to central during the same 24 h block",
            "How calculated": (
                f"Qextra {system.q_extra_to_central_ml_min:g} mL/min x "
                f"{fos.reservoir_replacement_interval_h:g} h x 60 = {transfer_volume_ml:.1f} mL"
            ),
            "Drug per interval": f"{transfer_amount_mg:.3f} mg",
            "24 h +10%": "not separately prepared",
            f"{duration_h:g} h total +10%": "not separately prepared",
        },
        {
            "Use": "Total solution to prepare for one 24 h block",
            "How calculated": f"pressure/filter-controlled setup: prepare fill only = {prepared_volume_ml:.1f} mL",
            "Drug per interval": f"{prepared_amount_mg:.3f} mg",
            "24 h +10%": f"{prepared_volume_ml * 1.10:.1f} mL; {prepared_amount_mg * 1.10:.3f} mg",
            f"{duration_h:g} h total +10%": f"{prepared_volume_ml * replacements * 1.10:.1f} mL; {prepared_amount_mg * replacements * 1.10:.3f} mg",
        }
    ]


def _replacement_solution_summary(system: SystemConfig, fos: FosfomycinConfig, duration_h: float) -> dict[str, float]:
    replacements = max(1, math.ceil(duration_h / fos.reservoir_replacement_interval_h))
    fill_volume_ml = system.extra_volume_ml
    fill_drug_mg = fill_volume_ml * fos.extra_stock_mg_ml
    transfer_volume_ml = system.q_extra_to_central_ml_min * fos.reservoir_replacement_interval_h * 60
    transfer_drug_mg = transfer_volume_ml * fos.extra_stock_mg_ml
    return {
        "concentration_mg_ml": fos.extra_stock_mg_ml,
        "replacements": replacements,
        "fill_volume_ml": fill_volume_ml,
        "fill_drug_mg": fill_drug_mg,
        "transfer_volume_ml": transfer_volume_ml,
        "transfer_drug_mg": transfer_drug_mg,
        "prepared_volume_per_interval_ml": fill_volume_ml,
        "prepared_drug_per_interval_mg": fill_drug_mg,
        "total_volume_with_overfill_ml": fill_volume_ml * replacements * 1.10,
        "total_drug_with_overfill_mg": fill_drug_mg * replacements * 1.10,
    }


def _central_diluent_reservoir_summary(summary: dict, duration_h: float) -> dict[str, str]:
    volumes = [
        item.get("central_diluent_volume_per_24h_ml", 0.0)
        for drug, item in summary.items()
        if drug != "drug_preparation"
        and isinstance(item, dict)
        and item.get("central_diluent_concentration_mg_ml") is not None
    ]
    volume_ml = max(volumes) if volumes else 0.0
    replacements = max(1, math.ceil(duration_h / 24))
    return {
        "volume_q24h": f"{volume_ml:.1f} mL",
        "extra_volume_q24h_10_percent": f"{volume_ml * 0.10:.1f} mL",
        "prepared_volume_q24h": f"{volume_ml * 1.10:.1f} mL",
        "prepared_volume_total": f"{volume_ml * replacements * 1.10:.1f} mL",
        "replacements": f"{replacements:g}",
    }


def _central_diluent_reservoir_rows(summary: dict, duration_h: float) -> list[dict]:
    rows = []
    replacements = max(1, math.ceil(duration_h / 24))
    for drug, item in summary.items():
        if drug == "drug_preparation" or not isinstance(item, dict):
            continue
        concentration = item.get("central_diluent_concentration_mg_ml")
        if concentration is None:
            continue
        volume_ml = item.get("central_diluent_volume_per_24h_ml", 0.0)
        drug_mg = item.get("central_diluent_drug_per_24h_mg", 0.0)
        rows.append({
            "Drug": drug,
            "Target": f"{item.get('target_concentration_mg_l', 0):g} mg/L central",
            "Central diluent concentration": f"{concentration:.6f} mg/mL ({concentration * 1000:.3f} ug/mL)",
            "Required drug per q24h": f"{drug_mg:.3f} mg",
            "10% extra drug q24h": f"{drug_mg * 0.10:.3f} mg",
            "Drug to weigh q24h": f"{drug_mg * 1.10:.3f} mg",
            f"Drug to weigh {duration_h:g} h": f"{drug_mg * replacements * 1.10:.3f} mg",
            "Note": "add to the same shared reservoir volume; do not multiply volume by drug count",
        })
    return rows


def _prep_group_title(rows: list[dict]) -> str:
    names = []
    for row in rows:
        name = row["Drug"]
        if name not in names:
            names.append(name)
    return " / ".join(names)


def _format_amount(value: float, component: str) -> str:
    unit = "mg/h" if component == "continuous infusion" else "mg"
    return f"{value:.3f} {unit}"


def _setup_assistant_panel(st, agent_context: dict) -> None:
    if "setup_agent_messages" not in st.session_state:
        st.session_state.setup_agent_messages = [
            {
                "role": "assistant",
                "content": "You can ask me whether the current HFIM setup is reasonable, or ask for help choosing loading-dose targets, maintenance dosing, and extra-compartment settings.",
            }
        ]

    with st.container(border=True):
        for message in st.session_state.setup_agent_messages[-6:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask the HFIM setup assistant...")
        if question:
            st.session_state.setup_agent_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Checking setup..."):
                    reply = ask_setup_agent(question, agent_context, project_root=Path(__file__).resolve().parents[1])
                label = "Gemini" if reply["source"] == "gemini" else "Local rules"
                content = f"**Source: {label}**\n\n{reply['message']}"
                st.markdown(content)
            st.session_state.setup_agent_messages.append({"role": "assistant", "content": content})


def _interpretation_text(scenario: str, setup_drug_name: str, setup_summary: dict, setup_target_auc: float, summary: dict, drug_names: list[str]) -> str:
    auc = setup_summary["central_auc_0_24_mg_h_l"]
    delta = auc - setup_target_auc
    setup_text = (
        f"Overflow setup keeps the extra-compartment volume fixed with an extra outflow line. The advantage is that you do not need to adjust the pump during each dose; the tradeoff is that some {setup_drug_name} can leave through overflow."
        if scenario == "overflow"
        else f"q24h replacement setup fills the extra compartment with {setup_drug_name} at t=0 and replaces it every 24 h. In the pressure/filter-controlled assumption, Qextra moves drug from that same fill into central; it is not a separately prepared reserve solution."
    )
    lines = [
        f"- {setup_text}\n"
        f"- {setup_drug_name} central AUC0-24 = **{auc:.1f} mg*h/L**"
        + (f"; difference from target {setup_target_auc:.1f} is **{delta:+.1f} mg*h/L**." if setup_target_auc else ".")
        + f"\n- Central Cavg = **{setup_summary['central_cavg_0_24_mg_l']:.1f} mg/L**; Cmax = **{setup_summary['central_cmax_mg_l']:.1f} mg/L**; Cmin after 24 h = **{setup_summary['central_cmin_after_24h_mg_l']:.1f} mg/L**."
    ]
    for name in drug_names:
        item = summary[name]
        lines.append(
            f"- {name} target central concentration = **{item['target_concentration_mg_l']:.2f} mg/L**; "
            f"dosing mode = **{item['dosing_mode']}**; loading dose = **{item['loading_dose_mg']:.3f} mg**; "
            f"continuous infusion = **{item['infusion_rate_mg_h']:.3f} mg/h**."
        )
    return "\n".join(lines)


def _slow_line_equation_lines(fos: FosfomycinConfig) -> list[str]:
    if not fos.has_slow_line:
        return []
    interval_h = fos.dosing_interval_min / 60
    start_h = fos.infusion_duration_min / 60
    return [
        "   Two-phase setup: a second central line runs right after the main infusion.",
        f"   Slow line window = h {start_h:g} to {start_h + fos.slow_duration_min / 60:g} of every q{interval_h:g}h interval",
        f"   Slow line volume = rate x duration = {fos.slow_infusion_ml_min:.6g} x {fos.slow_duration_min:g} = {fos.slow_dose_volume_ml:.3f} mL",
        f"   Slow line amount = stock x volume = {fos.slow_stock_mg_ml:.6g} x {fos.slow_dose_volume_ml:.3f} = {fos.slow_dose_mg:.3f} mg",
        f"   Daily slow line amount = {fos.slow_dose_mg:.3f} x 24 / {interval_h:g} = {fos.slow_dose_mg * 24 / interval_h:.3f} mg/day",
        "   Main rate, slow rate and extra concentration are the least-squares mix that best follows the true",
        "   one-compartment steady-state curve, then scaled so steady-state Cavg equals the target exactly.",
    ]


def _equation_text(
    system: SystemConfig,
    fos: FosfomycinConfig,
    setup_summary: dict,
    setup_target_auc: float,
    scenario: str,
    duration_h: float,
    q_central_diluent: float,
    q_extra_diluent: float,
    target_system_half_life: float,
) -> str:
    central_volume = system.central_volume_ml
    central_dose_volume = fos.central_infusion_ml_min * fos.infusion_duration_min
    dose_interval_h = fos.dosing_interval_min / 60
    central_dose_mg = fos.central_dose_mg
    daily_central_mg = central_dose_mg * 24 / dose_interval_h
    central_daily_volume = q_central_diluent * 24 * 60
    total_central_outflow = system.q_waste_ml_min
    setup_central_average_flow = average_intermittent_rate(
        fos.central_infusion_ml_min, fos.infusion_duration_min, fos.dosing_interval_min
    )
    ci_effective_outflow = total_central_outflow + setup_central_average_flow
    target_central_outflow = flow_for_half_life(central_volume, target_system_half_life)
    extra_replacement_mg = fos.extra_stock_mg_ml * system.extra_volume_ml
    extra_transfer_volume_ml = system.q_extra_to_central_ml_min * fos.reservoir_replacement_interval_h * 60
    extra_transfer_mg = extra_transfer_volume_ml * fos.extra_stock_mg_ml
    replacements = max(1, math.ceil(duration_h / fos.reservoir_replacement_interval_h))
    lines = [
        "1. Central effective volume",
        f"   Vc = central bottle + cartridge = {system.central_bottle_ml:g} + {system.cartridge_ml:g} = {central_volume:g} mL",
        "",
        "2. Flow settings",
        "   The shortest active drug half-life sets the shared central-to-waste flow:",
        "   Qcentral_out_target = ln(2) x Vcentral / (shortest t1/2 x 60)",
        f"   Qcentral_out_target = ln(2) x {central_volume:g} / ({target_system_half_life:g} x 60) = {target_central_outflow:.6g} mL/min",
        "   Qcentral_out_actual = Qextra_to_central + Qcentral_diluent",
        f"   Qcentral_out_actual = {system.q_extra_to_central_ml_min:.6g} + {q_central_diluent:.6g} = {total_central_outflow:.6g} mL/min",
        "   Qcentral_diluent = max(0, Qcentral_out_target - Qextra_to_central) in auto mode",
        "   Continuous-infusion drugs mixed into central diluent use a time-weighted effective outflow:",
        "   Qsetup_central_average = Qsetup_central x infusion_duration / dosing_interval",
        f"   Qsetup_central_average = {fos.central_infusion_ml_min:.6g} x {fos.infusion_duration_min:g} / {fos.dosing_interval_min:g} = {setup_central_average_flow:.6g} mL/min",
        "   QCI_effective_outflow = Qcentral_out_actual + Qsetup_central_average",
        f"   QCI_effective_outflow = {total_central_outflow:.6g} + {setup_central_average_flow:.6g} = {ci_effective_outflow:.6g} mL/min",
        "   For each CI drug: CI input mg/min = Css_target x QCI_effective_outflow / 1000",
        "   Central diluent reservoir concentration = CI input mg/min / Qcentral_diluent",
    ]
    if scenario == "q24_replacement":
        max_single_fill_qextra = system.extra_volume_ml / (fos.reservoir_replacement_interval_h * 60)
        lines.extend([
            "   In q24h replacement mode, Qextra is an independent physical transfer setting.",
            f"   Qextra_to_central = {system.q_extra_to_central_ml_min:.6g} mL/min",
            f"   One-fill q{fos.reservoir_replacement_interval_h:g}h Qextra limit = Vextra / (interval x 60) = {system.extra_volume_ml:g} / ({fos.reservoir_replacement_interval_h:g} x 60) = {max_single_fill_qextra:.6g} mL/min",
        ])
    else:
        lines.extend([
            "   Extra washout flow can also be calculated from a target half-life in overflow mode:",
            f"   Qextra = ln(2) x {system.extra_volume_ml:g} / ({target_system_half_life:g} x 60) = {system.q_extra_to_central_ml_min:.6g} mL/min",
        ])
    lines.extend([
        "",
        "3. Central q-hour dose",
        f"   Dose volume = central infusion rate x infusion duration = {fos.central_infusion_ml_min:.6g} x {fos.infusion_duration_min:g} = {central_dose_volume:.3f} mL",
        f"   Dose amount = central stock x dose volume = {fos.central_stock_mg_ml:.6g} x {central_dose_volume:.3f} = {central_dose_mg:.3f} mg",
        f"   Daily central amount = dose amount x 24 / interval = {central_dose_mg:.3f} x 24 / {dose_interval_h:g} = {daily_central_mg:.3f} mg/day",
        *_slow_line_equation_lines(fos),
        "",
        "4. AUC calculation",
        "   AUC0-24 = trapezoidal sum of central concentration over 0 to 24 h",
        f"   Simulated AUC0-24 = {setup_summary['central_auc_0_24_mg_h_l']:.3f} mg*h/L",
        f"   Cavg = AUC0-24 / 24 = {setup_summary['central_auc_0_24_mg_h_l']:.3f} / 24 = {setup_summary['central_cavg_0_24_mg_l']:.3f} mg/L",
    ])
    if setup_target_auc:
        lines.append(f"   Target error = simulated - target = {setup_summary['central_auc_0_24_mg_h_l']:.3f} - {setup_target_auc:.3f} = {setup_summary['central_auc_0_24_mg_h_l'] - setup_target_auc:+.3f} mg*h/L")
    lines.extend([
        "",
        "5. Blank solution volume to prepare",
        f"   Central diluent per 24 h = Qcentral_diluent x 1440 = {q_central_diluent:.6g} x 1440 = {central_daily_volume:.1f} mL",
        f"   Central diluent per 24 h with 10% extra = {central_daily_volume:.1f} x 1.10 = {central_daily_volume * 1.10:.1f} mL",
        f"   Central diluent for {duration_h:g} h = {q_central_diluent:.6g} x {duration_h:g} x 60 = {q_central_diluent * duration_h * 60:.1f} mL",
        f"   Central diluent for {duration_h:g} h with 10% extra = {q_central_diluent * duration_h * 60 * 1.10:.1f} mL",
    ])
    if scenario == "q24_replacement":
        lines.extend([
            "",
            f"6. Extra q{fos.reservoir_replacement_interval_h:g}h full replacement",
            f"   Aextra replacement = Cextra_replacement x Vextra = {fos.extra_stock_mg_ml:.6g} x {system.extra_volume_ml:g} = {extra_replacement_mg:.3f} mg",
            f"   Central input from extra during each interval = Qextra_to_central x Cextra_replacement",
            f"   Extra amount delivered to central per interval = {system.q_extra_to_central_ml_min:.6g} x {fos.extra_stock_mg_ml:.6g} x {fos.reservoir_replacement_interval_h:g} x 60 = {extra_transfer_mg:.3f} mg",
            "   This delivered amount is drawn from the same extra fill; it is not added again as a separate reserve preparation.",
            f"   Replacement count for {duration_h:g} h = ceiling({duration_h:g} / {fos.reservoir_replacement_interval_h:g}) = {replacements:g}",
            f"   Total prepared extra fill = {system.extra_volume_ml:g} x {replacements:g} = {system.extra_volume_ml * replacements:.1f} mL",
            f"   Total prepared extra fill with 10% extra = {system.extra_volume_ml * replacements * 1.10:.1f} mL and {extra_replacement_mg * replacements * 1.10:.3f} mg",
            "",
            "7. Differential equations used during each time step",
            "   Cextra = Aextra / Vextra",
            "   Ccentral = Acentral / Vcentral",
            "   In q24h replacement mode, Cextra is held constant during each 24 h interval.",
            "   dAextra/dt = 0 in the simulator boundary condition",
            "   dAcentral/dt = central input + Qextra_to_central x Cextra - central output x Ccentral",
            "   At each q24h replacement time: Aextra is reset to Cextra_replacement x Vextra.",
        ])
    else:
        extra_daily_volume = q_extra_diluent * 24 * 60
        lines.extend([
            "",
            "6. Extra diluent / feed volume",
            f"   Extra volume per 24 h = Qextra inlet x 1440 = {q_extra_diluent:.6g} x 1440 = {extra_daily_volume:.1f} mL",
            f"   Extra volume per 24 h with 10% extra = {extra_daily_volume:.1f} x 1.10 = {extra_daily_volume * 1.10:.1f} mL",
        ])
    return "```text\n" + "\n".join(lines) + "\n```"


# ---------------------------------------------------------------------------
# "2 half life - Blaser" page: the long-half-life drug gets its true half-life from
# a diluted extra compartment (Blaser design), dosed into central and extra at once.
# Every pump is set once: the two outflow pumps run slightly above the continuous
# inflow so they carry away the dose volumes, and nothing is touched during the run.
# ---------------------------------------------------------------------------


def _page_blaser() -> None:
    import streamlit as st

    st.title("HFIM PK Simulator - 2 half life (Blaser)")
    st.caption(
        "Two different half-lives with the true curve shape: the longer-half-life drug is dosed into central and an "
        "extra compartment together, and the extra compartment is diluted so both fall at that drug's own half-life. "
        "Every pump is set once for the whole run. Use this page when trough or T>MIC matters, or when the "
        "2 half life page reports a deviation above its limit."
    )

    st.subheader("1. Simulation setup")
    setup_cols = st.columns(3)
    active_drug_count = int(setup_cols[0].number_input(
        "Number of drugs",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
        help="One selected drug uses the central + extra setup; the others are dosed into central only and share the central washout.",
    ))
    duration_h = setup_cols[1].number_input("Simulation duration (h)", min_value=24.0, value=168.0, step=24.0, key="bl_duration")
    dt_min = setup_cols[2].number_input("Time step (min)", min_value=0.25, value=1.0, step=0.25, key="bl_dt")
    short_half_life_h = _shared_central_half_life_from_widget_state(active_drug_count, st.session_state)

    with st.expander("Compartment settings", expanded=True):
        st.markdown(
            "Only the central volumes are entered. Flows and the extra-compartment volume are not free choices here: "
            "they follow from the two half-lives and the central volume."
        )
        volume_cols = st.columns(3)
        central_bottle_ml = volume_cols[0].number_input("Central bottle (mL)", min_value=1.0, value=100.0, step=1.0, key="bl_central")
        cartridge_ml = volume_cols[1].number_input("Cartridge (mL)", min_value=0.0, value=70.0, step=1.0, key="bl_cartridge")
        recirculation_ml_min = volume_cols[2].number_input(
            "Cartridge recirculation (mL/min)", min_value=0.0, value=120.0, step=5.0, key="bl_recirculation"
        )

    st.subheader("2. Drug targets and injection settings")
    st.caption("The shortest half-life entered here sets the central washout. One longer-half-life drug is then chosen for the extra compartment in Section 3.")
    drug_inputs = _drug_input_panel(st, active_drug_count)

    st.subheader("3. Long half-life drug: central + extra dosing")
    setup_drug_names = list(drug_inputs.keys())
    default_setup_index = max(
        range(len(setup_drug_names)), key=lambda index: drug_inputs[setup_drug_names[index]]["half_life_h"]
    )
    setup_drug_name = st.selectbox(
        "Drug that uses the extra compartment", setup_drug_names, index=default_setup_index, key="bl_setup_drug"
    )
    setup_values = drug_inputs[setup_drug_name]
    dose_cols = st.columns(5)
    target_css_mg_l = dose_cols[0].number_input(
        "Target Css / Cavg (mg/L)", min_value=0.0, value=setup_values["target_concentration_mg_l"], step=5.0, key="bl_target"
    )
    central_dose_volume_ml = dose_cols[1].number_input("Central dose volume (mL)", min_value=0.1, value=6.0, step=0.5, key="bl_central_volume")
    extra_dose_volume_ml = dose_cols[2].number_input("Extra dose volume (mL)", min_value=0.1, value=6.0, step=0.5, key="bl_extra_volume")
    infusion_duration_h = dose_cols[3].number_input(
        "Infusion duration (h)", min_value=0.01, value=float(setup_values.get("maintenance_duration_h") or 1.0), step=0.25, key="bl_infusion"
    )
    dosing_frequency_h = dose_cols[4].number_input(
        "Dosing frequency (h)", min_value=0.1, value=float(setup_values["dosing_frequency_h"] or 6.0), step=1.0, key="bl_frequency"
    )
    infusion_min = int(round(infusion_duration_h * 60))
    interval_min = int(round(dosing_frequency_h * 60))

    setup = solve_blaser_setup(
        central_bottle_ml,
        cartridge_ml,
        short_half_life_h,
        setup_values["half_life_h"],
        target_css_mg_l,
        central_dose_volume_ml,
        extra_dose_volume_ml,
        infusion_min,
        interval_min,
        drug_name=setup_drug_name,
        dt_min=dt_min,
        duration_h=duration_h,
    )
    if setup.central_stock_mg_ml <= 0:
        st.error(setup.message)
        return

    pump_cols = st.columns(4)
    pump_cols[0].metric("Extra diluent pump", f"{setup.q_extra_diluent_ml_min:.3f} mL/min")
    pump_cols[1].metric("Extra to central pump", f"{setup.q_extra_to_central_ml_min:.3f} mL/min")
    pump_cols[2].metric("Central diluent pump", f"{setup.q_central_diluent_ml_min:.3f} mL/min")
    pump_cols[3].metric("Waste pump", f"{setup.q_waste_ml_min:.3f} mL/min")
    solved_cols = st.columns(4)
    solved_cols[0].metric("Extra compartment volume", f"{setup.extra_volume_ml:.1f} mL")
    solved_cols[1].metric("Central dose syringe", f"{setup.central_stock_mg_ml:.6f} mg/mL")
    solved_cols[2].metric("Extra dose syringe", f"{setup.extra_stock_mg_ml:.6f} mg/mL")
    solved_cols[3].metric(
        "Max deviation from true curve",
        f"{setup.max_deviation_pct:.1f}%",
        f"limit {setup.deviation_limit_pct:g}%",
        delta_color="off",
    )
    steady_cols = st.columns(4)
    steady_cols[0].metric("Steady-state Cavg", f"{setup.predicted_cavg_mg_l:.1f} mg/L")
    steady_cols[1].metric(
        "Steady-state Cmax", f"{setup.predicted_cmax_mg_l:.1f} mg/L", f"true curve {setup.reference_cmax_mg_l:.1f}", delta_color="off"
    )
    steady_cols[2].metric(
        "Steady-state Cmin", f"{setup.predicted_cmin_mg_l:.1f} mg/L", f"true curve {setup.reference_cmin_mg_l:.1f}", delta_color="off"
    )
    steady_cols[3].metric(
        "Achieved half-life", f"{setup.apparent_half_life_h:.2f} h", f"target {setup.long_half_life_h:g} h", delta_color="off"
    )
    if setup.feasible:
        st.success(setup.message)
    else:
        st.warning(setup.message)
    st.info(_blaser_setup_notes(setup, central_dose_volume_ml, extra_dose_volume_ml, dosing_frequency_h, duration_h))

    system = SystemConfig(
        central_bottle_ml=central_bottle_ml,
        cartridge_ml=cartridge_ml,
        extra_volume_ml=setup.extra_volume_ml,
        q_extra_to_central_ml_min=setup.q_extra_to_central_ml_min,
        q_extra_diluent_ml_min=setup.q_extra_diluent_ml_min,
        q_central_diluent_ml_min=setup.q_central_diluent_ml_min,
        q_waste_set_ml_min=setup.q_waste_ml_min,
    )
    fos = FosfomycinConfig(
        drug_name=setup_drug_name,
        central_stock_mg_ml=setup.central_stock_mg_ml,
        extra_stock_mg_ml=setup.extra_stock_mg_ml,
        central_infusion_ml_min=central_dose_volume_ml / infusion_min,
        extra_infusion_ml_min=extra_dose_volume_ml / infusion_min,
        infusion_duration_min=infusion_min,
        dosing_interval_min=interval_min,
        preload_extra_mg=0.0,
    )
    drugs = [
        DrugConfig(
            name,
            target_concentration_mg_l=values["target_concentration_mg_l"],
            half_life_h=values["half_life_h"],
            dosing_mode=values["dosing_mode"],
            loading_target_concentration_mg_l=values["loading_target_concentration_mg_l"],
            loading_duration_h=values["loading_duration_h"],
            loading_volume_ml=values["loading_volume_ml"],
            intermittent_interval_h=values["dosing_frequency_h"] or 6.0,
            intermittent_duration_h=values["maintenance_duration_h"],
        )
        for name, values in drug_inputs.items()
        if name != setup_drug_name
    ]
    result = simulate_hfim("blaser", system, fos, drugs, duration_h=duration_h, dt_min=dt_min)
    summary = result.summary[setup_drug_name]

    st.subheader("4. Setup and pump settings")
    apparatus_fig = _plot_two_half_life_apparatus(
        _blaser_apparatus_view(system, fos, drug_inputs, result.summary, duration_h, recirculation_ml_min, setup)
    )
    st.image(_figure_export_bytes(apparatus_fig, "png", dpi=300), width="stretch")
    st.caption(
        "Volumes on the waste and diluent bottles are totals for the whole run. The central compartment is "
        "magnetically stirred. Both outflow tubes sit near the bottom of their bottle, and both bottles are vented."
    )
    _render_schematic_export_buttons(st, apparatus_fig, "apparatus_blaser", "hfim-apparatus-blaser")
    st.markdown("**Pump settings (set once, never changed during the run)**")
    st.dataframe(_blaser_pump_rows(setup, duration_h), width="stretch", hide_index=True, column_config={"Why": st.column_config.Column(width="large")})

    st.subheader("5. Result overview")
    target_auc = target_css_mg_l * 24
    result_cols = st.columns(4)
    result_cols[0].metric("Steady-state AUC per 24 h", f"{summary['central_auc_last_24h_mg_h_l']:.1f}", f"target {target_auc:g}")
    result_cols[1].metric(
        "Steady-state Cmax", f"{summary['central_cmax_last_24h_mg_l']:.1f} mg/L", f"true curve {setup.reference_cmax_mg_l:.1f}", delta_color="off"
    )
    result_cols[2].metric(
        "Steady-state Cmin", f"{summary['central_cmin_last_24h_mg_l']:.1f} mg/L", f"true curve {setup.reference_cmin_mg_l:.1f}", delta_color="off"
    )
    result_cols[3].metric("AUC0-24 (first day)", f"{summary['central_auc_0_24_mg_h_l']:.1f}")
    st.caption(
        "Steady-state values come from the last 24 h of the run. The first day is lower because the drug is still "
        "accumulating, exactly as it would in a patient starting the same regimen."
        + ("" if duration_h >= 48 else " Run at least 48 h to see the settled values.")
    )

    st.subheader("6. PK concentration")
    st.pyplot(_plot_static(
        result.rows,
        [setup_drug_name],
        f"{setup_drug_name} central and extra concentration",
        include_extra=True,
        reference=_true_one_compartment_curve(target_css_mg_l, setup.long_half_life_h, interval_min, infusion_min, duration_h, dt_min),
    ))
    st.caption(
        f"Dotted line: the true one-compartment curve for a {setup.long_half_life_h:g} h half-life with the same dose "
        "timing. Central and extra are dosed together and fall together, which is what gives central the longer half-life."
    )
    central_drugs = [drug.name for drug in drugs]
    if central_drugs:
        st.pyplot(_plot_static(result.rows, central_drugs, "Central concentration for loading/infusion drugs", include_extra=False))
        st.caption(
            "Small ripple on continuous-infusion drugs comes from central volume rising and settling by "
            f"{setup.central_volume_range_ml[1] - setup.central_volume_range_ml[0]:.1f} mL around each dose."
        )

    st.subheader("7. Preparation and weighing plan")
    st.markdown(f"**{setup_drug_name} dose syringes**")
    st.dataframe(
        _blaser_dose_rows(setup, fos, central_dose_volume_ml, extra_dose_volume_ml, duration_h),
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Solutions to prepare**")
    st.dataframe(
        [
            _solution_volume_row("Central diluent", setup.q_central_diluent_ml_min, duration_h),
            _solution_volume_row("Extra diluent", setup.q_extra_diluent_ml_min, duration_h),
        ],
        width="stretch",
        hide_index=True,
    )
    central_diluent_ci_rows = _central_diluent_reservoir_rows(result.summary, duration_h)
    if central_diluent_ci_rows:
        st.markdown("**Central diluent q24h shared reservoir recipe**")
        st.caption("Continuous-infusion drugs are mixed into the central diluent reservoir, which is replaced every 24 h.")
        st.dataframe(central_diluent_ci_rows, width="stretch", hide_index=True, column_config={"Note": st.column_config.Column(width="large")})
    other_prep = [
        row for row in _format_preparation_rows(result.summary["drug_preparation"]) if row["Drug"] != setup_drug_name
    ]
    if other_prep:
        st.markdown(f"**{_prep_group_title(other_prep)}**")
        st.dataframe(other_prep, width="stretch", hide_index=True, column_config={"Note": st.column_config.Column(width="large")})

    st.subheader("8. Equations")
    st.markdown(_blaser_equation_text(setup, central_dose_volume_ml, extra_dose_volume_ml, infusion_min, interval_min))


def _blaser_setup_notes(setup, central_dose_volume_ml: float, extra_dose_volume_ml: float, interval_h: float, duration_h: float) -> str:
    central_low, central_high = setup.central_volume_range_ml
    extra_low, extra_high = setup.extra_volume_range_ml
    return (
        f"Why two pumps are set above the textbook values: each q{interval_h:g}h dose adds {central_dose_volume_ml:g} mL to central "
        f"and {extra_dose_volume_ml:g} mL to extra. With the textbook flows ({setup.uncorrected_q_extra_to_central_ml_min:.3f} and "
        f"{setup.uncorrected_q_waste_ml_min:.3f} mL/min) that volume is never removed: after {duration_h:g} h the vessels would hold "
        f"{setup.uncorrected_end_volumes_ml[0]:.0f} mL and {setup.uncorrected_end_volumes_ml[1]:.0f} mL and the curve would be "
        f"{setup.uncorrected_end_deviation_pct:.0f}% off. Here the extra-to-central pump and the waste pump are set once to also "
        f"carry the dose volume, averaged over the interval. Central then moves between {central_low:.0f} and {central_high:.0f} mL "
        f"and extra between {extra_low:.0f} and {extra_high:.0f} mL each interval, and returns to the start every time. "
        f"Doses are {(setup.dose_scale - 1) * 100:+.1f}% versus the textbook amounts to keep the daily AUC exact. "
        "Bench notes: keep both outflow tubes near the bottom of the bottle, vent both bottles, leave headroom for the rise, "
        "and mark the starting liquid level so pump drift shows up early."
    )


def _blaser_pump_rows(setup, duration_h: float) -> list[dict]:
    def row(name: str, flow: float, textbook: float, why: str) -> dict:
        return {
            "Pump": name,
            "Set to": f"{flow:.3f} mL/min ({flow * 60:.2f} mL/h)",
            "Textbook Blaser": f"{textbook:.3f} mL/min",
            f"Volume in {duration_h:g} h": f"{flow * duration_h * 60:,.0f} mL",
            "Why": why,
        }

    return [
        row("Extra diluent -> extra", setup.q_extra_diluent_ml_min, setup.q_extra_diluent_ml_min,
            "washes the extra compartment out at the long half-life"),
        row("Extra -> central", setup.q_extra_to_central_ml_min, setup.uncorrected_q_extra_to_central_ml_min,
            "extra diluent flow plus the extra dose volume averaged over the dosing interval"),
        row("Central diluent -> central", setup.q_central_diluent_ml_min, setup.q_central_diluent_ml_min,
            "the rest of the inflow needed for the short central half-life; continuous-infusion drugs are mixed into it"),
        row("Central -> waste", setup.q_waste_ml_min, setup.uncorrected_q_waste_ml_min,
            "all continuous inflow plus both dose volumes averaged over the dosing interval"),
    ]


def _blaser_dose_rows(setup, fos: FosfomycinConfig, central_dose_volume_ml: float, extra_dose_volume_ml: float, duration_h: float) -> list[dict]:
    interval_h = fos.dosing_interval_min / 60
    duration_infusion_h = fos.infusion_duration_min / 60
    doses_per_day = 24 / interval_h
    total_doses = math.ceil(duration_h / interval_h)

    def row(destination: str, dose_mg: float, volume_ml: float, concentration: float) -> dict:
        return {
            "Into": destination,
            "Timing": f"q{interval_h:g}h, {volume_ml:g} mL over {duration_infusion_h:g} h ({volume_ml / duration_infusion_h:.2f} mL/h)",
            "Syringe concentration": f"{concentration:.6f} mg/mL",
            "Per dose": f"{dose_mg:.3f} mg",
            "Per day": f"{dose_mg * doses_per_day:.3f} mg in {volume_ml * doses_per_day:g} mL",
            "Per day +10%": f"{dose_mg * doses_per_day * 1.10:.3f} mg in {volume_ml * doses_per_day * 1.10:.1f} mL",
            f"{duration_h:g} h total +10%": f"{dose_mg * total_doses * 1.10:.1f} mg in {volume_ml * total_doses * 1.10:.1f} mL",
        }

    return [
        row("Central compartment", setup.central_dose_mg, central_dose_volume_ml, setup.central_stock_mg_ml),
        row("Extra compartment (same time)", setup.extra_dose_mg, extra_dose_volume_ml, setup.extra_stock_mg_ml),
    ]


def _blaser_apparatus_view(
    system: SystemConfig,
    fos: FosfomycinConfig,
    drug_inputs: dict,
    summary: dict,
    duration_h: float,
    recirculation_ml_min: float,
    setup,
) -> dict:
    interval_h = fos.dosing_interval_min / 60
    infusion_h = fos.infusion_duration_min / 60
    central_volume_ml = fos.central_infusion_ml_min * fos.infusion_duration_min
    extra_volume_ml = fos.extra_infusion_ml_min * fos.infusion_duration_min
    central_groups = [(
        f"{fos.drug_name} central q{interval_h:g}h",
        [
            f"hour 0-{infusion_h:g} of every q{interval_h:g}h dose",
            f"{fos.central_infusion_ml_min * 60:.2f} mL/h",
            f"{fos.central_dose_mg:.2f} mg in {central_volume_ml:g} mL",
            f"syringe {fos.central_stock_mg_ml:.2f} mg/mL",
        ],
    )]
    central_groups.extend(_central_drug_apparatus_groups(drug_inputs, summary, skip=fos.drug_name))
    extra_groups = [(
        f"{fos.drug_name} extra q{interval_h:g}h",
        [
            "same time as the central dose",
            f"{fos.extra_infusion_ml_min * 60:.2f} mL/h",
            f"{fos.extra_dose_mg:.2f} mg in {extra_volume_ml:g} mL",
            f"syringe {fos.extra_stock_mg_ml:.2f} mg/mL",
            f"central + extra: {(fos.central_dose_mg + fos.extra_dose_mg) * 24 / interval_h:.2f} mg/day",
        ],
    )]
    return {
        "title": "HFIM apparatus - 2 half life (Blaser)",
        "subtitle": (
            f"{fos.drug_name} at {setup.long_half_life_h:g} h via the extra compartment   |   central washout "
            f"{setup.short_half_life_h:g} h   |   {duration_h:g} h run"
        ),
        "cartridge_volume": f"{system.cartridge_ml:g} mL",
        "central_volume": f"{system.central_bottle_ml:g} mL",
        "extra_volume": f"{system.extra_volume_ml:.0f} mL",
        "waste_total": f"{system.q_waste_ml_min * duration_h * 60:,.0f} mL",
        "central_diluent_total": f"{system.q_central_diluent_ml_min * duration_h * 60:,.0f} mL",
        "extra_diluent_total": f"{system.q_extra_diluent_ml_min * duration_h * 60:,.0f} mL",
        "recirculation": f"{recirculation_ml_min:g} mL/min",
        "waste_flow": f"{system.q_waste_ml_min:.3f} mL/min",
        "extra_to_central_flow": f"{system.q_extra_to_central_ml_min:.3f} mL/min",
        "central_diluent_flow": f"{system.q_central_diluent_ml_min:.3f} mL/min",
        "extra_diluent_flow": f"{system.q_extra_diluent_ml_min:.3f} mL/min",
        "show_extra_diluent": True,
        "central_injection_groups": central_groups,
        "extra_injection_groups": extra_groups,
    }


def _blaser_equation_text(setup, central_dose_volume_ml: float, extra_dose_volume_ml: float, infusion_min: int, interval_min: int) -> str:
    vc = setup.central_volume_ml
    interval_h = interval_min / 60
    q_short = setup.uncorrected_q_waste_ml_min
    lines = [
        "1. Textbook Blaser flows (two half-lives, fixed volumes)",
        f"   Vc = central bottle + cartridge = {vc:g} mL",
        f"   Q_short = ln(2) x Vc / (short t1/2 x 60) = ln(2) x {vc:g} / ({setup.short_half_life_h:g} x 60) = {q_short:.6g} mL/min",
        f"   Q_long  = ln(2) x Vc / (long t1/2 x 60)  = ln(2) x {vc:g} / ({setup.long_half_life_h:g} x 60) = {setup.q_central_diluent_ml_min:.6g} mL/min",
        f"   Central diluent pump = Q_long = {setup.q_central_diluent_ml_min:.6g} mL/min",
        f"   Extra diluent pump   = Q_short - Q_long = {setup.q_extra_diluent_ml_min:.6g} mL/min",
        f"   Extra volume = Vc x (Q_short - Q_long) / Q_long = {setup.extra_volume_ml:.1f} mL",
        "   With these, extra washes out at the long half-life and central follows it for the dosed drug,",
        "   while drugs dosed into central only still wash out at the short half-life.",
        "",
        "2. Outflow pumps, set once to also remove the dose volumes",
        f"   Extra dose flow averaged over the interval   = {extra_dose_volume_ml:g} mL / {interval_min:g} min = {extra_dose_volume_ml / interval_min:.6g} mL/min",
        f"   Central dose flow averaged over the interval = {central_dose_volume_ml:g} mL / {interval_min:g} min = {central_dose_volume_ml / interval_min:.6g} mL/min",
        f"   Extra -> central pump = {setup.q_extra_diluent_ml_min:.6g} + {extra_dose_volume_ml / interval_min:.6g} = {setup.q_extra_to_central_ml_min:.6g} mL/min",
        f"   Waste pump = {q_short:.6g} + {central_dose_volume_ml / interval_min:.6g} + {extra_dose_volume_ml / interval_min:.6g} = {setup.q_waste_ml_min:.6g} mL/min",
        "   Over one interval each vessel gains its dose volume during the infusion and loses the same",
        "   volume afterwards, so nothing accumulates.",
        "",
        "3. Doses",
        f"   Concentration step per dose = Cavg x ln(2) / long t1/2 x interval = {setup.target_css_mg_l:g} x ln(2) / {setup.long_half_life_h:g} x {interval_h:g}",
        "   Textbook central dose = step x Vc;  textbook extra dose = step x extra volume (same step in both)",
        f"   Scale so simulated steady-state Cavg = target: x {setup.dose_scale:.4f}",
        f"   Central dose = {setup.central_dose_mg:.3f} mg in {central_dose_volume_ml:g} mL = {setup.central_stock_mg_ml:.6g} mg/mL",
        f"   Extra dose   = {setup.extra_dose_mg:.3f} mg in {extra_dose_volume_ml:g} mL = {setup.extra_stock_mg_ml:.6g} mg/mL",
        "",
        "4. Continuous-infusion drugs in the central diluent",
        "   CI input mg/min = Css x waste pump / 1000",
        "   Central diluent concentration = CI input mg/min / central diluent pump",
        "",
        "5. Differential equations used during each time step",
        "   dAextra/dt   = extra dose input - Q(extra->central) x Cextra",
        "   dAcentral/dt = central dose input + Q(extra->central) x Cextra - Q(waste) x Ccentral",
        "   dVextra/dt   = extra diluent + extra dose flow - Q(extra->central)",
        "   dVcentral/dt = Q(extra->central) + central diluent + central dose flow - Q(waste)",
        "   C = A / V with the volume of that moment",
    ]
    return "```text\n" + "\n".join(lines) + "\n```"


# ---------------------------------------------------------------------------
# "1 half life" page: single drug, or a combination where every drug shares one
# half-life. Central compartment only - no extra compartment exists, so central
# washout comes straight from that one half-life with no shortest-half-life
# compromise, and Cmax is a reported outcome rather than a solvable target.
# ---------------------------------------------------------------------------

_ONE_TARGET_TYPES = ["Maintain concentration", "AUC0-24 exposure"]
_ONE_MAINTENANCE = ["continuous infusion", "intermittent infusion", "no maintenance"]
_PEAK_SHAPING_INTERVALS_H = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0)


def _one_half_life_drug_defaults(index: int) -> dict:
    presets = [
        {"name": "imipenem", "target_type": "Maintain concentration", "target_value": 9.0, "loading_dose": True, "maintenance": "continuous infusion"},
        {"name": "relebactam", "target_type": "Maintain concentration", "target_value": 6.0, "loading_dose": True, "maintenance": "continuous infusion"},
        {"name": "meropenem", "target_type": "Maintain concentration", "target_value": 16.0, "loading_dose": False, "maintenance": "intermittent infusion"},
    ]
    if index < len(presets):
        return presets[index]
    return {"name": f"drug{index + 1}", "target_type": "Maintain concentration", "target_value": 10.0, "loading_dose": False, "maintenance": "continuous infusion"}


def _one_half_life_drug_panel(st, active_drug_count: int, shared_half_life_h: float) -> dict[str, dict]:
    """Drug cards for the shared-half-life page: no per-drug half-life, plus a dose volume for
    intermittent drugs so the bench gets a stock concentration rather than only mg per dose."""
    selected: dict[str, dict] = {}
    for index in range(active_drug_count):
        default = _one_half_life_drug_defaults(index)
        with st.container(border=True):
            st.markdown(f"**Drug {index + 1}**")
            cols = st.columns(4)
            raw_name = cols[0].text_input("Drug name", value=default["name"], key=f"one_name_{index}")
            name = _normalized_unique_drug_name(raw_name, index, set(selected))
            if raw_name.strip().lower() != name:
                st.caption(f"Internal simulation name: {name}")
            target_type = cols[1].selectbox(
                "Simulation target",
                _ONE_TARGET_TYPES,
                index=_ONE_TARGET_TYPES.index(default["target_type"]),
                key=f"one_target_type_{index}",
                help="Cmax is not a target on this page - it follows from the interval and the shared half-life.",
            )
            target_label = "Target Css (mg/L)" if target_type == "Maintain concentration" else "Target AUC0-24 (mg*h/L)"
            target_value = cols[2].number_input(target_label, min_value=0.0, value=float(default["target_value"]), step=1.0, key=f"one_target_value_{index}")
            cols[3].metric("Half-life", f"{shared_half_life_h:g} h")

            dosing_cols = st.columns(3)
            loading_dose = dosing_cols[0].checkbox("Loading dose", value=default["loading_dose"], key=f"one_loading_{index}")
            maintenance = dosing_cols[1].selectbox(
                "Maintenance dosing",
                _ONE_MAINTENANCE,
                index=_ONE_MAINTENANCE.index(default["maintenance"]),
                key=f"one_maintenance_{index}",
            )
            dosing_frequency_h = dosing_cols[2].number_input(
                "Dosing frequency (h)",
                min_value=0.0,
                value=8.0 if maintenance == "intermittent infusion" else 0.0,
                step=1.0,
                disabled=maintenance != "intermittent infusion",
                help="Only intermittent infusion uses a q-hour dosing interval.",
                key=f"one_frequency_{index}",
            )

            loading_target = 0.0
            loading_duration_h = 0.0
            loading_volume_ml = 5.0
            maintenance_duration_h = 0.5
            dose_volume_ml = 8.0
            detail_cols = st.columns(4)
            if loading_dose:
                loading_target = detail_cols[0].number_input(
                    "Loading target (mg/L)",
                    min_value=0.0,
                    value=target_value * (1.0 if target_type == "Maintain concentration" else 1.0 / 24.0) * 2.0,
                    step=1.0,
                    key=f"one_loading_target_{index}",
                )
                loading_duration_h = detail_cols[1].number_input("Loading infusion duration (h)", min_value=0.0, value=0.5, step=0.25, key=f"one_loading_duration_{index}")
                loading_volume_ml = detail_cols[2].number_input("Loading dose volume (mL)", min_value=0.01, value=5.0, step=0.5, key=f"one_loading_volume_{index}")
            if maintenance == "intermittent infusion":
                maintenance_duration_h = detail_cols[3].number_input("Maintenance infusion duration (h)", min_value=0.01, value=0.5, step=0.25, key=f"one_maintenance_duration_{index}")
                dose_volume_ml = st.number_input(
                    "Dose volume per intermittent dose (mL)",
                    min_value=0.01,
                    value=8.0,
                    step=0.5,
                    help="Volume pumped per dose. Sets the stock concentration you actually prepare, not the mg per dose.",
                    key=f"one_dose_volume_{index}",
                )

            selected[name] = {
                "target_type": target_type,
                "target_value": target_value,
                "target_concentration_mg_l": _target_to_concentration(target_type, target_value),
                "half_life_h": shared_half_life_h,
                "dosing_mode": _dosing_mode_from_controls(loading_dose, maintenance),
                "loading_dose": loading_dose,
                "maintenance": maintenance,
                "dosing_frequency_h": dosing_frequency_h,
                "loading_target_concentration_mg_l": loading_target if loading_dose else None,
                "loading_duration_h": loading_duration_h,
                "loading_volume_ml": loading_volume_ml,
                "maintenance_duration_h": maintenance_duration_h,
                "dose_volume_ml": dose_volume_ml if maintenance == "intermittent infusion" else None,
            }
    return selected


def _peak_shaping_rows(drug_inputs: dict, summary: dict, shared_half_life_h: float) -> list[dict]:
    """What Cmax/Cmin each candidate interval would give at the same Cavg.

    Cavg is the only target this page can hit, so the dosing interval is the lever that shapes the
    peak. Listing the trade-off makes that second knob visible instead of leaving the user to guess
    why there is no Target Cmax box.
    """
    rows = []
    for name, values in drug_inputs.items():
        if values.get("maintenance") != "intermittent infusion":
            continue
        item = summary.get(name)
        if not isinstance(item, dict):
            continue
        cavg = item.get("target_concentration_mg_l", 0.0)
        duration_h = item.get("intermittent_duration_h") or 0.5
        current_interval = item.get("intermittent_interval_h")
        row: dict[str, str] = {"Drug": name, "Target Cavg": f"{cavg:.1f} mg/L"}
        for interval_h in _PEAK_SHAPING_INTERVALS_H:
            cmax, cmin = intermittent_peak_trough(cavg, shared_half_life_h, interval_h, duration_h)
            marker = " *" if current_interval and abs(interval_h - current_interval) < 1e-9 else ""
            row[f"q{interval_h:g}h"] = f"{cmax:.1f} / {cmin:.1f}{marker}"
        rows.append(row)
    return rows


def _one_half_life_equation_text(
    system: SystemConfig,
    shared_half_life_h: float,
    duration_h: float,
    drug_inputs: dict,
    summary: dict,
) -> str:
    q = system.q_central_diluent_ml_min
    lines = [
        "1. Central effective volume",
        f"   Vc = central bottle + cartridge = {system.central_bottle_ml:g} + {system.cartridge_ml:g} = {system.central_volume_ml:g} mL",
        "",
        "2. Central washout from the one shared half-life",
        "   Q = ln(2) x Vc / (t1/2 x 60)",
        f"   Q = ln(2) x {system.central_volume_ml:g} / ({shared_half_life_h:g} x 60) = {q:.6g} mL/min",
        "   Every drug shares this one flow, so there is no shortest-half-life compromise and no",
        "   extra compartment is needed to reshape any drug's apparent half-life.",
        "",
        "3. Maintenance dose from the target concentration",
        "   Continuous infusion:  rate mg/min = Css x Q / 1000",
        "   Central diluent reservoir concentration = rate mg/min / Q",
        "   Intermittent infusion:  dose per interval = Css x Q x interval / 1000",
        "",
        "4. Resulting peak and trough (not independent targets)",
        "   Cmax,ss = Cavg x (interval / infusion duration) x (1 - e^-kT) / (1 - e^-k*tau)",
        "   Cmin,ss = Cmax,ss x e^-k(tau - T),  where k = ln(2) / t1/2",
        "",
        "5. Solution volume to prepare",
        f"   Central diluent per 24 h = Q x 1440 = {q:.6g} x 1440 = {q * 1440:.1f} mL",
        f"   Central diluent per 24 h with 10% extra = {q * 1440 * 1.10:.1f} mL",
        f"   Central diluent for {duration_h:g} h = {q * duration_h * 60:.1f} mL",
        "",
        "6. Per-drug numbers",
    ]
    for name, values in drug_inputs.items():
        item = summary.get(name)
        if not isinstance(item, dict):
            continue
        lines.append(f"   {name}: target {item.get('target_concentration_mg_l', 0):g} mg/L, mode = {item.get('dosing_mode')}")
        if values.get("loading_dose"):
            lines.append(f"      loading dose = {item.get('loading_dose_mg', 0):.3f} mg in {item.get('loading_volume_ml', 0):g} mL")
        if values.get("maintenance") == "continuous infusion":
            lines.append(f"      CI rate = {item.get('infusion_rate_mg_h', 0):.3f} mg/h -> {item.get('daily_amount_mg', 0):.3f} mg/day")
        if values.get("maintenance") == "intermittent infusion":
            lines.append(
                f"      q{item.get('intermittent_interval_h', 0):g}h dose = {item.get('intermittent_dose_mg', 0):.3f} mg"
                f" over {item.get('intermittent_duration_h', 0):g} h"
            )
    lines.extend([
        "",
        "7. Differential equation used during each time step",
        "   Ccentral = Acentral / Vcentral",
        "   dAcentral/dt = dosing input - Q x Ccentral",
    ])
    return "```text\n" + "\n".join(lines) + "\n```"


def _page_one_half_life() -> None:
    import streamlit as st

    st.title("HFIM PK Simulator - 1 half life")
    st.caption(
        "Single drug, or a combination where every drug shares one half-life. Central compartment only: "
        "central washout comes straight from that shared half-life, so no extra compartment is needed."
    )

    st.subheader("1. Simulation setup")
    setup_cols = st.columns(3)
    active_drug_count = int(setup_cols[0].number_input(
        "Number of drugs", min_value=1, max_value=6, value=1, step=1, key="one_drug_count",
        help="Works for a single drug as well as a combination, as long as they share one half-life.",
    ))
    duration_h = setup_cols[1].number_input("Simulation duration (h)", min_value=24.0, value=168.0, step=24.0, key="one_duration")
    dt_min = setup_cols[2].number_input("Time step (min)", min_value=0.25, value=1.0, step=0.25, key="one_dt")

    with st.expander("Compartment and flow settings", expanded=True):
        st.markdown(
            "Central washout is set directly by the one shared half-life: **Q = ln(2) x Vc / t1/2**. "
            "There is no extra compartment and no extra-to-central transfer on this page."
        )
        cols = st.columns(4)
        central_bottle_ml = cols[0].number_input("Central bottle (mL)", min_value=1.0, value=100.0, step=5.0, key="one_bottle")
        cartridge_ml = cols[1].number_input("Cartridge (mL)", min_value=1.0, value=70.0, step=5.0, key="one_cartridge")
        shared_half_life_h = cols[2].number_input(
            "Shared half-life (h)", min_value=0.01, value=1.25, step=0.05, key="one_half_life",
            help="Every drug on this page uses this half-life. Use the 2 half life page when they differ.",
        )
        recirculation_ml_min = cols[3].number_input(
            "Cartridge recirculation (mL/min)", min_value=1.0, value=120.0, step=5.0, key="one_recirc",
            help="Shown on the diagram only; it does not feed the PK calculation.",
        )
        central_volume_ml = central_bottle_ml + cartridge_ml
        auto_flow = flow_for_half_life(central_volume_ml, shared_half_life_h)
        flow_cols = st.columns(3)
        flow_mode = flow_cols[0].selectbox("Flow setup mode", ["Auto flow from shared half-life", "Manual flow entry"], key="one_flow_mode")
        auto_flow_mode = flow_mode.startswith("Auto")
        q_central = flow_cols[1].number_input(
            "Central diluent = central outflow (mL/min)",
            min_value=0.0,
            value=auto_flow,
            step=0.001,
            format="%.3f",
            disabled=auto_flow_mode,
            key=f"one_qcentral_{'auto' if auto_flow_mode else 'manual'}_{central_volume_ml:.1f}_{shared_half_life_h:.3f}",
        )
        achieved = half_life_for_flow(central_volume_ml, q_central) if q_central > 0 else None
        flow_cols[2].metric("Achieved half-life", f"{achieved:.2f} h" if achieved else "not defined")
        st.info(
            f"Vc = {central_bottle_ml:g} + {cartridge_ml:g} = {central_volume_ml:g} mL. "
            f"Q = ln(2) x {central_volume_ml:g} mL / ({shared_half_life_h:g} h x 60) = {auto_flow:.3f} mL/min. "
            f"Inflow and outflow are matched, so the central volume stays fixed."
        )

    st.subheader("2. Drug targets and dosing")
    st.caption(
        "Set a Css or AUC0-24 target per drug. Cmax is not an input here: with one shared washout it follows "
        "from the dosing interval and the half-life, and Section 5 shows how to shape it."
    )
    drug_inputs = _one_half_life_drug_panel(st, active_drug_count, shared_half_life_h)

    system = SystemConfig(
        central_bottle_ml=central_bottle_ml,
        cartridge_ml=cartridge_ml,
        extra_volume_ml=1.0,
        q_extra_to_central_ml_min=0.0,
        q_extra_diluent_ml_min=0.0,
        q_central_diluent_ml_min=q_central,
    )
    drugs = [
        DrugConfig(
            name,
            target_concentration_mg_l=values["target_concentration_mg_l"],
            half_life_h=values["half_life_h"],
            dosing_mode=values["dosing_mode"],
            loading_target_concentration_mg_l=values["loading_target_concentration_mg_l"],
            loading_duration_h=values["loading_duration_h"],
            loading_volume_ml=values["loading_volume_ml"],
            intermittent_interval_h=values["dosing_frequency_h"] or 8.0,
            intermittent_duration_h=values["maintenance_duration_h"],
        )
        for name, values in drug_inputs.items()
    ]
    result = simulate_hfim("central_only", system, None, drugs, duration_h=duration_h, dt_min=dt_min)

    st.subheader("3. Setup and injection overview")
    apparatus_fig = _plot_one_half_life_apparatus(_one_half_life_apparatus_view(
        system, drug_inputs, result.summary, duration_h, recirculation_ml_min, shared_half_life_h,
    ))
    st.image(_figure_export_bytes(apparatus_fig, "png", dpi=300), width="stretch")
    st.caption(
        "Volumes on the waste and diluent bottles are totals for the whole run. Concentrations are shown "
        "in µg/mL. The central compartment is magnetically stirred."
    )
    _render_schematic_export_buttons(st, apparatus_fig, "apparatus1", "hfim-apparatus-1-half-life")

    st.markdown("**System solution volumes**")
    st.dataframe(_solution_volume_rows(q_central, 0.0, "central_only", duration_h), width="stretch", hide_index=True)

    st.subheader("4. Result overview")
    for name in drug_inputs:
        item = result.summary.get(name)
        if not isinstance(item, dict):
            continue
        drug_rows = [row for row in result.rows if row["drug"] == name]
        window = [row for row in drug_rows if row["time_h"] >= max(0.0, duration_h - 24)]
        cavg = sum(row["central_mg_l"] for row in window) / len(window) if window else 0.0
        cmax = max((row["central_mg_l"] for row in window), default=0.0)
        cmin = min((row["central_mg_l"] for row in window), default=0.0)
        target = item.get("target_concentration_mg_l", 0.0)
        st.markdown(f"**{name}**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Target Css", f"{target:.2f} mg/L")
        metric_cols[1].metric("Simulated Cavg", f"{cavg:.2f} mg/L", f"{cavg - target:+.2f}")
        metric_cols[2].metric("Resulting Cmax", f"{cmax:.2f} mg/L")
        metric_cols[3].metric("Resulting Cmin", f"{cmin:.2f} mg/L")

    peak_rows = _peak_shaping_rows(drug_inputs, result.summary, shared_half_life_h)
    if peak_rows:
        with st.expander("How to shape Cmax without changing Cavg", expanded=True):
            st.caption(
                "Cmax cannot be entered as a target here - the dosing interval is the lever. Each cell shows "
                "Cmax / Cmin at that interval while holding the same Cavg. The current setting is marked *."
            )
            st.dataframe(peak_rows, width="stretch", hide_index=True)

    st.subheader("5. PK concentration")
    st.pyplot(_plot_static(result.rows, list(drug_inputs), "Central concentration", include_extra=False))

    st.subheader("6. Preparation and weighing plan")
    _render_preparation_styles(st)
    prep_rows = _format_preparation_rows(result.summary["drug_preparation"])
    review_rows = _preparation_review_rows(prep_rows, result.summary, system, None, duration_h)
    st.markdown("**Final preparation review**")
    st.caption("Bench checklist: which drug goes into which dosing part, with the amount to weigh.")
    st.dataframe(
        review_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Drug": st.column_config.Column(width="small"),
            "Add into": st.column_config.Column(width="medium"),
            "Dosing part": st.column_config.Column(width="medium"),
            "Note": st.column_config.Column(width="large"),
        },
    )
    central_diluent_ci_rows = _central_diluent_reservoir_rows(result.summary, duration_h)
    if central_diluent_ci_rows:
        recipe = _central_diluent_reservoir_summary(result.summary, duration_h)
        st.markdown("**Central diluent q24h shared reservoir recipe**")
        st.caption("One shared reservoir every 24 h; continuous-infusion drugs are mixed into this same volume.")
        recipe_cols = st.columns(4)
        recipe_cols[0].metric("Required volume q24h", recipe["volume_q24h"])
        recipe_cols[1].metric("10% extra volume q24h", recipe["extra_volume_q24h_10_percent"])
        recipe_cols[2].metric("Total to prepare q24h", recipe["prepared_volume_q24h"])
        recipe_cols[3].metric(f"Total to prepare {duration_h:g} h", recipe["prepared_volume_total"])
        st.dataframe(
            central_diluent_ci_rows,
            width="stretch",
            hide_index=True,
            column_config={"Note": st.column_config.Column(width="large")},
        )

    st.subheader("7. Equations")
    st.markdown(_one_half_life_equation_text(system, shared_half_life_h, duration_h, drug_inputs, result.summary))

    st.subheader("8. HFIM Setup Assistant")
    agent_context = build_agent_context(
        system={
            "central_volume_ml": system.central_volume_ml,
            "extra_volume_ml": 0.0,
            "q_extra_to_central_ml_min": 0.0,
            "q_extra_diluent_ml_min": 0.0,
            "q_central_diluent_ml_min": q_central,
            "scenario": "central_only",
            "shared_half_life_h": shared_half_life_h,
        },
        setup_drug_name=next(iter(drug_inputs), ""),
        drug_inputs=drug_inputs,
        summary={name: value for name, value in result.summary.items() if name != "drug_preparation"},
    )
    _setup_assistant_panel(st, agent_context)

    if st.button("Run and save to SQLite", key="one_save"):
        store = SimulationStore(Path("data") / "hfim-simulations.sqlite")
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = store.create_run("central_only", started_at, {
            "scenario": "central_only",
            "duration_h": duration_h,
            "dt_min": dt_min,
            "shared_half_life_h": shared_half_life_h,
            "central_volume_ml": system.central_volume_ml,
            "q_central_diluent_ml_min": q_central,
            "drugs": drug_inputs,
        })
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
        store.finish_run(run_id, "success", datetime.now(timezone.utc).isoformat(), f"timepoints={counts}; prep={prep_counts}")
        st.success(f"Saved run {run_id} to data/hfim-simulations.sqlite")


if __name__ == "__main__":
    main()
