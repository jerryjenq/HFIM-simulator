from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import math
from pathlib import Path

from .agent import ask_setup_agent, build_agent_context
from .pk import DrugConfig, FosfomycinConfig, SystemConfig, flow_for_half_life, half_life_for_flow, simulate_hfim, solve_css_cmax_replacement
from .store import SimulationStore


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="HFIM PK Simulator", layout="wide")
    st.title("HFIM PK Simulator")
    st.caption("Enter experimental conditions, simulate central and extra-compartment PK concentrations, and estimate how much drug to prepare.")

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
        qextra_label = "Extra to central fixed transfer (mL/min)" if scenario == "q24_replacement" else "Extra to central (mL/min)"
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

    st.subheader("2. Drug targets and injection settings")
    st.caption("Choose the number of drugs, then set whether each drug should target AUC0-24, Cmax, or maintained Css. Dosing settings are split into loading dose, maintenance dosing, and dosing frequency.")
    drug_inputs = _drug_input_panel(st, active_drug_count)

    st.subheader("3. Css/Cmax solver and replacement setup" if scenario == "q24_replacement" else "3. Intermittent / extra-compartment drug setup")
    st.caption(
        "This section solves central stock and q24h extra replacement concentration from target Css/Cavg and Cmax."
        if scenario == "q24_replacement"
        else "This section corresponds to the drug-injection lines in the HFIM setup. Select which drug uses this central/extra setup, then enter stock concentration, dose volume, infusion duration, and dosing frequency."
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
        target_cols = st.columns(3)
        target_css_mg_l = target_cols[0].number_input("Target Css / Cavg (mg/L)", min_value=0.0, value=setup_drug_values["target_concentration_mg_l"], step=5.0)
        target_cmax_mg_l = target_cols[1].number_input("Target Cmax (mg/L)", min_value=0.0, value=250.0, step=5.0)
        reservoir_replacement_interval_h = target_cols[2].number_input("Extra replacement interval (h)", min_value=0.1, value=24.0, step=1.0)

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
        solver_result = solve_css_cmax_replacement(
            q24_system,
            setup_drug_name,
            target_css_mg_l,
            target_cmax_mg_l,
            fos_central_volume / fos_duration,
            fos_duration,
            fos_interval,
            replacement_interval_h=reservoir_replacement_interval_h,
            duration_h=duration_h,
            dt_min=dt_min,
        )
        setup_cols[3].metric("Solved central stock", f"{solver_result.central_stock_mg_ml:.6f} mg/mL")
        fos_central_stock = solver_result.central_stock_mg_ml if scenario == "q24_replacement" else 5.897897
        fos_central_rate = fos_central_volume / fos_duration

        if scenario == "q24_replacement":
            fos_extra_stock = solver_result.extra_replacement_concentration_mg_ml
            fos_extra_volume = 0.0
            fos_extra_rate = 0.0
            preload_extra_mg = 0.0
            extra_transfer_volume_ml = q_extra_to_central * reservoir_replacement_interval_h * 60
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
            extra_cols = st.columns(3)
            fos_extra_stock = extra_cols[0].number_input("Extra stock (mg/mL)", min_value=0.0, value=8.351422, step=0.1, format="%.6f")
            fos_extra_volume = extra_cols[1].number_input("Extra dose volume (mL)", min_value=0.0, value=6.0, step=0.5)
            preload_default = fos_extra_stock * fos_extra_volume
            preload_extra_mg = extra_cols[2].number_input("Extra preload amount (mg)", min_value=0.0, value=preload_default, step=1.0)
            fos_extra_rate = fos_extra_volume / fos_duration
        if scenario == "q24_replacement":
            st.caption(
                f"Calculated central pump rate: {fos_central_rate:.3f} mL/min. "
                f"Each central dose = {fos_central_stock * fos_central_volume:.3f} mg. "
                f"Prepare {extra_volume_ml:.1f} mL of extra replacement solution at {fos_extra_stock:.6f} mg/mL "
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
                intermittent_interval_h=values["dosing_frequency_h"] or 6.0,
                intermittent_duration_h=values["maintenance_duration_h"],
            ))
    result = simulate_hfim(scenario, system, fos, drugs, duration_h=duration_h, dt_min=dt_min)

    st.pyplot(_plot_setup_schematic(system_values={
        "Central": f"{central_bottle_ml:g} mL",
        "Cartridge": f"{cartridge_ml:g} mL",
        "Extra": f"{extra_volume_ml:g} mL",
        "Extra to central": f"{q_extra_to_central:g} mL/min",
        "Extra diluent": f"{q_extra_diluent:g} mL/min",
        "Central diluent": f"{q_central_diluent:g} mL/min",
        "Waste": f"{q_extra_to_central + q_central_diluent:g} mL/min",
        "Extra overflow": f"{fos.extra_infusion_ml_min:g} mL/min while dosing",
        "Reservoir interval": f"q{reservoir_replacement_interval_h:g}h",
    }, injection_values=_schematic_injection_values(drug_inputs, fos, scenario, result.summary), scenario=scenario))
    st.dataframe(_setup_overview_rows(central_bottle_ml, cartridge_ml, extra_volume_ml, q_extra_to_central, q_extra_diluent, q_central_diluent, scenario, fos), width="stretch", hide_index=True)
    st.markdown("**System solution volumes**")
    st.dataframe(_solution_volume_rows(q_central_diluent, q_extra_diluent, scenario, duration_h), width="stretch", hide_index=True)
    st.markdown("**Drug injection plan**")
    st.dataframe(_injection_plan_rows(drug_inputs, fos, scenario, setup_drug_name), width="stretch", hide_index=True)

    run_and_save = st.button("Run and save to SQLite")
    setup_summary = result.summary[setup_drug_name]
    setup_target_auc = target_css_mg_l * 24 if scenario == "q24_replacement" else setup_drug_values["target_value"] if setup_drug_values["target_type"] == "AUC0-24 exposure" else 0

    st.subheader("5. Result overview")
    cols = st.columns(5)
    cols[0].metric(f"{setup_drug_name} AUC0-24", f"{setup_summary['central_auc_0_24_mg_h_l']:.1f}", f"target {setup_target_auc:g}" if setup_target_auc else None)
    cols[1].metric("Central Cavg/Css", f"{setup_summary['central_cavg_0_24_mg_l']:.1f} mg/L", f"target {target_css_mg_l:g}" if scenario == "q24_replacement" else None)
    cols[2].metric(f"{setup_drug_name} Cmax central", f"{setup_summary['central_cmax_mg_l']:.1f} mg/L", f"target {target_cmax_mg_l:g}" if scenario == "q24_replacement" else None)
    cmin_value = setup_summary.get("central_cmin_after_24h_mg_l", setup_summary["central_cmin_mg_l"])
    cols[3].metric(f"{setup_drug_name} Cmin after 24h", f"{cmin_value:.1f} mg/L")
    if scenario == "q24_replacement":
        cols[4].metric("Extra replacement", f"{fos.extra_stock_mg_ml:.6f} mg/mL")
    else:
        cols[4].metric(f"{setup_drug_name} overflow loss", f"{setup_summary['overflow_loss_mg']:.2f} mg")

    rows = result.rows
    st.subheader("6. PK concentration")
    if setup_drug_values:
        st.pyplot(_plot_static(rows, [setup_drug_name], f"{setup_drug_name} central and extra concentration", include_extra=True))
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
    st.dataframe(review_rows, width="stretch", hide_index=True)

    st.markdown("**Calculation details**")
    if setup_prep:
        st.markdown(f"**{setup_drug_name} central dosing**")
        st.dataframe(setup_prep, width="stretch", hide_index=True)
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
        st.dataframe(extra_solution_rows, width="stretch", hide_index=True)
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
        st.dataframe(central_diluent_ci_rows, width="stretch", hide_index=True)
    if other_prep:
        st.markdown(f"**{_prep_group_title(other_prep)}**")
        st.dataframe(other_prep, width="stretch", hide_index=True)

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
                loading_dose = dosing_cols[0].checkbox("Loading dose", value=default["loading_dose"], key=f"loading_dose_{index}")
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
            detail_cols = st.columns(3)
            if loading_dose:
                loading_target = detail_cols[0].number_input(
                    "Loading target (mg/L)",
                    min_value=0.0,
                    value=target_value * default["loading_target_multiplier"],
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
            if maintenance == "intermittent infusion":
                maintenance_duration_h = detail_cols[2].number_input(
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


def _qextra_default_for_scenario(scenario: str, flow_mode: str, auto_extra_flow: float) -> float:
    if scenario == "q24_replacement":
        return 0.167
    return auto_extra_flow if _is_auto_flow_mode(flow_mode) else 0.921


def _flow_widget_key(prefix: str, scenario: str, auto_flow_mode: bool, volume_or_space: float, target_half_life_h: float) -> str:
    if not auto_flow_mode:
        return f"{prefix}_{scenario}_manual"
    return f"{prefix}_{scenario}_auto_{volume_or_space:.3f}_{target_half_life_h:.3f}"


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
        {"Part": "Central diluent", "Current value": f"{q_central_diluent:g} mL/min", "How to use": "remaining blank inflow after Qextra; together with Qextra it sets central washout"},
        {"Part": "Extra volume", "Current value": f"{extra_volume_ml:g} mL", "How to use": extra_volume_use},
        {"Part": "Extra to central", "Current value": f"{q_extra_to_central:g} mL/min", "How to use": extra_transfer_use},
        {"Part": "Waste", "Current value": f"{total_central_outflow:g} mL/min", "How to use": f"total central output; shared central half-life ≈ {_fmt_optional(central_half_life)} h"},
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
            rows.append({
                "Drug": name,
                "Target": f"{values['target_type']} = {values['target_value']:g}",
                "Half-life": f"{values['half_life_h']:g} h",
                "Dosing plan": (
                    f"loading target: {values['loading_target_concentration_mg_l']:g} mg/L over {values['loading_duration_h']:g} h; "
                    if values["loading_dose"]
                    else "loading dose: no; "
                ) + f"maintenance: {values['maintenance']}",
                "Physical setting": "calculated from loading/Css target and the shared central-to-waste flow",
            })
    return rows


def _fmt_optional(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.2f}"


def _extra_dosing_plan_label(scenario: str) -> str:
    if scenario == "overflow":
        return "extra intermittent infusion with overflow"
    return "no extra dosing"


def _schematic_injection_values(
    drug_inputs: dict[str, dict],
    fos: FosfomycinConfig,
    scenario: str,
    summary: dict,
) -> dict[str, list[str]]:
    setup_drug_name = fos.drug_name
    setup_central_lines = []
    central_other_lines = []
    central_diluent_lines = []
    extra_lines = []
    if setup_drug_name in drug_inputs:
        setup_central_lines.extend([
            f"{setup_drug_name} central",
            f"stock {fos.central_stock_mg_ml:g} mg/mL",
            f"{fos.central_infusion_ml_min * fos.infusion_duration_min:g} mL over {fos.infusion_duration_min / 60:g} h",
            f"q{fos.dosing_interval_min / 60:g}h = {fos.central_dose_mg:.2f} mg/dose",
        ])
        if scenario == "overflow":
            extra_lines.extend([
                f"{setup_drug_name} extra",
                f"stock {fos.extra_stock_mg_ml:g} mg/mL",
                f"{fos.extra_infusion_ml_min * fos.infusion_duration_min:g} mL over {fos.infusion_duration_min / 60:g} h",
                f"q{fos.dosing_interval_min / 60:g}h = {fos.extra_dose_mg:.2f} mg/dose",
            ])
        elif scenario == "q24_replacement":
            extra_lines.extend([
                f"{setup_drug_name} q24h replacement",
                f"stock {fos.extra_stock_mg_ml:g} mg/mL",
                f"{fos.extra_stock_mg_ml * summary[setup_drug_name]['final_extra_volume_ml']:.2f} mg/replacement",
                f"replace q{fos.reservoir_replacement_interval_h:g}h",
            ])

    for name in drug_inputs:
        if name == setup_drug_name:
            continue
        values = drug_inputs.get(name)
        item = summary.get(name)
        if not values or not item:
            continue
        central_other_lines.append(f"{name} central")
        if values["loading_dose"]:
            central_other_lines.append(f"LD target {item['loading_target_concentration_mg_l']:.1f} mg/L")
            central_other_lines.append(f"{item['loading_dose_mg']:.3f} mg over {item['loading_duration_h']:.2f} h")
        if values["maintenance"] == "continuous infusion":
            central_other_lines.append("CI in Diluent Central")
            concentration = item.get("central_diluent_concentration_mg_ml")
            if concentration is not None:
                central_diluent_lines.append(f"{name} {concentration * 1000:.2f} ug/mL")
                central_diluent_lines.append(f"{item['central_diluent_drug_per_24h_mg']:.2f} mg/q24h")
            else:
                central_diluent_lines.append(f"{name} needs separate CI")
        if values["maintenance"] == "intermittent infusion":
            central_other_lines.append(
                f"q{item['intermittent_interval_h']:.1f}h {item['intermittent_dose_mg']:.3f} mg/{item['intermittent_duration_h']:.2f}h"
            )
    if not setup_central_lines:
        setup_central_lines = [f"No {setup_drug_name} central dosing"]
    if not central_other_lines:
        central_other_lines = ["No other central dosing"]
    if not extra_lines:
        extra_lines = ["No extra drug dosing selected"]
    return {
        "setup_central": setup_central_lines,
        "central_other": central_other_lines,
        "central_other_drugs": [name for name in drug_inputs if name != setup_drug_name],
        "central_diluent": central_diluent_lines,
        "extra": extra_lines,
        "setup_drug": setup_drug_name,
    }


def _plot_setup_schematic(system_values: dict[str, str], injection_values: dict[str, list[str]], scenario: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.35, 7.2)
    ax.axis("off")

    def box(x, y, w, h, title, value, facecolor="#e6f3ff", title_size=12, value_size=None):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.6, edgecolor="#2b6cb0", facecolor="#e6f3ff")
        patch.set_facecolor(facecolor)
        ax.add_patch(patch)
        title_y = y + h * (0.76 if "\n" in value else 0.62)
        value_y = y + h * (0.38 if "\n" in value else 0.30)
        ax.text(x + w / 2, title_y, title, ha="center", va="center", fontsize=title_size, weight="bold")
        value_font_size = value_size if value_size is not None else (9.0 if "\n" in value else 11)
        ax.text(x + w / 2, value_y, value, ha="center", va="center", fontsize=value_font_size, color="#0a7f35", linespacing=1.08)

    def note_box(x, y, w, h, title, lines):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.5, edgecolor="#0f766e", facecolor="#ecfeff")
        ax.add_patch(patch)
        ax.text(x + 0.12, y + h - 0.22, title, ha="left", va="top", fontsize=10.2, weight="bold", color="#0f172a")
        ax.text(x + 0.12, y + h - 0.55, "\n".join(lines), ha="left", va="top", fontsize=8.1, color="#0f172a", linespacing=1.18)

    def arrow(start, end, label):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=2, color="#f97316"))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.2, label, ha="center", fontsize=10, color="#f97316")

    setup_drug = injection_values["setup_drug"]
    box(1.0, 2.05, 1.8, 1.2, "Waste", system_values["Waste"])
    box(4.0, 2.05, 2.0, 1.4, "Central", system_values["Central"])
    box(4.1, 3.85, 1.8, 0.65, "Cartridge", system_values["Cartridge"])
    box(7.1, 2.05, 1.8, 1.2, "Extra", system_values["Extra"])
    if scenario == "overflow":
        box(9.55, 2.1, 1.25, 1.05, "Extra waste", "overflow", facecolor="#fff7ed")
    if scenario != "q24_replacement":
        box(7.1, 0.35, 1.8, 0.9, "Diluent Extra", system_values["Extra diluent"])
    central_diluent_lines = injection_values.get("central_diluent", [])
    central_diluent_value = system_values["Central diluent"]
    if central_diluent_lines:
        central_diluent_value = "\n".join([system_values["Central diluent"], *central_diluent_lines[:4]])
    central_diluent_h = 1.55 if central_diluent_lines else 0.9
    central_diluent_y = 0.05 if central_diluent_lines else 0.35
    central_diluent_w = 3.15 if central_diluent_lines else 2.3
    central_diluent_x = 3.35 if central_diluent_lines else 3.85
    box(
        central_diluent_x,
        central_diluent_y,
        central_diluent_w,
        central_diluent_h,
        "Diluent Central q24h",
        central_diluent_value,
        title_size=11.5,
        value_size=9.7 if central_diluent_lines else None,
    )
    note_box(0.35, 5.35, 2.85, 1.55, f"{setup_drug} to central", injection_values["setup_central"])
    other_title = " / ".join(injection_values.get("central_other_drugs", [])) or "Other central dosing"
    note_box(3.55, 5.35, 2.9, 1.55, other_title, injection_values["central_other"])
    extra_title = _extra_schematic_title(setup_drug, scenario)
    if scenario != "q24_replacement":
        note_box(6.8, 5.35, 2.85, 1.55, extra_title, injection_values["extra"])
    else:
        note_box(6.8, 5.35, 2.85, 1.55, extra_title, injection_values["extra"])

    arrow((7.1, 2.65), (6.0, 2.65), system_values["Extra to central"])
    if scenario == "overflow":
        arrow((8.9, 2.65), (9.55, 2.65), "overflow")
    arrow((4.0, 2.65), (2.8, 2.65), system_values["Waste"])
    arrow((5.0, central_diluent_y + central_diluent_h), (5.0, 2.05), system_values["Central diluent"])
    if scenario != "q24_replacement":
        arrow((8.0, 1.25), (8.0, 2.05), system_values["Extra diluent"])
    arrow((5.0, 3.45), (5.0, 3.85), "120 mL/min")
    arrow((2.65, 5.35), (4.15, 3.42), "dose to central")
    arrow((5.95, 5.35), (5.25, 3.45), "LD to central")
    if scenario != "q24_replacement":
        extra_arrow_label = _extra_schematic_arrow_label(scenario)
        arrow((8.1, 5.35), (8.1, 3.25), extra_arrow_label)
    else:
        arrow((8.1, 5.35), (8.1, 3.25), "q24h full replacement")
    fig.tight_layout()
    return fig


def _extra_schematic_title(setup_drug: str, scenario: str) -> str:
    if scenario == "overflow":
        return f"{setup_drug} to extra"
    return f"{setup_drug} extra q24h"


def _extra_schematic_arrow_label(scenario: str) -> str:
    if scenario == "overflow":
        return "dose to extra"
    return "extra"


def _plot_static(rows: list[dict], drugs: list[str], title: str, include_extra: bool):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
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
    fos: FosfomycinConfig,
    duration_h: float,
) -> list[dict]:
    rows = []
    central_recipe = _central_diluent_reservoir_summary(summary, duration_h)
    extra_summary = _replacement_solution_summary(system, fos, duration_h)
    interval_h = fos.reservoir_replacement_interval_h

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
        .prep-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 16px 16px 14px 16px;
            min-height: 238px;
            background: rgba(15, 23, 42, 0.32);
        }
        .prep-card-blue { border-top: 4px solid #60a5fa; }
        .prep-card-teal { border-top: 4px solid #2dd4bf; }
        .prep-card-amber { border-top: 4px solid #f59e0b; }
        .prep-card-title {
            font-size: 1.03rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .prep-card-drugs {
            color: #cbd5e1;
            font-size: 0.9rem;
            min-height: 42px;
            margin-bottom: 12px;
        }
        .prep-card-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .prep-card-value {
            color: #f8fafc;
            font-size: 1.75rem;
            line-height: 1.15;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .prep-card-caption {
            color: #cbd5e1;
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
    if scenario not in {"q24_replacement"}:
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
        + f"\n- Central Cavg = **{setup_summary['central_cavg_0_24_mg_l']:.1f} mg/L**; Cmax = **{setup_summary['central_cmax_mg_l']:.1f} mg/L**; Cmin after 24 h = **{setup_summary.get('central_cmin_after_24h_mg_l', setup_summary['central_cmin_mg_l']):.1f} mg/L**."
    ]
    for name in drug_names:
        item = summary[name]
        lines.append(
            f"- {name} target central concentration = **{item['target_concentration_mg_l']:.2f} mg/L**; "
            f"dosing mode = **{item['dosing_mode']}**; loading dose = **{item['loading_dose_mg']:.3f} mg**; "
            f"continuous infusion = **{item['infusion_rate_mg_h']:.3f} mg/h**."
        )
    return "\n".join(lines)


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


if __name__ == "__main__":
    main()
