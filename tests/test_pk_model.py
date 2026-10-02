import math
import unittest

from hfim_simulator.pk import (
    DrugConfig,
    FosfomycinConfig,
    SystemConfig,
    compute_continuous_infusion,
    flow_for_half_life,
    half_life_for_flow,
    intermittent_peak_trough,
    one_compartment_steady_state_profile,
    simulate_hfim,
    solve_blaser_setup,
    solve_css_cmax_replacement,
    solve_two_phase_replacement,
)


class HfimPkModelTest(unittest.TestCase):
    def test_flow_and_half_life_conversions(self):
        self.assertAlmostEqual(flow_for_half_life(170, 3), 0.65464, places=5)
        self.assertAlmostEqual(flow_for_half_life(241, 3), 0.928, places=3)
        self.assertAlmostEqual(half_life_for_flow(241, 0.921), 3.02, places=2)

    def test_imipenem_loading_and_infusion_for_9_mg_l(self):
        system = SystemConfig()
        regimen = compute_continuous_infusion(
            target_concentration_mg_l=9,
            half_life_h=1.25,
            central_volume_ml=system.central_volume_ml,
        )

        self.assertAlmostEqual(regimen.loading_dose_mg, 1.53, places=2)
        self.assertAlmostEqual(regimen.infusion_rate_mg_h, 0.848, places=3)
        self.assertAlmostEqual(regimen.daily_amount_mg, 20.36, places=2)

    def test_central_only_drugs_use_shared_physical_waste_flow_plus_setup_pump_average(self):
        shared_flow = flow_for_half_life(170, 1.25)
        setup_pump_average_flow = 0.1 * 60 / 360
        result = simulate_hfim(
            scenario="q24_replacement",
            system=SystemConfig(
                q_extra_to_central_ml_min=0.167,
                q_central_diluent_ml_min=shared_flow - 0.167,
                q_extra_diluent_ml_min=0,
            ),
            fos=FosfomycinConfig(central_stock_mg_ml=0, extra_stock_mg_ml=0),
            drugs=[
                DrugConfig(
                    "testdrug",
                    target_concentration_mg_l=9,
                    half_life_h=3.0,
                    dosing_mode="continuous infusion only",
                )
            ],
            duration_h=24,
            dt_min=1,
        )

        expected_rate_mg_h = (9 / 1000) * (shared_flow + setup_pump_average_flow) * 60
        self.assertAlmostEqual(result.summary["testdrug"]["infusion_rate_mg_h"], expected_rate_mg_h)
        self.assertAlmostEqual(result.summary["testdrug"]["daily_amount_mg"], expected_rate_mg_h * 24)

    def test_continuous_infusion_drug_is_formulated_in_central_diluent(self):
        shared_flow = flow_for_half_life(170, 1.25)
        q_central_diluent = shared_flow - 0.167
        result = simulate_hfim(
            scenario="q24_replacement",
            system=SystemConfig(
                q_extra_to_central_ml_min=0.167,
                q_central_diluent_ml_min=q_central_diluent,
                q_extra_diluent_ml_min=0,
            ),
            fos=FosfomycinConfig(central_stock_mg_ml=0, extra_stock_mg_ml=0),
            drugs=[
                DrugConfig(
                    "imipenem",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="loading dose + continuous infusion",
                    loading_target_concentration_mg_l=18,
                    loading_duration_h=0.5,
                )
            ],
            duration_h=24,
            dt_min=1,
        )

        imipenem = result.summary["imipenem"]
        expected_concentration = (imipenem["infusion_rate_mg_h"] / 60) / q_central_diluent
        self.assertAlmostEqual(imipenem["central_diluent_concentration_mg_ml"], expected_concentration)
        self.assertAlmostEqual(imipenem["central_diluent_volume_per_24h_ml"], q_central_diluent * 1440)
        self.assertAlmostEqual(imipenem["central_diluent_drug_per_24h_mg"], imipenem["daily_amount_mg"])

    def test_relebactam_is_two_thirds_of_imipenem(self):
        system = SystemConfig()
        imipenem = compute_continuous_infusion(9, 1.25, system.central_volume_ml)
        relebactam = compute_continuous_infusion(6, 1.25, system.central_volume_ml)

        self.assertAlmostEqual(relebactam.loading_dose_mg, imipenem.loading_dose_mg * 2 / 3, places=6)
        self.assertAlmostEqual(relebactam.infusion_rate_mg_h, imipenem.infusion_rate_mg_h * 2 / 3, places=6)

    def test_auc_target_is_converted_to_average_concentration(self):
        system = SystemConfig()
        regimen = compute_continuous_infusion(
            target_concentration_mg_l=3600 / 24,
            half_life_h=3,
            central_volume_ml=system.central_volume_ml,
        )

        self.assertAlmostEqual(regimen.target_concentration_mg_l, 150)

    def test_continuous_only_starts_without_loading_dose(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[
                DrugConfig(
                    "imipenem",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="continuous infusion only",
                ),
            ],
            duration_h=1,
            dt_min=1,
        )

        first_imipenem = next(row for row in result.rows if row["drug"] == "imipenem")
        self.assertEqual(first_imipenem["central_mg_l"], 0)

    def test_loading_only_declines_after_initial_target(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[
                DrugConfig(
                    "imipenem",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="loading dose only",
                ),
            ],
            duration_h=2,
            dt_min=1,
        )

        imipenem_rows = [row for row in result.rows if row["drug"] == "imipenem"]
        self.assertAlmostEqual(imipenem_rows[0]["central_mg_l"], 9)
        self.assertLess(imipenem_rows[-1]["central_mg_l"], 9)

    def test_loading_infusion_starts_at_zero_and_uses_loading_target(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[
                DrugConfig(
                    "imipenem",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="loading dose only",
                    loading_target_concentration_mg_l=18,
                    loading_duration_h=0.5,
                    loading_volume_ml=5,
                ),
            ],
            duration_h=1,
            dt_min=1,
        )

        imipenem_rows = [row for row in result.rows if row["drug"] == "imipenem"]
        self.assertEqual(imipenem_rows[0]["central_mg_l"], 0)
        self.assertGreater(max(row["central_mg_l"] for row in imipenem_rows), 9)
        self.assertAlmostEqual(result.summary["imipenem"]["loading_concentration_mg_ml"], 0.612)
        self.assertAlmostEqual(result.summary["imipenem"]["loading_infusion_rate_ml_h"], 10)

    def test_no_dose_mode_does_not_add_loading_or_infusion(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[
                DrugConfig(
                    "imipenem",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="no dose",
                ),
            ],
            duration_h=2,
            dt_min=1,
        )

        imipenem_rows = [row for row in result.rows if row["drug"] == "imipenem"]
        self.assertTrue(all(row["central_mg_l"] == 0 for row in imipenem_rows))

    def test_overflow_keeps_volumes_fixed_and_tracks_loss(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[
                DrugConfig("imipenem", target_concentration_mg_l=9, half_life_h=1.25),
                DrugConfig("relebactam", target_concentration_mg_l=6, half_life_h=1.25),
            ],
            duration_h=24,
            dt_min=1,
        )

        self.assertAlmostEqual(result.summary["fosfomycin"]["final_central_volume_ml"], 170, places=6)
        self.assertAlmostEqual(result.summary["fosfomycin"]["final_extra_volume_ml"], 241, places=6)
        self.assertGreater(result.summary["fosfomycin"]["overflow_loss_mg"], 0)
        self.assertIn("central_auc_0_24_mg_h_l", result.summary["fosfomycin"])

    def test_q24_replacement_has_extra_drug_at_time_zero(self):
        result = simulate_hfim(
            scenario="q24_replacement",
            system=SystemConfig(
                q_extra_to_central_ml_min=0.928047,
                q_extra_diluent_ml_min=0,
                q_central_diluent_ml_min=0.654639,
            ),
            fos=FosfomycinConfig(
                central_stock_mg_ml=1,
                extra_stock_mg_ml=2,
                extra_infusion_ml_min=0,
                preload_extra_mg=0,
            ),
            drugs=[],
            duration_h=1,
            dt_min=1,
        )

        first = next(row for row in result.rows if row["drug"] == "fosfomycin")
        self.assertAlmostEqual(first["extra_mg_l"], 2000)

    def test_q24_replacement_keeps_extra_concentration_constant_between_replacements(self):
        result = simulate_hfim(
            scenario="q24_replacement",
            system=SystemConfig(
                q_extra_to_central_ml_min=0.928047,
                q_extra_diluent_ml_min=0,
                q_central_diluent_ml_min=0.654639,
            ),
            fos=FosfomycinConfig(
                central_stock_mg_ml=1,
                extra_stock_mg_ml=2,
                extra_infusion_ml_min=0,
                preload_extra_mg=0,
                reservoir_replacement_interval_h=24,
            ),
            drugs=[],
            duration_h=49,
            dt_min=60,
        )

        extra_by_time = {
            row["time_h"]: row["extra_mg_l"]
            for row in result.rows
            if row["drug"] == "fosfomycin"
        }
        self.assertAlmostEqual(extra_by_time[0], 2000)
        self.assertAlmostEqual(extra_by_time[23], 2000)
        self.assertAlmostEqual(extra_by_time[24], 2000)
        self.assertAlmostEqual(extra_by_time[47], 2000)
        self.assertAlmostEqual(extra_by_time[48], 2000)

    def test_css_cmax_solver_reaches_default_cavg_and_cmax_targets(self):
        system = SystemConfig(
            q_extra_to_central_ml_min=0.928047,
            q_extra_diluent_ml_min=0,
            q_central_diluent_ml_min=0.654639,
        )
        solver = solve_css_cmax_replacement(
            system=system,
            drug_name="fosfomycin",
            target_css_mg_l=150,
            target_cmax_mg_l=250,
            central_infusion_ml_min=0.1,
            infusion_duration_min=60,
            dosing_interval_min=360,
            replacement_interval_h=24,
            duration_h=168,
            dt_min=1,
        )

        self.assertAlmostEqual(solver.predicted_auc_0_24_mg_h_l, 3600, places=6)
        self.assertAlmostEqual(solver.predicted_cavg_mg_l, 150, places=6)
        self.assertGreater(solver.extra_replacement_concentration_mg_ml, 0)
        self.assertLess(abs(solver.predicted_cmax_mg_l - 250), 5)
        self.assertGreater(solver.predicted_cmin_mg_l, 0)
        self.assertTrue(solver.feasible)

    def test_css_cmax_solver_falls_back_to_central_only_when_extra_has_no_contribution(self):
        system = SystemConfig(
            q_extra_to_central_ml_min=0,
            q_extra_diluent_ml_min=0,
            q_central_diluent_ml_min=0.654639,
        )
        solver = solve_css_cmax_replacement(
            system=system,
            drug_name="fosfomycin",
            target_css_mg_l=150,
            target_cmax_mg_l=250,
            central_infusion_ml_min=0.1,
            infusion_duration_min=60,
            dosing_interval_min=360,
            replacement_interval_h=24,
            duration_h=168,
            dt_min=1,
        )

        self.assertAlmostEqual(solver.predicted_auc_0_24_mg_h_l, 3600, places=6)
        self.assertAlmostEqual(solver.predicted_cavg_mg_l, 150, places=6)
        self.assertEqual(solver.extra_replacement_concentration_mg_ml, 0)
        self.assertGreater(solver.central_stock_mg_ml, 0)

    def test_q24_replacement_preparation_table_includes_extra_replacement(self):
        result = simulate_hfim(
            scenario="q24_replacement",
            system=SystemConfig(
                extra_volume_ml=241,
                q_extra_to_central_ml_min=0.928047,
                q_extra_diluent_ml_min=0,
            ),
            fos=FosfomycinConfig(
                central_stock_mg_ml=1,
                extra_stock_mg_ml=2,
                extra_infusion_ml_min=0,
                preload_extra_mg=0,
                reservoir_replacement_interval_h=24,
            ),
            drugs=[],
            duration_h=168,
            dt_min=60,
        )

        prep = result.summary["drug_preparation"]
        extra_rows = [row for row in prep if row["component"] == "extra q24h fixed-concentration solution"]
        self.assertEqual(len(extra_rows), 1)
        self.assertAlmostEqual(extra_rows[0]["amount_mg"], 482, places=4)
        self.assertIn("+10% volume", extra_rows[0]["note"])
        self.assertIn("pressure/filter-controlled q24h replacement", extra_rows[0]["note"])
        self.assertIn("Qextra transfer is modeled as drug leaving this same fill", extra_rows[0]["note"])
        self.assertNotIn("transfer demand", extra_rows[0]["note"])
        self.assertNotIn("minimum same-concentration solution", extra_rows[0]["note"])

    def test_central_diluent_drug_washout_includes_setup_drug_pump_flow(self):
        system = SystemConfig(
            central_bottle_ml=100,
            cartridge_ml=0,
            extra_volume_ml=50,
            q_extra_to_central_ml_min=0.0,
            q_extra_diluent_ml_min=0.0,
            q_central_diluent_ml_min=0.2,
        )
        fos = FosfomycinConfig(
            drug_name="fosfomycin",
            central_stock_mg_ml=0.0,
            extra_stock_mg_ml=0.0,
            central_infusion_ml_min=0.3,
            extra_infusion_ml_min=0.0,
            infusion_duration_min=60,
            dosing_interval_min=360,
        )
        imipenem = DrugConfig(
            "imipenem",
            target_concentration_mg_l=0.0,
            half_life_h=1.25,
            dosing_mode="loading dose only",
            loading_target_concentration_mg_l=10.0,
            loading_duration_h=0.0,
            loading_volume_ml=5.0,
        )

        result = simulate_hfim("overflow", system, fos, [imipenem], duration_h=1, dt_min=1)

        imipenem_rows = [row for row in result.rows if row["drug"] == "imipenem"]
        vc = system.central_volume_ml
        initial_amount_mg = 10.0 / 1000 * vc
        # Total central outflow during the fosfomycin dosing window = q_central_diluent (0.2) +
        # q_fos_central (0.3) = 0.5 mL/min, not just q_central_diluent.
        expected_amount_mg = initial_amount_mg * (1 - 0.5 / vc * 1)
        expected_conc_mg_l = expected_amount_mg / vc * 1000

        self.assertAlmostEqual(imipenem_rows[0]["central_mg_l"], 10.0)
        self.assertAlmostEqual(imipenem_rows[1]["central_mg_l"], expected_conc_mg_l, places=6)

    def test_continuous_infusion_recipe_uses_time_weighted_setup_drug_pump_flow(self):
        system = SystemConfig(
            central_bottle_ml=100,
            cartridge_ml=0,
            extra_volume_ml=50,
            q_extra_to_central_ml_min=0.1,
            q_extra_diluent_ml_min=0.0,
            q_central_diluent_ml_min=0.2,
        )
        fos = FosfomycinConfig(
            drug_name="fosfomycin",
            central_stock_mg_ml=0.0,
            extra_stock_mg_ml=0.0,
            central_infusion_ml_min=0.6,
            extra_infusion_ml_min=0.0,
            infusion_duration_min=120,
            dosing_interval_min=360,
        )
        imipenem = DrugConfig(
            "imipenem",
            target_concentration_mg_l=10.0,
            half_life_h=1.25,
            dosing_mode="continuous infusion only",
        )

        result = simulate_hfim("q24_replacement", system, fos, [imipenem], duration_h=24, dt_min=1)

        average_setup_pump_ml_min = 0.6 * 120 / 360
        effective_outflow_ml_min = system.q_waste_ml_min + average_setup_pump_ml_min
        expected_rate_mg_h = (10.0 / 1000) * effective_outflow_ml_min * 60
        imipenem_summary = result.summary["imipenem"]

        self.assertAlmostEqual(imipenem_summary["elimination_flow_ml_min"], effective_outflow_ml_min)
        self.assertAlmostEqual(imipenem_summary["infusion_rate_mg_h"], expected_rate_mg_h)
        self.assertAlmostEqual(
            imipenem_summary["central_diluent_concentration_mg_ml"],
            (expected_rate_mg_h / 60) / system.q_central_diluent_ml_min,
        )

    def test_continuous_infusion_true_simulated_cavg_converges_to_target(self):
        # Independent end-to-end check: run the full ODE (not the analytical recipe formula used to
        # size the infusion rate) and confirm the CI drug's actual simulated concentration converges
        # to its target Css, given that the setup drug's own pump flow perturbs the shared central
        # outflow only during its dosing window. This is the check that would fail if the
        # time-weighted effective outflow approximation were not actually representative of the real
        # simulated dynamics - unlike a test that re-derives the same formula as the implementation.
        system = SystemConfig(
            central_bottle_ml=100, cartridge_ml=70, extra_volume_ml=241,
            q_extra_to_central_ml_min=0.921, q_extra_diluent_ml_min=0.921, q_central_diluent_ml_min=0.65,
        )
        fos = FosfomycinConfig(
            central_stock_mg_ml=5.9, extra_stock_mg_ml=8.35,
            central_infusion_ml_min=0.1, extra_infusion_ml_min=0.1,
            infusion_duration_min=60, dosing_interval_min=360,
        )
        imipenem = DrugConfig(
            "imipenem",
            target_concentration_mg_l=9.0,
            half_life_h=1.25,
            dosing_mode="loading dose + continuous infusion",
            loading_target_concentration_mg_l=18.0,
            loading_duration_h=0.5,
        )

        result = simulate_hfim("overflow", system, fos, [imipenem], duration_h=96, dt_min=1)
        rows = [row for row in result.rows if row["drug"] == "imipenem"]
        late_rows = [row for row in rows if row["time_h"] >= 72]
        simulated_cavg = sum(row["central_mg_l"] for row in late_rows) / len(late_rows)

        self.assertAlmostEqual(simulated_cavg, 9.0, delta=0.05)

    def test_intermittent_maintenance_dose_matches_continuous_infusion_average_rate(self):
        system = SystemConfig()
        regimen = compute_continuous_infusion(
            target_concentration_mg_l=9,
            half_life_h=1.25,
            central_volume_ml=system.central_volume_ml,
            intermittent_interval_h=6,
        )

        self.assertAlmostEqual(
            regimen.intermittent_dose_mg,
            regimen.infusion_rate_mg_h * regimen.intermittent_interval_h,
            places=9,
        )

    def test_intermittent_maintenance_converges_to_target_instead_of_drifting_with_interval(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(central_stock_mg_ml=0, extra_stock_mg_ml=0),
            drugs=[
                DrugConfig(
                    "testdrug",
                    target_concentration_mg_l=9,
                    half_life_h=1.25,
                    dosing_mode="intermittent infusion only",
                    intermittent_interval_h=6,
                    intermittent_duration_h=1,
                )
            ],
            duration_h=96,
            dt_min=1,
        )

        rows = [row for row in result.rows if row["drug"] == "testdrug"]
        late_rows = [row for row in rows if row["time_h"] >= 72]
        late_avg_mg_l = sum(row["central_mg_l"] for row in late_rows) / len(late_rows)

        # With the default SystemConfig flow/interval, the pre-fix formula (dose = target x volume,
        # ignoring washout between doses) converges to roughly target / (k x interval) =~ 2.7 mg/L here,
        # not the intended 9 mg/L target.
        self.assertAlmostEqual(late_avg_mg_l, 9.0, delta=1.0)

    def test_overflow_scenario_uses_solved_central_stock_not_a_fixed_constant(self):
        system = SystemConfig(extra_volume_ml=241, q_extra_to_central_ml_min=0, q_central_diluent_ml_min=0.65)
        target_css_mg_l = 9.0

        solver = solve_css_cmax_replacement(
            system=system,
            drug_name="imipenem",
            target_css_mg_l=target_css_mg_l,
            target_cmax_mg_l=20.0,
            central_infusion_ml_min=6 / 60,
            infusion_duration_min=60,
            dosing_interval_min=360,
            duration_h=48,
            dt_min=1,
        )
        fos = FosfomycinConfig(
            drug_name="imipenem",
            central_stock_mg_ml=solver.central_stock_mg_ml,
            extra_stock_mg_ml=0.0,
            central_infusion_ml_min=6 / 60,
            extra_infusion_ml_min=0.0,
            infusion_duration_min=60,
            dosing_interval_min=360,
        )

        result = simulate_hfim("overflow", system, fos, [], duration_h=48, dt_min=1)
        cavg = result.summary["imipenem"]["central_cavg_0_24_mg_l"]

        # A fosfomycin-specific hardcoded stock (5.897897 mg/mL, tuned for a ~150 mg/L target) would be
        # wildly wrong for a 9 mg/L target; the solved stock should land close to the actual target instead.
        self.assertNotAlmostEqual(solver.central_stock_mg_ml, 5.897897, places=1)
        self.assertAlmostEqual(cavg, target_css_mg_l, delta=target_css_mg_l * 0.25)

    def test_overflow_css_cmax_solver_uses_overflow_extra_infusion_basis(self):
        system = SystemConfig(
            central_bottle_ml=100,
            cartridge_ml=70,
            extra_volume_ml=241,
            q_extra_to_central_ml_min=0.921,
            q_extra_diluent_ml_min=0.921,
            q_central_diluent_ml_min=0.65,
        )

        solver = solve_css_cmax_replacement(
            system=system,
            drug_name="fosfomycin",
            target_css_mg_l=150,
            target_cmax_mg_l=250,
            central_infusion_ml_min=0.1,
            infusion_duration_min=60,
            dosing_interval_min=360,
            scenario="overflow",
            extra_infusion_ml_min=0.1,
            duration_h=48,
            dt_min=1,
        )
        fos = FosfomycinConfig(
            drug_name="fosfomycin",
            central_stock_mg_ml=solver.central_stock_mg_ml,
            extra_stock_mg_ml=solver.extra_replacement_concentration_mg_ml,
            central_infusion_ml_min=0.1,
            extra_infusion_ml_min=0.1,
            infusion_duration_min=60,
            dosing_interval_min=360,
        )

        result = simulate_hfim("overflow", system, fos, [], duration_h=48, dt_min=1)
        summary = result.summary["fosfomycin"]

        self.assertGreater(solver.extra_replacement_concentration_mg_ml, 0)
        self.assertAlmostEqual(summary["central_auc_0_24_mg_h_l"], solver.predicted_auc_0_24_mg_h_l, places=6)
        self.assertAlmostEqual(summary["central_cmax_mg_l"], solver.predicted_cmax_mg_l, places=6)
        self.assertAlmostEqual(summary["central_cavg_0_24_mg_l"], 150, places=6)

    def test_cmin_overall_is_the_whole_series_minimum_not_a_steady_state_trough(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(),
            drugs=[],
            duration_h=48,
            dt_min=1,
        )

        summary = result.summary["fosfomycin"]

        self.assertEqual(summary["central_cmin_overall_mg_l"], 0)
        self.assertGreater(summary["central_cmin_after_24h_mg_l"], summary["central_cmin_overall_mg_l"])

    def test_central_only_emits_no_phantom_setup_drug_rows_or_prep_lines(self):
        system = SystemConfig(
            central_bottle_ml=100, cartridge_ml=70, extra_volume_ml=1.0,
            q_extra_to_central_ml_min=0.0, q_extra_diluent_ml_min=0.0,
            q_central_diluent_ml_min=flow_for_half_life(170, 1.25),
        )

        result = simulate_hfim(
            "central_only", system, None,
            [DrugConfig("imipenem", 9.0, 1.25, dosing_mode="continuous infusion only")],
            duration_h=24, dt_min=1,
        )

        # Without a setup drug there must be no placeholder compartment: no extra summary entry,
        # no extra rows, and above all no 0.000 mg prep lines that would read as bench instructions.
        self.assertEqual(sorted(result.summary), ["drug_preparation", "imipenem"])
        self.assertEqual({row["drug"] for row in result.rows}, {"imipenem"})
        self.assertEqual(
            [row["component"] for row in result.summary["drug_preparation"]],
            ["continuous infusion"],
        )
        self.assertTrue(all(row["extra_mg_l"] == 0.0 for row in result.rows))

    def test_central_only_rejects_a_setup_drug_and_other_scenarios_require_one(self):
        system = SystemConfig()

        with self.assertRaises(ValueError):
            simulate_hfim("q24_replacement", system, None, [], duration_h=24, dt_min=1)
        with self.assertRaises(ValueError):
            simulate_hfim("not_a_scenario", system, None, [], duration_h=24, dt_min=1)
        # A setup drug passed alongside central_only is ignored rather than silently dosed.
        result = simulate_hfim("central_only", system, FosfomycinConfig(central_stock_mg_ml=99.0), [], duration_h=24, dt_min=1)
        self.assertEqual(sorted(result.summary), ["drug_preparation"])
        self.assertEqual(result.summary["drug_preparation"], [])

    def test_central_only_hits_each_target_for_a_shared_half_life_combination(self):
        t_half = 1.25
        system = SystemConfig(
            central_bottle_ml=100, cartridge_ml=70, extra_volume_ml=1.0,
            q_extra_to_central_ml_min=0.0, q_extra_diluent_ml_min=0.0,
            q_central_diluent_ml_min=flow_for_half_life(170, t_half),
        )
        drugs = [
            DrugConfig("imipenem", 9.0, t_half, dosing_mode="loading dose + continuous infusion",
                       loading_target_concentration_mg_l=18.0, loading_duration_h=0.5),
            DrugConfig("relebactam", 6.0, t_half, dosing_mode="loading dose + continuous infusion",
                       loading_target_concentration_mg_l=12.0, loading_duration_h=0.5),
            DrugConfig("meropenem", 16.0, t_half, dosing_mode="intermittent infusion only",
                       intermittent_interval_h=8.0, intermittent_duration_h=0.5),
        ]

        result = simulate_hfim("central_only", system, None, drugs, duration_h=96, dt_min=1)

        for drug in drugs:
            window = [r for r in result.rows if r["drug"] == drug.name and r["time_h"] >= 72]
            cavg = sum(r["central_mg_l"] for r in window) / len(window)
            self.assertAlmostEqual(cavg, drug.target_concentration_mg_l, delta=0.05)

    def test_intermittent_peak_trough_matches_the_simulated_profile(self):
        t_half, target, interval_h, duration_h = 1.25, 16.0, 8.0, 0.5
        system = SystemConfig(
            central_bottle_ml=100, cartridge_ml=70, extra_volume_ml=1.0,
            q_extra_to_central_ml_min=0.0, q_extra_diluent_ml_min=0.0,
            q_central_diluent_ml_min=flow_for_half_life(170, t_half),
        )
        result = simulate_hfim(
            "central_only", system, None,
            [DrugConfig("meropenem", target, t_half, dosing_mode="intermittent infusion only",
                        intermittent_interval_h=interval_h, intermittent_duration_h=duration_h)],
            duration_h=96, dt_min=1,
        )
        window = [r for r in result.rows if r["drug"] == "meropenem" and r["time_h"] >= 72]

        cmax, cmin = intermittent_peak_trough(target, t_half, interval_h, duration_h)

        # The closed form drives the peak-shaping table, so it has to agree with the integrated ODE
        # rather than just being internally consistent with itself.
        self.assertAlmostEqual(cmax, max(r["central_mg_l"] for r in window), delta=0.5)
        self.assertAlmostEqual(cmin, min(r["central_mg_l"] for r in window), delta=0.05)

    def test_intermittent_peak_trough_flattens_as_the_interval_shortens(self):
        peaks = [intermittent_peak_trough(16.0, 1.25, interval, 0.5)[0] for interval in (2, 3, 4, 6, 8, 12)]

        self.assertEqual(peaks, sorted(peaks))
        self.assertGreater(peaks[-1] / peaks[0], 3.0)
        self.assertEqual(intermittent_peak_trough(0.0, 1.25, 8.0, 0.5), (0.0, 0.0))

    def test_intermediate_extra_drug_name_is_not_hard_coded_to_fosfomycin(self):
        result = simulate_hfim(
            scenario="overflow",
            system=SystemConfig(),
            fos=FosfomycinConfig(drug_name="testdrug"),
            drugs=[],
            duration_h=1,
            dt_min=1,
        )

        self.assertIn("testdrug", result.summary)
        self.assertNotIn("fosfomycin", {row["drug"] for row in result.rows})
        self.assertTrue(all(row["drug"] == "testdrug" for row in result.rows))


def _two_phase_default_system() -> SystemConfig:
    # Website defaults: 170 mL central, washout set by the 1.25 h drugs, Qextra 0.167 mL/min.
    q_waste = flow_for_half_life(170, 1.25)
    return SystemConfig(
        central_bottle_ml=100,
        cartridge_ml=70,
        extra_volume_ml=241,
        q_extra_to_central_ml_min=0.167,
        q_extra_diluent_ml_min=0.0,
        q_central_diluent_ml_min=q_waste - 0.167,
    )


def _two_phase_fos(solution, interval_min: int = 360, duration_min: int = 60) -> FosfomycinConfig:
    return FosfomycinConfig(
        central_stock_mg_ml=solution.central_stock_mg_ml,
        extra_stock_mg_ml=solution.extra_replacement_concentration_mg_ml,
        central_infusion_ml_min=0.1,
        extra_infusion_ml_min=0.0,
        infusion_duration_min=duration_min,
        dosing_interval_min=interval_min,
        preload_extra_mg=0.0,
        slow_stock_mg_ml=solution.slow_stock_mg_ml,
        slow_infusion_ml_min=solution.slow_infusion_ml_min,
        slow_duration_min=solution.slow_duration_min,
    )


class TwoPhaseReplacementTest(unittest.TestCase):
    def test_reference_profile_matches_closed_form_peak_trough_and_average(self):
        profile = one_compartment_steady_state_profile(150, 3, 360, 60, dt_min=1)
        cmax, cmin = intermittent_peak_trough(150, 3, 6, 1)

        self.assertEqual(len(profile), 360)
        self.assertAlmostEqual(profile[60], cmax, places=6)
        self.assertAlmostEqual(profile[0], cmin, places=6)
        self.assertAlmostEqual(cmax, 247.56, places=2)
        self.assertAlmostEqual(cmin, 77.98, places=2)
        self.assertAlmostEqual(sum(profile) / len(profile), 150, delta=0.2)

    def test_default_setup_follows_the_three_hour_curve_within_limit(self):
        solution = solve_two_phase_replacement(_two_phase_default_system(), "fosfomycin", 150, 3, 0.1, 60, 360)

        self.assertTrue(solution.feasible)
        self.assertLess(solution.max_deviation_pct, 6.5)
        self.assertGreater(solution.single_phase_max_deviation_pct, 15)
        self.assertAlmostEqual(solution.predicted_cavg_mg_l, 150, places=6)
        self.assertAlmostEqual(solution.predicted_cmin_mg_l, solution.reference_cmin_mg_l, delta=3)
        self.assertAlmostEqual(solution.predicted_cmax_mg_l, solution.reference_cmax_mg_l, delta=3)
        self.assertEqual(solution.slow_duration_min, 150)
        # Same concentration, separate syringe: the second pump just runs slower.
        self.assertAlmostEqual(solution.slow_stock_mg_ml, solution.central_stock_mg_ml, places=9)
        self.assertGreater(solution.slow_infusion_ml_min, 0)
        self.assertLess(solution.slow_infusion_ml_min, 0.1)

    def test_full_simulation_settles_onto_the_solved_profile_with_exact_daily_auc(self):
        system = _two_phase_default_system()
        solution = solve_two_phase_replacement(system, "fosfomycin", 150, 3, 0.1, 60, 360)
        result = simulate_hfim("q24_replacement", system, _two_phase_fos(solution), [], duration_h=168, dt_min=1)
        summary = result.summary["fosfomycin"]
        last_interval = [row["central_mg_l"] for row in result.rows if 162 <= row["time_h"] < 168]

        self.assertAlmostEqual(summary["central_auc_last_24h_mg_h_l"], 3600, delta=1)
        self.assertAlmostEqual(summary["central_cmin_last_24h_mg_l"], solution.predicted_cmin_mg_l, places=2)
        self.assertAlmostEqual(summary["central_cmax_last_24h_mg_l"], solution.predicted_cmax_mg_l, places=2)
        for simulated, solved in zip(last_interval, solution.predicted_profile_mg_l):
            self.assertAlmostEqual(simulated, solved, places=2)
        # The first day is still filling up, so it sits below the settled daily exposure.
        self.assertLess(summary["central_auc_0_24_mg_h_l"], summary["central_auc_last_24h_mg_h_l"])

    def test_interval_much_longer_than_half_life_is_flagged_but_keeps_exact_auc(self):
        solution = solve_two_phase_replacement(_two_phase_default_system(), "fosfomycin", 150, 3, 0.1, 60, 720)

        self.assertFalse(solution.feasible)
        self.assertGreater(solution.max_deviation_pct, 10)
        self.assertIn("Blaser", solution.message)
        self.assertAlmostEqual(solution.predicted_cavg_mg_l, 150, places=6)

    def test_short_interval_relative_to_half_life_is_tracked_closely(self):
        system = SystemConfig(100, 70, 241, 0.167, 0.0, flow_for_half_life(170, 2) - 0.167)
        solution = solve_two_phase_replacement(system, "drug", 150, 8, 0.1, 180, 720)

        self.assertTrue(solution.feasible)
        self.assertLess(solution.max_deviation_pct, 8)

    def test_separate_weaker_stock_at_a_fixed_slow_pump_rate(self):
        solution = solve_two_phase_replacement(
            _two_phase_default_system(), "fosfomycin", 150, 3, 0.1, 60, 360, slow_infusion_ml_min=0.1
        )

        self.assertAlmostEqual(solution.slow_infusion_ml_min, 0.1, places=9)
        self.assertLess(solution.slow_stock_mg_ml, solution.central_stock_mg_ml)
        self.assertLess(solution.max_deviation_pct, 6.5)
        self.assertAlmostEqual(solution.predicted_cavg_mg_l, 150, places=6)

    def test_slow_line_appears_in_preparation_table_with_delivered_amount(self):
        system = _two_phase_default_system()
        solution = solve_two_phase_replacement(system, "fosfomycin", 150, 3, 0.1, 60, 360)
        fos = _two_phase_fos(solution)
        result = simulate_hfim("q24_replacement", system, fos, [], duration_h=24, dt_min=1)
        slow_rows = [row for row in result.summary["drug_preparation"] if "slow line" in row["component"]]

        self.assertEqual(len(slow_rows), 1)
        expected_mg = solution.slow_stock_mg_ml * solution.slow_infusion_ml_min * solution.slow_duration_min
        self.assertAlmostEqual(slow_rows[0]["amount_mg"], expected_mg, places=9)
        self.assertAlmostEqual(slow_rows[0]["daily_amount_mg"], expected_mg * 4, places=9)
        self.assertEqual(slow_rows[0]["component"], "central q6h slow line (h 1-3.5)")

    def test_setup_without_slow_line_is_unchanged(self):
        fos = FosfomycinConfig()

        self.assertFalse(fos.has_slow_line)
        self.assertEqual(fos.slow_dose_mg, 0.0)
        result = simulate_hfim("q24_replacement", SystemConfig(), fos, [], duration_h=24, dt_min=1)
        self.assertFalse(any("slow line" in row["component"] for row in result.summary["drug_preparation"]))

    def test_invalid_inputs_return_an_infeasible_result_instead_of_raising(self):
        solution = solve_two_phase_replacement(_two_phase_default_system(), "fosfomycin", 150, 3, 0.1, 480, 360)

        self.assertFalse(solution.feasible)
        self.assertEqual(solution.central_stock_mg_ml, 0.0)


def _blaser_system_and_fos(setup, central_volume_ml: float = 6, extra_volume_ml: float = 6, duration_min: int = 60, interval_min: int = 360):
    system = SystemConfig(
        central_bottle_ml=100,
        cartridge_ml=70,
        extra_volume_ml=setup.extra_volume_ml,
        q_extra_to_central_ml_min=setup.q_extra_to_central_ml_min,
        q_extra_diluent_ml_min=setup.q_extra_diluent_ml_min,
        q_central_diluent_ml_min=setup.q_central_diluent_ml_min,
        q_waste_set_ml_min=setup.q_waste_ml_min,
    )
    fos = FosfomycinConfig(
        central_stock_mg_ml=setup.central_stock_mg_ml,
        extra_stock_mg_ml=setup.extra_stock_mg_ml,
        central_infusion_ml_min=central_volume_ml / duration_min,
        extra_infusion_ml_min=extra_volume_ml / duration_min,
        infusion_duration_min=duration_min,
        dosing_interval_min=interval_min,
        preload_extra_mg=0.0,
    )
    return system, fos


class BlaserSetupTest(unittest.TestCase):
    def test_textbook_flows_and_extra_volume_for_default_half_lives(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)

        self.assertAlmostEqual(setup.uncorrected_q_waste_ml_min, flow_for_half_life(170, 1.25), places=9)
        self.assertAlmostEqual(setup.q_central_diluent_ml_min, flow_for_half_life(170, 3), places=9)
        self.assertAlmostEqual(setup.q_extra_diluent_ml_min, 0.9165, places=4)
        self.assertAlmostEqual(setup.extra_volume_ml, 238.0, places=6)
        # The extra compartment washes out at the long half-life.
        self.assertAlmostEqual(half_life_for_flow(setup.extra_volume_ml, setup.q_extra_diluent_ml_min), 3.0, places=9)

    def test_outflow_pumps_carry_the_dose_volume_averaged_over_the_interval(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)

        self.assertAlmostEqual(setup.q_extra_to_central_ml_min, setup.q_extra_diluent_ml_min + 6 / 360, places=9)
        self.assertAlmostEqual(setup.q_waste_ml_min, flow_for_half_life(170, 1.25) + 12 / 360, places=9)

    def test_default_setup_tracks_the_true_curve_with_exact_cavg(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)

        self.assertTrue(setup.feasible)
        self.assertLess(setup.max_deviation_pct, 1.5)
        self.assertAlmostEqual(setup.predicted_cavg_mg_l, 150, places=6)
        self.assertAlmostEqual(setup.apparent_half_life_h, 3.0, delta=0.08)
        self.assertAlmostEqual(setup.predicted_cmin_mg_l, setup.reference_cmin_mg_l, delta=1.5)
        self.assertAlmostEqual(setup.dose_scale, 1.021, places=3)
        self.assertAlmostEqual(setup.central_dose_mg, 36.10, places=2)
        self.assertAlmostEqual(setup.extra_dose_mg, 50.54, places=2)

    def test_volumes_rise_by_the_dose_and_return_every_interval_without_drift(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)
        system, fos = _blaser_system_and_fos(setup)
        result = simulate_hfim("blaser", system, fos, [], duration_h=168, dt_min=1)
        rows = result.rows
        at_dose_start = [row for row in rows if row["time_min"] % 360 == 0]

        for row in at_dose_start:
            self.assertAlmostEqual(row["central_volume_ml"], 170, places=6)
            self.assertAlmostEqual(row["extra_volume_ml"], 238, places=6)
        self.assertAlmostEqual(max(row["central_volume_ml"] for row in rows), 175, places=6)
        self.assertAlmostEqual(max(row["extra_volume_ml"] for row in rows), 243, places=6)
        self.assertAlmostEqual(setup.central_volume_range_ml[1] - setup.central_volume_range_ml[0], 5, places=6)

    def test_full_run_holds_steady_state_auc_and_half_life_through_day_seven(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)
        system, fos = _blaser_system_and_fos(setup)
        result = simulate_hfim("blaser", system, fos, [], duration_h=168, dt_min=1)
        summary = result.summary["fosfomycin"]
        rows = result.rows

        def daily_auc(day: int) -> float:
            day_rows = [row for row in rows if day * 24 <= row["time_h"] <= (day + 1) * 24]
            return sum(
                (a["central_mg_l"] + b["central_mg_l"]) * 0.5 * (b["time_h"] - a["time_h"])
                for a, b in zip(day_rows, day_rows[1:])
            )

        self.assertAlmostEqual(summary["central_auc_last_24h_mg_h_l"], 3600, delta=1.5)
        # No drift: day 3 and day 7 give the same exposure.
        self.assertAlmostEqual(daily_auc(2), daily_auc(6), delta=0.5)
        self.assertAlmostEqual(summary["central_cmin_last_24h_mg_l"], setup.predicted_cmin_mg_l, delta=0.05)

    def test_published_flows_without_the_pump_correction_drift_badly(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360, duration_h=168)

        self.assertGreater(setup.uncorrected_end_deviation_pct, 30)
        self.assertAlmostEqual(setup.uncorrected_end_volumes_ml[0], 170 + 28 * 6, delta=0.5)
        self.assertAlmostEqual(setup.uncorrected_end_volumes_ml[1], 238 + 28 * 6, delta=0.5)

    def test_continuous_infusion_drug_still_averages_its_target(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)
        system, fos = _blaser_system_and_fos(setup)
        imipenem = DrugConfig("imipenem", 9, 1.25, dosing_mode="continuous infusion only")
        result = simulate_hfim("blaser", system, fos, [imipenem], duration_h=168, dt_min=1)
        late = [row["central_mg_l"] for row in result.rows if row["drug"] == "imipenem" and row["time_h"] >= 144]

        self.assertAlmostEqual(sum(late) / len(late), 9, delta=0.05)
        self.assertGreater(min(late), 8.7)
        self.assertLess(max(late), 9.2)
        # Reservoir concentration uses the waste pump setting, not the textbook outflow.
        expected = 9 / 1000 * setup.q_waste_ml_min / setup.q_central_diluent_ml_min
        self.assertAlmostEqual(result.summary["imipenem"]["central_diluent_concentration_mg_ml"], expected, places=9)

    def test_preparation_table_lists_central_and_extra_doses(self):
        setup = solve_blaser_setup(100, 70, 1.25, 3, 150, 6, 6, 60, 360)
        system, fos = _blaser_system_and_fos(setup)
        prep = simulate_hfim("blaser", system, fos, [], duration_h=24, dt_min=1).summary["drug_preparation"]
        components = {row["component"]: row for row in prep}

        self.assertAlmostEqual(components["central q6h infusion"]["amount_mg"], setup.central_dose_mg, places=9)
        self.assertAlmostEqual(components["extra q6h infusion"]["amount_mg"], setup.extra_dose_mg, places=9)

    def test_drug_no_longer_than_central_washout_is_rejected_with_guidance(self):
        setup = solve_blaser_setup(100, 70, 1.25, 1.25, 150, 6, 6, 60, 360)

        self.assertFalse(setup.feasible)
        self.assertEqual(setup.central_stock_mg_ml, 0.0)
        self.assertIn("1 half life page", setup.message)

    def test_other_half_life_pairs_and_intervals_stay_close(self):
        for short, long, interval in [(1.0, 6, 720), (2.0, 8, 720), (1.0, 2, 480), (1.25, 3, 720), (1.0, 12, 1440)]:
            setup = solve_blaser_setup(100, 70, short, long, 150, 6, 6, 60, interval)
            with self.subTest(short=short, long=long, interval=interval):
                self.assertTrue(setup.feasible)
                self.assertLess(setup.max_deviation_pct, 4)
                self.assertAlmostEqual(setup.predicted_cavg_mg_l, 150, places=6)

    def test_existing_scenarios_keep_inflow_equals_outflow_waste(self):
        system = SystemConfig()

        self.assertIsNone(system.q_waste_set_ml_min)
        self.assertAlmostEqual(system.q_waste_ml_min, system.q_extra_to_central_ml_min + system.q_central_diluent_ml_min, places=12)


if __name__ == "__main__":
    unittest.main()
