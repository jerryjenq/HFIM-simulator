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
    simulate_hfim,
    solve_css_cmax_replacement,
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


if __name__ == "__main__":
    unittest.main()
