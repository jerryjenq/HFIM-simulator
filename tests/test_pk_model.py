import math
import unittest

from hfim_simulator.pk import (
    DrugConfig,
    FosfomycinConfig,
    SystemConfig,
    compute_continuous_infusion,
    flow_for_half_life,
    half_life_for_flow,
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

    def test_central_only_drugs_use_shared_physical_waste_flow(self):
        shared_flow = flow_for_half_life(170, 1.25)
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

        self.assertAlmostEqual(result.summary["testdrug"]["infusion_rate_mg_h"], 0.848, places=3)
        self.assertAlmostEqual(result.summary["testdrug"]["daily_amount_mg"], 20.36, places=2)

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
                ),
            ],
            duration_h=1,
            dt_min=1,
        )

        imipenem_rows = [row for row in result.rows if row["drug"] == "imipenem"]
        self.assertEqual(imipenem_rows[0]["central_mg_l"], 0)
        self.assertGreater(max(row["central_mg_l"] for row in imipenem_rows), 9)

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
