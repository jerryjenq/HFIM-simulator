import unittest

from hfim_simulator.app import (
    _central_diluent_default_for_flow_mode,
    _central_diluent_reservoir_rows,
    _central_diluent_reservoir_summary,
    _preparation_destination_cards,
    _preparation_review_rows,
    _shared_central_half_life_from_widget_state,
    _prep_rows_for_display,
    _replacement_solution_rows,
    _replacement_solution_summary,
    _normalized_unique_drug_name,
    _qextra_default_for_scenario,
)
from hfim_simulator.pk import FosfomycinConfig, SystemConfig


class HfimAppLogicTest(unittest.TestCase):
    def test_q24_replacement_extra_transfer_default_is_not_auto_scaled_by_volume(self):
        default_small_volume = _qextra_default_for_scenario(
            "q24_replacement",
            "Auto from target half-life",
            auto_extra_flow=0.2,
        )
        default_large_volume = _qextra_default_for_scenario(
            "q24_replacement",
            "Auto from target half-life",
            auto_extra_flow=1.2,
        )

        self.assertAlmostEqual(default_small_volume, 0.167)
        self.assertAlmostEqual(default_large_volume, 0.167)

    def test_overflow_extra_transfer_default_can_use_auto_half_life_flow(self):
        self.assertAlmostEqual(
            _qextra_default_for_scenario("overflow", "Auto from target half-life", auto_extra_flow=0.928),
            0.928,
        )
        self.assertAlmostEqual(
            _qextra_default_for_scenario("overflow", "Manual flow entry", auto_extra_flow=0.928),
            0.921,
        )

    def test_auto_flow_mode_uses_recalculated_central_diluent_flow(self):
        self.assertAlmostEqual(
            _central_diluent_default_for_flow_mode(
                "Auto flow from target half-life (fixed volume)",
                auto_central_flow=1.571,
                q_extra_to_central=0.167,
            ),
            1.404,
            places=3,
        )
        self.assertAlmostEqual(
            _central_diluent_default_for_flow_mode(
                "Manual flow entry (do not auto-adjust)",
                auto_central_flow=0.479,
                q_extra_to_central=0.167,
            ),
            0.65,
        )

    def test_shared_central_half_life_uses_shortest_active_drug_half_life(self):
        state = {
            "half_life_0": 3.0,
            "half_life_1": 1.25,
            "half_life_2": 1.5,
        }

        self.assertAlmostEqual(_shared_central_half_life_from_widget_state(3, state), 1.25)

    def test_drug_names_are_normalized_and_kept_unique(self):
        names = set()
        first = _normalized_unique_drug_name("Imipenem", 0, names)
        names.add(first)
        duplicate = _normalized_unique_drug_name(" imipenem ", 1, names)
        names.add(duplicate)
        blank = _normalized_unique_drug_name("", 2, names)

        self.assertEqual(first, "imipenem")
        self.assertEqual(duplicate, "imipenem_2")
        self.assertEqual(blank, "drug3")

    def test_q24_display_separates_central_dosing_from_extra_replacement(self):
        rows = [
            {"Drug": "fosfomycin", "Component": "central q6h infusion", "Amount": "31.391 mg", "Daily amount": "125.562 mg/day", "Note": "central"},
            {"Drug": "fosfomycin", "Component": "extra q24h fixed-concentration solution", "Amount": "147.299 mg", "Daily amount": "147.299 mg/day", "Note": "extra"},
            {"Drug": "imipenem", "Component": "loading dose", "Amount": "3.060 mg", "Daily amount": "", "Note": "other"},
        ]

        setup_rows, extra_rows, other_rows = _prep_rows_for_display(rows, "fosfomycin", "q24_replacement")

        self.assertEqual([row["Component"] for row in setup_rows], ["central q6h infusion"])
        self.assertEqual([row["Component"] for row in extra_rows], ["extra q24h fixed-concentration solution"])
        self.assertEqual([row["Drug"] for row in other_rows], ["imipenem"])

    def test_overflow_display_keeps_setup_drug_rows_together(self):
        rows = [
            {"Drug": "fosfomycin", "Component": "central q6h infusion", "Amount": "31.391 mg", "Daily amount": "125.562 mg/day", "Note": "central"},
            {"Drug": "fosfomycin", "Component": "extra q6h infusion", "Amount": "50.109 mg", "Daily amount": "200.434 mg/day", "Note": "extra"},
        ]

        setup_rows, extra_rows, other_rows = _prep_rows_for_display(rows, "fosfomycin", "overflow")

        self.assertEqual(len(setup_rows), 2)
        self.assertEqual(extra_rows, [])
        self.assertEqual(other_rows, [])

    def test_replacement_solution_summary_counts_only_the_extra_fill_as_preparation(self):
        summary = _replacement_solution_summary(
            SystemConfig(extra_volume_ml=241, q_extra_to_central_ml_min=0.167),
            FosfomycinConfig(extra_stock_mg_ml=0.3, reservoir_replacement_interval_h=24),
            duration_h=168,
        )

        self.assertAlmostEqual(summary["prepared_volume_per_interval_ml"], 241)
        self.assertAlmostEqual(summary["prepared_drug_per_interval_mg"], 72.3)
        self.assertAlmostEqual(summary["total_volume_with_overfill_ml"], 1855.7)
        self.assertAlmostEqual(summary["total_drug_with_overfill_mg"], 556.71)
        self.assertAlmostEqual(summary["transfer_volume_ml"], 240.48)
        self.assertAlmostEqual(summary["transfer_drug_mg"], 72.144)

    def test_replacement_solution_rows_use_experimental_prep_language(self):
        rows = _replacement_solution_rows(
            SystemConfig(extra_volume_ml=241, q_extra_to_central_ml_min=0.167),
            FosfomycinConfig(extra_stock_mg_ml=0.3, reservoir_replacement_interval_h=24),
            duration_h=168,
        )

        self.assertEqual(rows[0]["Use"], "Fill the extra compartment at the start of each 24 h block")
        self.assertEqual(rows[1]["Use"], "Drug delivered from extra to central during the same 24 h block")
        self.assertEqual(rows[2]["Use"], "Total solution to prepare for one 24 h block")
        self.assertIn("241.0 mL", rows[0]["How calculated"])
        self.assertIn("0.167 mL/min x 24 h", rows[1]["How calculated"])
        self.assertEqual(rows[2]["Drug per interval"], "72.300 mg")

    def test_central_diluent_reservoir_rows_show_daily_ci_drug_prep(self):
        summary = {
            "imipenem": {
                "dosing_mode": "loading dose + continuous infusion",
                "target_concentration_mg_l": 9,
                "central_diluent_concentration_mg_ml": 0.01007,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 20.362,
            },
            "relebactam": {
                "dosing_mode": "loading dose + continuous infusion",
                "target_concentration_mg_l": 6,
                "central_diluent_concentration_mg_ml": 0.006713,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 13.575,
            },
        }
        recipe = _central_diluent_reservoir_summary(summary, duration_h=168)
        rows = _central_diluent_reservoir_rows(summary, duration_h=168)

        self.assertEqual(recipe["volume_q24h"], "2022.0 mL")
        self.assertEqual(recipe["extra_volume_q24h_10_percent"], "202.2 mL")
        self.assertEqual(recipe["prepared_volume_q24h"], "2224.2 mL")
        self.assertEqual(recipe["prepared_volume_total"], "15569.4 mL")
        self.assertEqual([row["Drug"] for row in rows], ["imipenem", "relebactam"])
        self.assertIn("0.010070 mg/mL", rows[0]["Central diluent concentration"])
        self.assertEqual(rows[0]["Required drug per q24h"], "20.362 mg")
        self.assertEqual(rows[0]["10% extra drug q24h"], "2.036 mg")
        self.assertEqual(rows[0]["Drug to weigh q24h"], "22.398 mg")
        self.assertEqual(rows[0]["Drug to weigh 168 h"], "156.787 mg")
        self.assertIn("same shared reservoir", rows[0]["Note"])

    def test_preparation_cards_group_all_drugs_by_dosing_destination(self):
        prep_rows = [
            {"Drug": "fosfomycin", "Component": "central q6h infusion", "Amount": "31.391 mg", "Daily amount": "125.562 mg/day", "Note": "6 mL over 1 h"},
            {"Drug": "fosfomycin", "Component": "extra q24h fixed-concentration solution", "Amount": "73.729 mg", "Daily amount": "73.729 mg/day", "Note": "prepare 241 mL"},
            {"Drug": "imipenem", "Component": "loading dose", "Amount": "3.060 mg", "Daily amount": "", "Note": "target 18 mg/L over 0.5 h"},
            {"Drug": "imipenem", "Component": "continuous infusion", "Amount": "0.848 mg/h", "Daily amount": "20.362 mg/day", "Note": "mixed into central diluent reservoir"},
            {"Drug": "relebactam", "Component": "loading dose", "Amount": "2.040 mg", "Daily amount": "", "Note": "target 12 mg/L over 0.5 h"},
            {"Drug": "relebactam", "Component": "continuous infusion", "Amount": "0.566 mg/h", "Daily amount": "13.575 mg/day", "Note": "mixed into central diluent reservoir"},
        ]
        summary = {
            "imipenem": {
                "target_concentration_mg_l": 9,
                "central_diluent_concentration_mg_ml": 0.01007,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 20.362,
            },
            "relebactam": {
                "target_concentration_mg_l": 6,
                "central_diluent_concentration_mg_ml": 0.006713,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 13.575,
            },
        }

        cards = _preparation_destination_cards(
            prep_rows,
            summary,
            SystemConfig(extra_volume_ml=241),
            FosfomycinConfig(drug_name="fosfomycin", extra_stock_mg_ml=0.305929, reservoir_replacement_interval_h=24),
            duration_h=168,
        )

        self.assertEqual([card["title"] for card in cards], [
            "Central direct dosing",
            "Central diluent q24h reservoir",
            "Extra q24h replacement",
        ])
        self.assertEqual(cards[0]["drug_names"], "fosfomycin, imipenem, relebactam")
        self.assertEqual(cards[1]["primary_value"], "2224.2 mL")
        self.assertIn("one shared reservoir", cards[1]["caption"])
        self.assertEqual(cards[2]["primary_value"], "265.1 mL")

    def test_preparation_review_rows_indicate_destination_and_amount_to_weigh(self):
        prep_rows = [
            {"Drug": "fosfomycin", "Component": "central q6h infusion", "Amount": "31.391 mg", "Daily amount": "125.562 mg/day", "Note": "6 mL over 1 h"},
            {"Drug": "fosfomycin", "Component": "extra q24h fixed-concentration solution", "Amount": "73.729 mg", "Daily amount": "73.729 mg/day", "Note": "prepare 241 mL"},
            {"Drug": "imipenem", "Component": "loading dose", "Amount": "3.060 mg", "Daily amount": "", "Note": "target 18 mg/L over 0.5 h"},
            {"Drug": "imipenem", "Component": "continuous infusion", "Amount": "0.848 mg/h", "Daily amount": "20.362 mg/day", "Note": "mixed into central diluent reservoir"},
            {"Drug": "relebactam", "Component": "loading dose", "Amount": "2.040 mg", "Daily amount": "", "Note": "target 12 mg/L over 0.5 h"},
            {"Drug": "relebactam", "Component": "continuous infusion", "Amount": "0.566 mg/h", "Daily amount": "13.575 mg/day", "Note": "mixed into central diluent reservoir"},
        ]
        summary = {
            "imipenem": {
                "target_concentration_mg_l": 9,
                "central_diluent_concentration_mg_ml": 0.01007,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 20.362,
            },
            "relebactam": {
                "target_concentration_mg_l": 6,
                "central_diluent_concentration_mg_ml": 0.006713,
                "central_diluent_volume_per_24h_ml": 2022.0,
                "central_diluent_drug_per_24h_mg": 13.575,
            },
        }

        rows = _preparation_review_rows(
            prep_rows,
            summary,
            SystemConfig(extra_volume_ml=241),
            FosfomycinConfig(drug_name="fosfomycin", extra_stock_mg_ml=0.305929, reservoir_replacement_interval_h=24),
            duration_h=168,
        )

        self.assertEqual({row["Drug"] for row in rows}, {"fosfomycin", "imipenem", "relebactam"})
        imipenem_ci = [row for row in rows if row["Drug"] == "imipenem" and row["Dosing part"] == "continuous infusion"][0]
        self.assertEqual(imipenem_ci["Add into"], "Central diluent q24h reservoir")
        self.assertEqual(imipenem_ci["Required amount"], "20.362 mg/q24h")
        self.assertEqual(imipenem_ci["10% extra"], "2.036 mg")
        self.assertEqual(imipenem_ci["Amount to weigh"], "22.398 mg/q24h")
        self.assertEqual(imipenem_ci["Volume"], "shared 2224.2 mL q24h")

        extra_row = [row for row in rows if row["Add into"] == "Extra q24h replacement"][0]
        self.assertEqual(extra_row["Drug"], "fosfomycin")
        self.assertEqual(extra_row["Concentration"], "0.305929 mg/mL")
        self.assertEqual(extra_row["Amount to weigh"], "81.102 mg/q24h")


if __name__ == "__main__":
    unittest.main()
