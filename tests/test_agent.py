import os
import tempfile
import unittest
from pathlib import Path

from hfim_simulator.agent import ask_setup_agent, build_agent_context, load_env_file, rule_based_setup_reply


class HfimSetupAgentTest(unittest.TestCase):
    def test_load_env_file_reads_simple_key_without_leaking_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=secret-value\nGEMINI_MODEL=test-model\n", encoding="utf-8")

            loaded = load_env_file(env_path)

        self.assertEqual(loaded["GEMINI_API_KEY"], "secret-value")
        self.assertEqual(loaded["GEMINI_MODEL"], "test-model")

    def test_rule_based_reply_explains_loading_target_and_duration(self):
        context = build_agent_context(
            system={"central_volume_ml": 170, "extra_volume_ml": 241},
            setup_drug_name="fosfomycin",
            drug_inputs={
                "imipenem": {
                    "target_concentration_mg_l": 9,
                    "half_life_h": 1.25,
                    "loading_dose": True,
                    "loading_target_concentration_mg_l": 18,
                    "loading_duration_h": 0.5,
                    "maintenance": "continuous infusion",
                }
            },
            summary={
                "imipenem": {
                    "loading_dose_mg": 3.06,
                    "infusion_rate_mg_h": 0.848,
                    "daily_amount_mg": 20.36,
                }
            },
        )

        reply = rule_based_setup_reply("why is central high at time zero?", context)

        self.assertIn("loading target", reply.lower())
        self.assertIn("0.5 h", reply)
        self.assertIn("3.060 mg", reply)

    def test_ask_setup_agent_falls_back_without_key_and_does_not_include_secret(self):
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            reply = ask_setup_agent(
                "help me setup imipenem",
                {"setup_drug_name": "fosfomycin", "drug_inputs": {}, "summary": {}},
                env={},
                use_gemini=False,
            )
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

        self.assertEqual(reply["source"], "local_rules")
        self.assertNotIn("GEMINI_API_KEY", reply["message"])


if __name__ == "__main__":
    unittest.main()
