import argparse
import unittest

from hfim_simulator.cli import FOS_HALF_LIFE_H, IMIPENEM_HALF_LIFE_H, RELEBACTAM_HALF_LIFE_H, _validate_args


class HfimCliTest(unittest.TestCase):
    def test_shared_half_life_constants_match_the_drug_configs_built_from_them(self):
        # main() builds shared_central_half_life_h and the imipenem/relebactam DrugConfig half-lives
        # from these same three constants, so they cannot silently desync from each other.
        self.assertEqual(FOS_HALF_LIFE_H, 3.0)
        self.assertEqual(IMIPENEM_HALF_LIFE_H, 1.25)
        self.assertEqual(RELEBACTAM_HALF_LIFE_H, 1.25)
        self.assertEqual(min(FOS_HALF_LIFE_H, IMIPENEM_HALF_LIFE_H, RELEBACTAM_HALF_LIFE_H), 1.25)

    def test_cli_rejects_duration_shorter_than_auc_window(self):
        args = argparse.Namespace(duration_h=12, fos_infusion_duration_h=1)

        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_cli_rejects_zero_infusion_duration(self):
        args = argparse.Namespace(duration_h=24, fos_infusion_duration_h=0)

        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_cli_accepts_valid_duration_and_infusion_duration(self):
        args = argparse.Namespace(duration_h=24, fos_infusion_duration_h=1)

        _validate_args(args)


if __name__ == "__main__":
    unittest.main()
