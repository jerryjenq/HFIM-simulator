import argparse
import unittest

from hfim_simulator.cli import _validate_args


class HfimCliTest(unittest.TestCase):
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
