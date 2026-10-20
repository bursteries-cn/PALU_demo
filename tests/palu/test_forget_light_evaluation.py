"""Light evaluation policy, cache provenance and same-step stopping integration."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
from forget_light_evaluation import prepare_observation, fixed_loss_provider
from forget_threshold_control import ForgetThresholdControl, validate_light_probes
from forget_until_threshold import protocol


def result():
    return dict(raw_sensitive_loss=.45, raw_common_kl=.2,
        sample_mean_raw_sensitive_loss=.45, sample_mean_raw_common_kl=.2,
        sensitive_token_count=2, common_token_count=3, forget_count=1,
        forget_zero_sensitive_examples=0, forget_zero_common_examples=0,
        fixed_dataset_sha256="frozen", elapsed_seconds=3., fixed_loss_samples=[dict(
            split="forget", dataset_index=0, raw_sensitive_loss=.45, raw_common_kl=.2,
            raw_retain_nll=None, sensitive_token_count=2, common_token_count=3, retain_token_count=None)])


def online():
    return SimpleNamespace(manifest=dict(threshold_plan=protocol(), final_model_step=63,
        schedule_metadata=dict(fixed_dataset_sha256="frozen")), runtime={}, schedule={},
        config={"trainer": {"method_args": {}}}, torch=None)


class LightTests(unittest.TestCase):
    def test_all_epoch_probe_samples_reconstruct_both_losses(self):
        raw = result()
        row = dict(raw, step=7)
        samples = [dict(r, step=7) for r in raw["fixed_loss_samples"]]
        metadata = dict(data_sizes=dict(forget=1), fixed_dataset_sha256="frozen",
                        fixed_target_counts=dict(sensitive=2, common=3))
        validate_light_probes([row], samples, metadata)
        for wrong in ([], samples + samples, [dict(samples[0], step=13)],
                      [dict(samples[0], raw_common_kl=.3)], [dict(samples[0], common_token_count=4)]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                validate_light_probes([row], wrong, metadata)

    def test_initial_and_intermediate_exclude_mu_final_includes_once(self):
        driver = online()
        with patch("forget_checkpoint_state.fixed_forget_probe", return_value=result()) as probe:
            initial, seconds = prepare_observation(driver, 0)
            self.assertEqual(initial["metric_names"], ["forget_quality"])
            self.assertEqual(seconds, 3.)
            self.assertFalse(initial["fixed_forget_result"]["fixed_loss_reused"])
            middle, _ = prepare_observation(driver, 13)
            self.assertEqual(middle["metric_names"], ["forget_quality"])
            driver.manifest["final_model_step"] = 25  # early stop replaces periodic event
            final, _ = prepare_observation(driver, 25)
            self.assertEqual(final["metric_names"], ["forget_quality", "model_utility"])
            self.assertEqual(probe.call_count, 3)
            self.assertTrue(probe.call_args.kwargs["include_common_kl"])

    def test_cache_reused_only_at_same_step_and_wrong_dataset_rejected(self):
        driver = online()
        driver.fixed_forget_probe_result = (13, result())
        with patch("forget_checkpoint_state.fixed_forget_probe", return_value=result()) as probe:
            options, seconds = prepare_observation(driver, 13)
            probe.assert_not_called()
            self.assertTrue(options["fixed_forget_result"]["fixed_loss_reused"])
            self.assertEqual(seconds, 0.)
            prepare_observation(driver, 25)
            probe.assert_called_once()
        driver.fixed_forget_probe_result[1]["fixed_dataset_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "different fixed dataset"):
            prepare_observation(driver, 25)

    def test_fixed_provider_records_only_f_and_missing_r_never_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = SimpleNamespace(epoch_output=Path(tmp))
            values = fixed_loss_provider(probe, dict(result(), fixed_loss_reused=True))(13, 2.08)
            self.assertAlmostEqual(values["loss_total"], .85)
            self.assertEqual(values["fixed_loss_scope"], "forget_only")
            self.assertEqual(values["retain_count"], 0)
            self.assertIsNone(values["raw_retain_nll"])
            self.assertIsNone(values["loss_retain_nll"])
            rows = [json.loads(x) for x in (Path(tmp)/"fixed_loss_samples.jsonl").read_text().splitlines()]
            self.assertEqual([(r["step"], r["epoch"], r["split"]) for r in rows], [(13, 2.08, "forget")])

    def test_epoch_stop_passes_probe_to_terminal_eval_without_second_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = online()
            obj = ForgetThresholdControl(protocol())
            obj.configure_online(driver)
            driver.manifest["evaluation_steps"] = [0, 13, 25, 38, 50, 63]
            args = SimpleNamespace(output=Path(tmp))
            runtime = dict(rank=0, distributed=False)
            with patch("forget_checkpoint_state.fixed_forget_probe", return_value=result()) as probe:
                for step in range(1, 8):
                    stop = obj.after_update(runtime, driver.manifest, driver.config, {}, args, step, driver, None)
                self.assertTrue(stop)
                options, seconds = prepare_observation(driver, 7)
                probe.assert_called_once()
                self.assertEqual(seconds, 0.)
                self.assertEqual(options["metric_names"], ["forget_quality", "model_utility"])
            history = json.loads((Path(tmp)/"forget_probe_history.jsonl").read_text())
            self.assertEqual(history["raw_common_kl"], .2)
            self.assertNotIn("fixed_loss_samples", history)
            self.assertEqual(json.loads((Path(tmp)/"forget_probe_samples.jsonl").read_text())["step"], 7)


if __name__ == "__main__":
    unittest.main()
