"""CPU orchestration integration using real online.evaluate_probe and providers.

Model/TOFU work and the Trainer constructor are synthetic. The RNG context is
stubbed (its arithmetic/state tests live elsewhere); no Torch, Transformers,
CUDA, DeepSpeed, or metric-equivalence claim is required by these tests.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
import online_epoch_evaluation as online
from forget_light_evaluation import prepare_observation


def fixed_result(loss=.45):
    return dict(raw_sensitive_loss=loss, raw_common_kl=.2,
        sample_mean_raw_sensitive_loss=loss, sample_mean_raw_common_kl=.2,
        sensitive_token_count=200, common_token_count=400, forget_count=200,
        forget_zero_sensitive_examples=0, forget_zero_common_examples=0,
        fixed_dataset_sha256="frozen", elapsed_seconds=3., fixed_loss_samples=[dict(
            split="forget", dataset_index=index, raw_sensitive_loss=loss,
            raw_common_kl=.2, raw_retain_nll=None, sensitive_token_count=1,
            common_token_count=2, retain_token_count=None) for index in range(200)])


class SyntheticProbe:
    """Implements only the interface consumed by the real online driver."""

    def __init__(self, output, metric_names):
        self._evaluation_metrics = tuple(metric_names)
        self.epoch_output = Path(output) / "online_evaluation_workspace"
        self.epoch_output.mkdir(parents=True, exist_ok=True)
        self._epoch_fixed_manifest = {"fixed_dataset_sha256": "frozen"}
        self._epoch_context = {}
        self.state = SimpleNamespace(global_step=0, epoch=0.)
        self.full_loss_calls = 0

    def _prepare_fixed_data(self):
        pass

    def _fixed_losses(self, step, epoch):
        self.full_loss_calls += 1
        return dict(raw_sensitive_loss=.8, raw_common_kl=.1, raw_retain_nll=1.2,
                    fixed_dataset_sha256="frozen", retain_count=3800)

    def evaluate(self):
        # The real Trainer's same provider hook is exercised through the real
        # online entry point; synthetic FQ/MU values are not benchmark results.
        provider = getattr(self, "_fixed_loss_provider", self._fixed_losses)
        values = provider(self.state.global_step, self.state.epoch)
        row = dict(values, step=self.state.global_step, forget_quality=.7,
            model_utility=.6 if "model_utility" in self._evaluation_metrics else None,
            evaluation_metrics=list(self._evaluation_metrics))
        with (self.epoch_output / "epoch_history.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


class DriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.runtime = dict(rank=0, distributed=False, model=object(),
                            reference=object(), engine=object(), device=object())
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.driver = SimpleNamespace(runtime=self.runtime, torch=self.torch,
            schedule={}, config={"trainer": {"method_args": {}}},
            manifest=dict(threshold_plan={"evaluation_policy": "forget_light_v1"},
                schedule_metadata={"fixed_dataset_sha256": "frozen"}, final_model_step=32))
        diagnostics = ModuleType("trainer.palu_diagnostics")
        diagnostics.preserve_rng_and_modes = lambda *_: nullcontext()
        modules = patch.dict(sys.modules, {"trainer.palu_diagnostics": diagnostics})
        modules.start()
        self.addCleanup(modules.stop)
        self.built = []

        def build(config, schedule, runtime, output, torch, *, metric_names=None):
            probe = SyntheticProbe(output, metric_names)
            self.built.append(probe)
            return probe

        construction = patch.object(online, "build_probe", side_effect=build)
        self.builder = construction.start()
        self.addCleanup(construction.stop)

    def evaluate(self, step, probe=None, **options):
        return online.evaluate_probe(self.driver.config, {}, self.runtime,
            self.output, self.torch, step=step, epoch=step * 32 / 200,
            epoch_index=step * 32 // 200, expected_fixed_hash="frozen",
            probe=probe, **options)

    def test_light_profiles_reuse_same_step_probe_and_rebuild_for_terminal_mu(self):
        with patch("forget_checkpoint_state.fixed_forget_probe", return_value=fixed_result()) as forward:
            initial_options, seconds = prepare_observation(self.driver, 0)
            self.assertEqual(seconds, 3.)
            initial, row, _ = self.evaluate(0, **initial_options)
            self.assertIsNone(row["model_utility"])
            self.assertFalse(row["fixed_loss_reused"])

            self.driver.fixed_forget_probe_result = (13, fixed_result(.55))
            options, seconds = prepare_observation(self.driver, 13)
            self.assertEqual(seconds, 0.)
            middle, row, _ = self.evaluate(13, probe=initial, **options)
            self.assertIs(middle, initial)
            self.assertEqual(self.builder.call_count, 1)
            self.assertEqual(row["raw_sensitive_loss"], .55)
            self.assertTrue(row["fixed_loss_reused"])
            self.assertIsNone(row["model_utility"])

            self.driver.fixed_forget_probe_result = (32, fixed_result(.45))
            options, seconds = prepare_observation(self.driver, 32)
            self.assertEqual(seconds, 0.)
            terminal, row, _ = self.evaluate(32, probe=middle, **options)
            self.assertIsNot(terminal, middle)
            self.assertEqual(self.builder.call_count, 2)
            self.assertEqual(row["model_utility"], .6)
            self.assertEqual(row["evaluation_metrics"], ["forget_quality", "model_utility"])
            self.assertIsNone(row["raw_retain_nll"])
            self.assertIsNone(row["loss_retain_nll"])
            self.assertEqual(row["retain_count"], 0)
            forward.assert_called_once()

        self.assertTrue(all(probe.full_loss_calls == 0 for probe in self.built))
        samples = [json.loads(line) for line in
            (terminal.epoch_output / "fixed_loss_samples.jsonl").read_text().splitlines()]
        self.assertEqual(len(samples), 600)
        self.assertEqual({sample["step"] for sample in samples}, {0, 13, 32})
        self.assertEqual({sample["split"] for sample in samples}, {"forget"})
        self.assertTrue(all(sample["raw_retain_nll"] is None for sample in samples))

    def test_legacy_default_still_constructs_full_metrics_and_fixed_losses(self):
        probe, row, _ = self.evaluate(0)
        self.assertEqual(probe._evaluation_metrics, ("forget_quality", "model_utility"))
        self.assertEqual(probe.full_loss_calls, 1)
        self.assertEqual(row["model_utility"], .6)
        self.assertEqual(row["raw_retain_nll"], 1.2)
        self.assertEqual(row["retain_count"], 3800)
        self.assertFalse(hasattr(probe, "_fixed_loss_provider"))

    def test_legacy_call_clears_light_provider_on_reused_full_metric_probe(self):
        self.driver.fixed_forget_probe_result = (32, fixed_result())
        options, _ = prepare_observation(self.driver, 32)
        probe, row, _ = self.evaluate(32, **options)
        self.assertIsNone(row["raw_retain_nll"])
        self.assertTrue(hasattr(probe, "_fixed_loss_provider"))
        same, row, _ = self.evaluate(33, probe=probe)
        self.assertIs(same, probe)
        self.assertEqual(self.builder.call_count, 1)
        self.assertFalse(hasattr(probe, "_fixed_loss_provider"))
        self.assertEqual(probe.full_loss_calls, 1)
        self.assertEqual(row["raw_retain_nll"], 1.2)


if __name__ == "__main__":
    unittest.main()
