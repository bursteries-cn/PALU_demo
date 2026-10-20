"""Sample-mask persistence and fixed-loss semantics; CPU mocks, not GPU proof."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
import forget_checkpoint_state as checkpoint
import forget_light_evaluation as light
from prepare_layout_schedule import canonical_sha


def state_fields():
    state = dict(rows=[
        dict(dataset_index=0, sensitive_token_count=2, active=False),
        dict(dataset_index=1, sensitive_token_count=1, active=True),
        dict(dataset_index=2, sensitive_token_count=0, active=False)])
    return dict(sample_stop_state=state, sample_stop_state_sha256=canonical_sha(state))


def probe_result():
    counts, losses = [2, 1, 0], [.3, 1.2, 0.]
    return dict(raw_sensitive_loss=.6, raw_common_kl=.2,
        sample_mean_raw_sensitive_loss=.5, sample_mean_raw_common_kl=.2,
        sensitive_token_count=3, common_token_count=3, forget_count=3,
        forget_zero_sensitive_examples=1, forget_zero_common_examples=0,
        fixed_dataset_sha256="frozen", elapsed_seconds=3.,
        fixed_loss_samples=[dict(split="forget", dataset_index=index,
            raw_sensitive_loss=loss, raw_common_kl=.2, raw_retain_nll=None,
            sensitive_token_count=count, common_token_count=1, retain_token_count=None)
            for index, (count, loss) in enumerate(zip(counts, losses))])


class MetadataTests(unittest.TestCase):
    def test_optional_state_is_validated_and_deep_copied(self):
        self.assertEqual(checkpoint.sample_stop_state_fields({}), {})
        fields = state_fields()
        saved = checkpoint.sample_stop_state_fields(fields)
        self.assertEqual(saved, fields)
        fields["sample_stop_state"]["rows"][0]["active"] = True
        self.assertFalse(saved["sample_stop_state"]["rows"][0]["active"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            checkpoint.sample_stop_state_fields(fields)
        for key in ("sample_stop_state", "sample_stop_state_sha256"):
            incomplete = state_fields()
            del incomplete[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing"):
                checkpoint.sample_stop_state_fields(incomplete)

    def test_masked_diagnostic_retains_original_denominator(self):
        result = light.sample_masked_diagnostic(probe_result(), state_fields())
        self.assertAlmostEqual(result["sample_masked_sensitive_loss_next_update"], .4)
        self.assertAlmostEqual(result["sample_masked_loss_total_next_update"], .8)
        self.assertIn("unmasked", result["loss_total_semantics"])
        self.assertIn("original full-pool", result["sample_masked_loss_semantics"])
        wrong = dict(probe_result(), sample_masked_sensitive_loss_next_update=1.2)
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            light.sample_masked_diagnostic(wrong, state_fields())

    def test_invalid_masks_rejected_and_all_stopped_still_keeps_kl(self):
        for change in ("missing", "duplicate", "count", "zero_active", "not_bool"):
            fields = state_fields()
            rows = fields["sample_stop_state"]["rows"]
            if change == "missing":
                rows.pop()
            elif change == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif change == "count":
                rows[0]["sensitive_token_count"] = 3
            elif change == "zero_active":
                rows[2]["active"] = True
            else:
                rows[1]["active"] = 1
            fields["sample_stop_state_sha256"] = canonical_sha(fields["sample_stop_state"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                light.sample_masked_diagnostic(probe_result(), fields)
        fields = state_fields()
        fields["sample_stop_state"]["rows"][1]["active"] = False
        fields["sample_stop_state_sha256"] = canonical_sha(fields["sample_stop_state"])
        result = light.sample_masked_diagnostic(probe_result(), fields)
        self.assertEqual(result["sample_masked_sensitive_loss_next_update"], 0.)
        self.assertAlmostEqual(result["sample_masked_loss_total_next_update"], .4)

    def test_initial_and_cached_evaluations_label_unmasked_loss(self):
        manifest = dict(threshold_plan=dict(evaluation_policy="forget_light_v1",
            stopping_scope="sample_permanent"), final_model_step=63,
            schedule_metadata=dict(fixed_dataset_sha256="frozen"), **state_fields())
        driver = SimpleNamespace(manifest=manifest, runtime={}, schedule={},
            config={"trainer": {"method_args": {}}}, torch=None)
        with patch.object(checkpoint, "fixed_forget_probe", return_value=probe_result()) as probe:
            options, _ = light.prepare_observation(driver, 0)
            cached, seconds = light.prepare_observation(driver, 0)
            probe.assert_called_once()
            self.assertEqual(seconds, 0.)
            self.assertEqual(options["metric_names"], ["forget_quality"])
            with tempfile.TemporaryDirectory() as temporary:
                target = SimpleNamespace(epoch_output=Path(temporary))
                values = light.fixed_loss_provider(target, cached["fixed_forget_result"])(0, 0.)
                self.assertAlmostEqual(values["loss_total"], 1.)
                self.assertAlmostEqual(values["sample_masked_loss_total_next_update"], .8)
                self.assertIn("unmasked", values["loss_total_semantics"])
                self.assertIsNone(values["raw_retain_nll"])
                self.assertEqual(len((Path(temporary)/"fixed_loss_samples.jsonl").read_text().splitlines()), 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class NativePersistenceMockTests(unittest.TestCase):
    def test_native_client_receipt_and_restore_bind_optional_mask(self):
        import torch
        import test_forget_checkpoint_state as existing
        fixture = existing.NativeContractMockTests()
        live = fixture.runtime()
        for _ in range(3):
            fixture.advance(live)
        manifest = dict.fromkeys(checkpoint.IDENTITY_KEYS, "synthetic-consistent-fingerprint")
        manifest.update(state_fields())
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(torch.distributed, "broadcast_object_list", side_effect=lambda *a, **k: None), \
                patch.object(checkpoint.state, "all_ranks", side_effect=lambda value, *a: [copy.deepcopy(value) for _ in range(4)]):
            folder = Path(temporary) / "native"
            receipt = checkpoint.save_resume_checkpoint(folder, live, manifest, 3, torch)
            saved = torch.load(folder / "step3/mp_rank_00_model_states.pt", map_location="cpu", weights_only=False)
            for key, value in state_fields().items():
                self.assertEqual(receipt[key], value)
                self.assertEqual(saved["client"][key], value)
            loaded = checkpoint.load_resume_checkpoint(folder, fixture.runtime(), manifest, torch)
            self.assertEqual(loaded["sample_stop_state"], manifest["sample_stop_state"])
            wrong = copy.deepcopy(manifest)
            wrong["sample_stop_state"]["rows"][1]["active"] = False
            wrong["sample_stop_state_sha256"] = canonical_sha(wrong["sample_stop_state"])
            with self.assertRaisesRegex(ValueError, "stopping state differ"):
                checkpoint.load_resume_checkpoint(folder, fixture.runtime(), wrong, torch)
            broken = copy.deepcopy(receipt)
            broken["sample_stop_state"]["rows"][1]["active"] = False
            with self.assertRaisesRegex(ValueError, "SHA256"):
                checkpoint.validate_resume_files(folder, broken)
            # A receipt cannot pretend this is a legacy checkpoint and silently
            # discard the sample mask still embedded in native client_state.
            stripped = {key: value for key, value in receipt.items()
                        if key not in state_fields()}
            (folder / "receipt.json").write_text(json.dumps(stripped))
            old_manifest = {key: manifest[key] for key in checkpoint.IDENTITY_KEYS}
            with self.assertRaisesRegex(ValueError, "complete expected restart state"):
                checkpoint.load_resume_checkpoint(folder, fixture.runtime(), old_manifest, torch)


if __name__ == "__main__":
    unittest.main()
