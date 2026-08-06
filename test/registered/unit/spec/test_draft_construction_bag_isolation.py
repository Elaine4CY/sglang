"""A draft runner's construction must not rewrite the process-wide config record.

The bags are the target's resolved configuration. Three writers used to run
during a draft's build or weight update and land on them:

- ``declare_load_time_override`` in model files (the nextn/MTP draft *is* a
  DeepSeek/GLM/Qwen3.5/MiniMax model, and its checkpoint's fusion/quantization
  decisions are its own, not the target's) — now a scoped window while the
  draft loads, so the draft's construction reads still see its declaration;
- the SM100 GDN prefill default (computed from the runner's own model config
  and mamba pool) — now recorded for the target runner only;
- ``update_model_fields`` after a draft weight update (the draft's new
  ``model_path`` is not the process's) — now target-only.
"""

import unittest

from sglang.srt.arg_groups.overrides import (
    declare_load_time_override,
    draft_model_load_scope,
)
from sglang.srt.layers.attention.linear.gdn_backend import record_gdn_prefill_default
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_exec, get_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftConstructionBagIsolation(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def _log_sources(self):
        return [source for source, _fields in get_context().overrides_log()]

    # -- declare_load_time_override ------------------------------------------

    def test_declaration_outside_a_draft_load_is_permanent(self):
        self._seed(disable_shared_experts_fusion=False)
        declare_load_time_override(
            "target-model", {"disable_shared_experts_fusion": True}
        )
        self.assertTrue(get_exec().moe.disable_shared_experts_fusion)
        self.assertIn("target-model", self._log_sources())

    def test_a_draft_declaration_is_a_window_not_a_config_change(self):
        self._seed(disable_shared_experts_fusion=False)
        with draft_model_load_scope():
            declare_load_time_override(
                "draft-model", {"disable_shared_experts_fusion": True}
            )
            # The draft's own construction reads see its declaration...
            self.assertTrue(get_exec().moe.disable_shared_experts_fusion)
        # ...and the target's record is untouched afterwards.
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)
        self.assertNotIn("draft-model", self._log_sources())

    def test_a_draft_window_still_validates_the_whitelist(self):
        self._seed()
        with draft_model_load_scope():
            with self.assertRaises(ValueError):
                declare_load_time_override("draft-model", {"model_path": "x"})

    def test_stacked_draft_declarations_all_restore(self):
        self._seed(disable_shared_experts_fusion=False, sampling_backend="pytorch")
        with draft_model_load_scope():
            declare_load_time_override(
                "draft-model", {"disable_shared_experts_fusion": True}
            )
            declare_load_time_override(
                "draft-model-2", {"sampling_backend": "flashinfer"}
            )
            self.assertTrue(get_exec().moe.disable_shared_experts_fusion)
            self.assertEqual(get_exec().kernel.sampling_backend, "flashinfer")
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)
        self.assertEqual(get_exec().kernel.sampling_backend, "pytorch")

    def test_a_target_declaration_inside_a_draft_window_stays_scoped(self):
        """The scope is per-load: everything declared under it restores.

        The weight-driven dtype fallback in load_model_utils also runs inside
        the draft's load; it must not leak either.
        """
        self._seed(disable_shared_experts_fusion=False)
        with draft_model_load_scope():
            declare_load_time_override(
                "load_model_utils.dtype_fallback",
                {"disable_shared_experts_fusion": True},
            )
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    # -- the SM100 GDN prefill default ---------------------------------------

    def _runner(self, is_draft_worker: bool) -> ModelRunner:
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = is_draft_worker
        return runner

    def test_the_gdn_default_is_recorded_for_the_target_runner(self):
        self._seed(linear_attn_prefill_backend=None)
        record_gdn_prefill_default(self._runner(is_draft_worker=False), "flashinfer")
        self.assertEqual(get_exec().mamba.linear_attn_prefill_backend, "flashinfer")
        self.assertIn("gdn_backend.sm100_flashinfer_default", self._log_sources())

    def test_a_draft_runner_does_not_record_the_gdn_default(self):
        self._seed(linear_attn_prefill_backend=None)
        record_gdn_prefill_default(self._runner(is_draft_worker=True), "flashinfer")
        self.assertIsNone(get_exec().mamba.linear_attn_prefill_backend)
        self.assertNotIn("gdn_backend.sm100_flashinfer_default", self._log_sources())

    def test_no_default_records_nothing(self):
        self._seed(linear_attn_prefill_backend=None)
        record_gdn_prefill_default(self._runner(is_draft_worker=False), None)
        self.assertNotIn("gdn_backend.sm100_flashinfer_default", self._log_sources())

    # -- update_model_fields after a weight update ---------------------------

    def _update_fields(self, *, is_draft_worker: bool):
        runner = self._runner(is_draft_worker)
        runner.update_model_fields(
            object(),
            model_path="/new/checkpoint",
            load_format="auto",
            load_config=object(),
        )

    def test_a_target_weight_update_is_recorded(self):
        self._seed()
        self._update_fields(is_draft_worker=False)
        self.assertEqual(get_model().model_path, "/new/checkpoint")
        self.assertIn("model_runner.update_model_fields", self._log_sources())

    def test_a_draft_weight_update_keeps_the_targets_record(self):
        seeded = self._seed()
        self._update_fields(is_draft_worker=True)
        self.assertEqual(get_model().model_path, seeded.model_path)
        self.assertNotIn("model_runner.update_model_fields", self._log_sources())


if __name__ == "__main__":
    unittest.main()
