"""Inkling's forward keys its FA4-only paths off the backend running THAT
forward, not the process config.

Target and draft coexist in one process; a draft configured with
``--speculative-draft-attention-backend`` runs a different backend than the
target, and its kernel selection must follow its own. The per-runner name is
stamped on the backend object at build time; the process config is only the
fallback for unstamped (v1 draft-factory) backends.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from sglang.srt.models.inkling_common.attn import active_attention_backend_str
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(mode: ForwardMode):
    return SimpleNamespace(forward_mode=mode)


def _backend(prefill=None, decode=None):
    return SimpleNamespace(
        prefill_attention_backend_str=prefill,
        decode_attention_backend_str=decode,
    )


class TestInklingBackendRead(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_the_stamped_backend_wins(self):
        self._seed(attention_backend="fa4")
        with forward_context(ForwardContext(attn_backend=_backend("triton", "triton"))):
            self.assertEqual(
                active_attention_backend_str(_batch(ForwardMode.DECODE)), "triton"
            )
            self.assertEqual(
                active_attention_backend_str(_batch(ForwardMode.EXTEND)), "triton"
            )

    def test_the_mode_picks_its_half_of_the_pair(self):
        self._seed(attention_backend="fa4")
        with forward_context(ForwardContext(attn_backend=_backend("fa4", "triton"))):
            self.assertEqual(
                active_attention_backend_str(_batch(ForwardMode.DECODE)), "triton"
            )
            self.assertEqual(
                active_attention_backend_str(_batch(ForwardMode.TARGET_VERIFY)), "fa4"
            )

    def test_an_unstamped_backend_falls_back_to_the_config(self):
        self._seed(attention_backend="fa4")
        with forward_context(ForwardContext(attn_backend=_backend())):
            self.assertEqual(
                active_attention_backend_str(_batch(ForwardMode.DECODE)), "fa4"
            )


if __name__ == "__main__":
    unittest.main()
