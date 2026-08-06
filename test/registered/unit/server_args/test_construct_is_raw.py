"""Construction parses; resolution is a separate, explicit step.

``ServerArgs(...)`` is the literal input record — no handler runs, no
declaration materializes, no device is probed. The pipeline runs once via
``resolve()``: at the CLI boundary (``from_cli_args``), in ``Engine``, and as
a safety net inside ``publish``. This split is what lets a config cross a
process boundary raw or resolved without re-running resolution, and keeps
variants (``derive``) and dummy fixtures from paying resolution they don't
want.
"""

import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The golden-override mini config: enough for ModelConfig to load without a
# checkpoint, host-independent with device="cuda".
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class TestConstructIsRaw(CustomTestCase):
    def tearDown(self):
        reset_context()

    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="raw_construct_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(_MINI_CONFIG, f)
        return config_dir

    def _construct(self, **kwargs) -> ServerArgs:
        kwargs.setdefault("device", "cuda")
        return ServerArgs(model_path=self._config_dir(), **kwargs)

    def test_a_bare_construct_is_the_literal_input(self):
        server_args = self._construct()
        self.assertFalse(getattr(server_args, "_declarations_materialized", False))
        # A resolution-computed default has not been filled in.
        self.assertIsNone(server_args.chunked_prefill_size)

    def test_resolve_materializes_once(self):
        server_args = self._construct().resolve()
        self.assertTrue(server_args._declarations_materialized)
        self.assertIsNotNone(server_args.chunked_prefill_size)
        # Idempotent: a second resolve is a no-op, not a re-resolution.
        snapshot = dict(vars(server_args))
        self.assertIs(server_args.resolve(), server_args)
        self.assertEqual(dict(vars(server_args)), snapshot)

    def test_a_dummy_config_never_resolves(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIs(server_args.resolve(), server_args)
        self.assertFalse(getattr(server_args, "_declarations_materialized", False))

    def test_publish_resolves_a_raw_config(self):
        server_args = self._construct()
        publish(server_args, role="test")
        self.assertTrue(server_args._declarations_materialized)
        # The bags were projected from the resolved values.
        self.assertEqual(
            get_schedule().chunked_prefill_size,
            server_args.chunked_prefill_size,
        )

    def test_publish_does_not_re_resolve(self):
        server_args = self._construct().resolve()
        snapshot = dict(vars(server_args))
        publish(server_args, role="test")
        self.assertEqual(dict(vars(server_args)), snapshot)

    def test_the_cli_boundary_hands_out_resolved_configs(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--model-path", self._config_dir(), "--device", "cuda"]
        )
        server_args = ServerArgs.from_cli_args(args)
        self.assertTrue(server_args._declarations_materialized)


if __name__ == "__main__":
    unittest.main()
