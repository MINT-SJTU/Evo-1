"""Device-agnostic behaviour of the XPU / CUDA / CPU plumbing.

Everything except the last test class runs on any host (CPU only, no checkpoint,
no network): detection order, dispatch behind the memory and synchronize helpers,
the flash-attn fallback, config device resolution, and the device handling of the
inference server. The last class compares a tiny random-weight action head on CPU
and on XPU and is skipped when no XPU is present.

Run from the repository root with ``python -m unittest discover -s tests -v``.
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Evo_1")
)

import device_utils as dev  # noqa: E402
from config import EvoConfig  # noqa: E402


def _patch_availability(*, xpu: bool, cuda: bool):
    """Make ``is_available`` report exactly the given backends."""

    def fake_is_available(device_type: str) -> bool:
        if device_type == "xpu":
            return xpu
        if device_type == "cuda":
            return cuda
        return device_type == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # missing optional dependency in a minimal CI environment
        return None


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.available_device_types(), ["xpu", "cuda", "cpu"])

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_detected_device_has_no_index(self):
        # A bare "xpu" or "cuda" follows the current device chosen by the launcher.
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.resolve_device(None), torch.device("xpu"))
            self.assertIsNone(dev.resolve_device(None).index)

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(
                dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1")
            )

    def test_device_type_from_tensor_string_and_device(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda")), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")

    def test_device_count_follows_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "device_count", return_value=2):
                self.assertEqual(dev.device_count(), 2)
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "device_count", return_value=8):
                self.assertEqual(dev.device_count(), 8)
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.device_count(), 0)


class AcceleratorHelperTest(unittest.TestCase):
    def test_cpu_helpers_are_noops(self):
        with _patch_availability(xpu=False, cuda=False):
            dev.synchronize()
            dev.empty_cache("cpu")
            dev.reset_peak_memory_stats()
            self.assertEqual(dev.memory_allocated(), 0)
            self.assertEqual(dev.max_memory_allocated("cpu"), 0)
            self.assertEqual(dev.memory_reserved(), 0)
            self.assertEqual(dev.max_memory_reserved(), 0)

    def test_missing_cuda_build_does_not_raise(self):
        # torch.cuda.synchronize() and torch.cuda.memory_allocated() raise on a wheel
        # without CUDA; the helpers must check availability first.
        with _patch_availability(xpu=False, cuda=False):
            with mock.patch.object(
                torch.cuda, "synchronize", side_effect=AssertionError("no cuda")
            ):
                with mock.patch.object(
                    torch.cuda,
                    "memory_allocated",
                    side_effect=AssertionError("no cuda"),
                ):
                    dev.synchronize("cuda")
                    self.assertEqual(dev.memory_allocated("cuda"), 0)

    def test_dispatch_goes_to_the_requested_backend(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(
                    torch.xpu, "memory_allocated", return_value=11
                ) as xpu_mem,
                mock.patch.object(
                    torch.cuda, "memory_allocated", return_value=22
                ) as cuda_mem,
                mock.patch.object(torch.xpu, "synchronize") as xpu_sync,
                mock.patch.object(torch.cuda, "synchronize") as cuda_sync,
            ):
                self.assertEqual(dev.memory_allocated("xpu:0"), 11)
                self.assertEqual(dev.memory_allocated(torch.device("cuda")), 22)
                self.assertEqual(dev.memory_allocated(), 11)  # detection picks xpu
                dev.synchronize("xpu")
                xpu_sync.assert_called_once()
                cuda_sync.assert_not_called()
                self.assertEqual(xpu_mem.call_count, 2)
                self.assertEqual(cuda_mem.call_count, 1)

    def test_reset_peak_memory_stats_dispatch(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(torch.xpu, "reset_peak_memory_stats") as xpu_reset,
                mock.patch.object(torch.cuda, "reset_peak_memory_stats") as cuda_reset,
            ):
                dev.reset_peak_memory_stats("cuda")
                cuda_reset.assert_called_once()
                xpu_reset.assert_not_called()


class FlashAttnFallbackTest(unittest.TestCase):
    def setUp(self):
        dev._WARNED.clear()

    def test_never_used_off_cuda_even_when_installed(self):
        with mock.patch.object(dev, "flash_attn_installed", return_value=True):
            self.assertFalse(dev.use_flash_attn("xpu"))
            self.assertFalse(dev.use_flash_attn(torch.device("xpu:1")))
            self.assertFalse(dev.use_flash_attn("cpu"))

    def test_used_on_cuda_when_installed(self):
        with mock.patch.object(dev, "flash_attn_installed", return_value=True):
            self.assertTrue(dev.use_flash_attn("cuda:0"))

    def test_cuda_without_package_falls_back(self):
        with mock.patch.object(dev, "flash_attn_installed", return_value=False):
            self.assertFalse(dev.use_flash_attn("cuda"))

    def test_fallback_is_logged_once(self):
        with mock.patch.object(dev, "flash_attn_installed", return_value=True):
            with self.assertLogs(dev.logger, level="WARNING") as captured:
                dev.use_flash_attn("xpu")
                dev.use_flash_attn("xpu")
        self.assertEqual(len(captured.records), 1)
        self.assertIn("xpu", captured.records[0].getMessage())


class ConfigDeviceTest(unittest.TestCase):
    def test_config_default_is_detection_not_cuda(self):
        self.assertIsNone(EvoConfig().device)

    def test_checkpoint_device_survives_round_trip(self):
        self.assertEqual(EvoConfig.from_dict({"device": "cuda"}).device, "cuda")

    def test_evo1_resolves_config_device(self):
        evo1 = _try_import("scripts.Evo1")
        if evo1 is None:
            self.skipTest("model dependencies (transformers, timm) are not installed")
        embedder = mock.MagicMock()
        head = mock.MagicMock()
        head.to.return_value = head
        with (
            _patch_availability(xpu=True, cuda=False),
            mock.patch.object(evo1, "InternVL3Embedder", embedder),
            mock.patch.object(evo1, "FlowmatchingActionHead", return_value=head),
        ):
            model = evo1.EVO1(EvoConfig(per_action_dim=2, horizon=2))
            self.assertEqual(model._device, torch.device("xpu"))
            self.assertEqual(embedder.call_args.kwargs["device"], torch.device("xpu"))
            model = evo1.EVO1(EvoConfig(device="cpu", per_action_dim=2, horizon=2))
            self.assertEqual(model._device, torch.device("cpu"))


class ServerDeviceTest(unittest.TestCase):
    """The websocket server must follow the model device instead of assuming CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.server = _try_import("scripts.Evo1_server")
        if cls.server is None:
            raise unittest.SkipTest(
                "server dependencies (websockets, cv2, fvcore, pandas) are not installed"
            )

    def test_decode_image_goes_to_requested_device(self):
        frame = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy().tolist()
        tensor = self.server.decode_image_from_list(frame, torch.device("cpu"))
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tuple(tensor.shape), (3, 448, 448))

    def test_inference_uses_model_device_for_autocast_and_inputs(self):
        seen = {}

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def run_inference(
                self, images, image_mask, prompt, state_input, action_mask
            ):
                seen["devices"] = {
                    t.device.type
                    for t in [*images, image_mask, state_input, action_mask]
                }
                return torch.zeros(1, 48)

        normalizer = types.SimpleNamespace(
            normalize_state=lambda state, arm, ds: state,
            denormalize_action=lambda action, arm, ds: action,
        )
        frame = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy().tolist()
        payload = {
            "image": [frame, frame, frame],
            "image_mask": [1, 1, 0],
            "state": [0.0] * 24,
            "action_mask": [[1] * 24],
            "prompt": "pick the cube",
        }
        real_autocast = torch.amp.autocast
        with mock.patch.object(
            self.server.torch.amp, "autocast", wraps=real_autocast
        ) as autocast:
            actions = self.server.infer_from_json_dict(
                payload, FakeModel(), normalizer, "arm", "ds"
            )
        self.assertEqual(autocast.call_args.kwargs["device_type"], "cpu")
        self.assertEqual(seen["devices"], {"cpu"})
        self.assertEqual(len(actions), 2)

    def test_load_model_overrides_checkpoint_device(self):
        import json
        import tempfile

        config = {"device": "cuda", "per_action_dim": 2, "horizon": 2}
        with tempfile.TemporaryDirectory() as ckpt:
            for name, payload in (("config.json", config), ("norm_stats.json", {})):
                with open(os.path.join(ckpt, name), "w") as handle:
                    json.dump(payload, handle)
            fake_model = mock.MagicMock()
            fake_model.eval.return_value = fake_model
            built = {}

            def build(cfg):
                built["device"] = cfg.device
                return fake_model

            with (
                mock.patch.object(self.server, "EVO1", side_effect=build),
                mock.patch.object(
                    self.server.torch, "load", return_value={"module": {}}
                ),
                mock.patch.object(self.server, "Normalizer"),
            ):
                self.server.load_model_and_normalizer(ckpt, device="cpu")
            self.assertEqual(built["device"], "cpu")
            fake_model.to.assert_called_once_with(torch.device("cpu"))


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU device"
)
class ActionHeadXPUParityTest(unittest.TestCase):
    """Tiny random-weight flow-matching head: CPU versus XPU with the same weights and noise."""

    def test_get_action_matches_cpu(self):
        from model.action_head.flow_matching import FlowmatchingActionHead

        config = EvoConfig(
            embed_dim=64,
            hidden_dim=64,
            state_hidden_dim=64,
            num_heads=4,
            num_layers=2,
            horizon=4,
            per_action_dim=6,
            state_dim=6,
            num_inference_timesteps=8,
        )
        torch.manual_seed(0)
        head = FlowmatchingActionHead(config=config).eval().to(torch.bfloat16)
        fused = torch.randn(1, 12, 64).to(torch.bfloat16)
        state = torch.randn(1, 6)
        mask = torch.ones(1, 4 * 6, dtype=torch.int32)
        noise = torch.rand(1, 24, generator=torch.Generator().manual_seed(1))

        def run(device):
            module = head.to(device)
            with mock.patch(
                "model.action_head.flow_matching.torch.rand",
                return_value=noise.to(device, torch.bfloat16),
            ):
                with torch.no_grad():
                    out = module.get_action(
                        fused.to(device),
                        state=state.to(device),
                        action_mask=mask.to(device),
                    )
            return out.float().cpu()

        cpu_out = run("cpu")
        xpu_out = run("xpu")
        self.assertEqual(cpu_out.shape, xpu_out.shape)
        self.assertTrue(torch.isfinite(xpu_out).all())
        torch.testing.assert_close(xpu_out, cpu_out, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
