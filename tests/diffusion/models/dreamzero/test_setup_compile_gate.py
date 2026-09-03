# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the accelerator gate on ``DreamZeroPipeline.setup_compile``.

The gate used to be ``torch.cuda.is_available()``, which left XPU (and ROCm, and
MUSA) in eager for no reason. It now asks the platform two questions instead, and
the *order* of those two questions is load-bearing:
``UnspecifiedOmniPlatform`` -- what resolves when no accelerator is detected --
overrides ``get_device_count()`` to return 0 but does **not** implement
``supports_torch_inductor()``, so asking about inductor support first turns a
CPU-only run from a clean skip into a ``NotImplementedError``.

These cover the three outcomes and pin that ordering. All CPU: the gate is reached
before ``setup_compile`` touches any pipeline attribute, so a bare instance is
enough, and a sentinel on the first call past the gate shows whether we got through.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero as mod
from vllm_omni.diffusion.models.dreamzero import wan_vae_feat_cache_patch as wan_vae_mod
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ReachedCompileBodyError(Exception):
    """Raised from the first call past the gate, to prove the gate let us through."""


class _FakePlatform:
    """Minimal stand-in for the two OmniPlatform hooks the gate calls.

    ``inductor=None`` models ``UnspecifiedOmniPlatform`` faithfully: it raises
    ``NotImplementedError``, because that class inherits the base's unimplemented
    ``supports_torch_inductor()``. A test that reaches it has caught a regression in
    the short-circuit order.
    """

    def __init__(self, device_count: int, inductor: bool | None) -> None:
        self._device_count = device_count
        self._inductor = inductor

    def get_device_count(self) -> int:
        return self._device_count

    def supports_torch_inductor(self) -> bool:
        if self._inductor is None:
            raise NotImplementedError
        return self._inductor


def _run_setup_compile(monkeypatch, platform):
    """Call setup_compile against a faked platform, with a tripwire past the gate.

    ``setup_compile`` reads no ``self`` attribute before the gate, so an
    ``__new__``-built instance is enough (same approach as test_pipeline_state.py).
    Raises ``_ReachedCompileBodyError`` iff the gate did not skip.
    """
    monkeypatch.setattr(mod, "current_omni_platform", platform)

    def _boom() -> None:
        raise _ReachedCompileBodyError

    # First call past the gate. Patched on its defining module because
    # setup_compile imports it inside the function body.
    monkeypatch.setattr(wan_vae_mod, "apply_wan_vae_feat_cache_tensor_patch", _boom)

    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline.setup_compile()


def test_no_accelerator_skips_compile(monkeypatch):
    """No accelerator: skip cleanly, and never ask about inductor support.

    The _FakePlatform raises NotImplementedError from supports_torch_inductor, so
    this also pins the short-circuit order -- reversing the two guards turns this
    test from a pass into an error.
    """
    _run_setup_compile(monkeypatch, _FakePlatform(device_count=0, inductor=None))


def test_platform_without_inductor_support_skips_compile(monkeypatch):
    """Devices present but the platform declares no inductor support: skip."""
    _run_setup_compile(monkeypatch, _FakePlatform(device_count=8, inductor=False))


def test_accelerator_with_inductor_support_proceeds(monkeypatch):
    """Devices present and inductor supported: the gate must not skip.

    Device *type* is deliberately not part of the gate any more, so there is no
    separate CUDA and XPU case to write here -- one graph of behaviour serves both.
    """
    with pytest.raises(_ReachedCompileBodyError):
        _run_setup_compile(monkeypatch, _FakePlatform(device_count=8, inductor=True))
