# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backend-selection and dispatch tests for AR-Diffusion paged self-attention.

Selection used to happen inside the per-layer forward: it read ``Tensor.is_cuda``
(``False`` on XPU, so every call went to the dense Python reference), probed
kernel availability, read an env var, and caught the kernel's own RuntimeError.
It now happens once, host-side, in ``resolve_ar_diffusion_attention_config``, and
the forward only executes the ``backend_id`` it is handed. The two halves are
tested separately:

Selection (host, no tensors involved):

1. CPU and management-only caches resolve to the reference backend.
2. An accelerator with a bound kernel resolves to that kernel; CUDA carries an
   ``fa_version``, XPU is FA2, and ROCm is identified before the CUDA branch.
3. A missing kernel raises during setup, and the opt-in env switch turns that
   into the reference backend instead.

Execution (forward, given a backend):

4. Each ``backend_id`` runs its own path, and the paged pools reach the kernel
   as-is (no dense gather).
5. Nothing is probed and no env var is read; kernel errors propagate rather than
   selecting a different backend behind the caller's back.

All CPU/mock based, so no accelerator is needed: ``torch.version.hip`` is pinned
to ``None`` so the host build cannot pick the ROCm branch, and vLLM's ``fa_utils``
entry points are stubbed. Note that the forward no longer inspects
``query.device``, so these tests need no fake-device tensor subclass.
"""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention as pa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5

# fa_utils only binds flash_attn_varlen_func on CUDA/XPU/ROCm, so on a CPU
# runner the attribute does not exist and patch() must be allowed to create it.
_FA_FUNC = "vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func"
_FA_AVAILABLE = "vllm.v1.attention.backends.fa_utils.is_flash_attn_varlen_func_available"

_real_import = builtins.__import__


def _make_paged_inputs(kv_len=BLOCK, q_len=BLOCK):
    """Minimal (query, KV pools, block-table metadata) for one attention call."""
    num_blocks = (kv_len + BLOCK - 1) // BLOCK + 1  # +1 spare (padding) block
    query = torch.randn(q_len, N_HEADS, HEAD_DIM)
    key_cache = torch.randn(num_blocks, BLOCK, N_HEADS, HEAD_DIM)
    value_cache = torch.randn_like(key_cache)
    n_used = (kv_len + BLOCK - 1) // BLOCK
    used = torch.arange(n_used, dtype=torch.int32).flip(0)
    pad = torch.full((num_blocks - n_used,), num_blocks - 1, dtype=torch.int32)
    block_table = torch.cat([used, pad]).view(1, num_blocks)
    query_start_loc = torch.tensor([0, q_len], dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32)
    return query, key_cache, value_cache, block_table, query_start_loc, seq_lens


def _call(query, key_cache, value_cache, block_table, query_start_loc, seq_lens, *, backend_id, fa_version=0):
    return pa.ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        backend_id=backend_id,
        fa_version=fa_version,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=BLOCK,
        max_seq_len=BLOCK,
        softmax_scale=SCALE,
    )


def _resolve(device_type, *, allow_reference=False, head_size=HEAD_DIM):
    device = torch.device(device_type) if device_type is not None else None
    return pa.resolve_ar_diffusion_attention_config(
        device=device,
        head_size=head_size,
        allow_reference=allow_reference,
    )


# ── Selection ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_type", ["cpu", None])
def test_devices_without_a_paged_kernel_resolve_to_reference(device_type):
    """(1) CPU and management-only caches (``device=None``) need no accelerator probe."""
    with patch(_FA_AVAILABLE, return_value=True) as available_mock:
        config = _resolve(device_type)

    assert config.backend_id == pa.BACKEND_REFERENCE
    assert config.backend_name == "reference"
    assert "no paged kernel" in config.reason
    # Decided before any accelerator extension is consulted.
    available_mock.assert_not_called()


def test_accelerator_with_bound_kernel_resolves_to_that_kernel():
    """(2) XPU is FA2; CUDA resolves a version from the head size."""
    with (
        patch("torch.version.hip", None),
        patch(_FA_AVAILABLE, return_value=True),
        patch.object(pa, "_resolve_fa_version", return_value=3) as version_mock,
    ):
        xpu = _resolve("xpu")
        assert version_mock.call_count == 0, "fa_version is a CUDA concern"
        cuda = _resolve("cuda")
        assert version_mock.call_count == 1

    assert (xpu.backend_id, xpu.fa_version, xpu.backend_name) == (pa.BACKEND_XPU, 2, "xpu")
    assert (cuda.backend_id, cuda.fa_version, cuda.backend_name) == (pa.BACKEND_CUDA, 3, "cuda")
    assert xpu.reason is None and cuda.reason is None


def test_rocm_is_identified_before_the_cuda_branch():
    """(2) HIP tensors report device.type == 'cuda', so hip must be checked first."""
    with (
        patch("torch.version.hip", "6.0.0"),
        patch(_FA_AVAILABLE, return_value=True) as available_mock,
    ):
        config = _resolve("cuda")

    assert config.backend_id == pa.BACKEND_ROCM
    assert config.backend_name == "rocm"
    # ROCm resolves its own entry point in the helper, not through fa_utils.
    available_mock.assert_not_called()


@pytest.mark.parametrize("device_type", ["xpu", "cuda"])
def test_missing_kernel_raises_during_setup(device_type):
    """(3) No silent reference fallback on a device that should have a kernel."""
    with (
        patch("torch.version.hip", None),
        patch(_FA_AVAILABLE, return_value=False),
    ):
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            _resolve(device_type, allow_reference=False)


@pytest.mark.parametrize("device_type", ["xpu", "cuda"])
def test_missing_kernel_with_opt_in_resolves_to_reference(device_type):
    """(3) The opt-in turns the setup failure into the reference backend."""
    with (
        patch("torch.version.hip", None),
        patch(_FA_AVAILABLE, return_value=False),
    ):
        config = _resolve(device_type, allow_reference=True)

    assert config.backend_id == pa.BACKEND_REFERENCE
    assert "no paged FlashAttention varlen entry point is bound" in config.reason


def test_unimportable_fa_utils_is_reported_as_nothing_bound():
    """(3) On XPU fa_utils pulls in vllm_xpu_kernels; a missing one is not a crash."""

    def _import_without_fa_utils(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and "fa_utils" in fromlist:
            raise ImportError("mock: no module named 'vllm_xpu_kernels'")
        return _real_import(name, globals, locals, fromlist, level)

    with patch("torch.version.hip", None), patch.object(builtins, "__import__", _import_without_fa_utils):
        with pytest.raises(RuntimeError, match="cannot be imported"):
            _resolve("xpu", allow_reference=False)
        config = _resolve("xpu", allow_reference=True)

    assert config.backend_id == pa.BACKEND_REFERENCE
    assert "cannot be imported" in config.reason


def test_env_switch_allows_but_does_not_force_reference(monkeypatch):
    """The env var only feeds ``allow_reference``; a bound kernel still wins."""
    monkeypatch.delenv(pa._ALLOW_REFERENCE_ATTN_ENV, raising=False)
    assert pa._reference_attn_allowed() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(pa._ALLOW_REFERENCE_ATTN_ENV, value)
        assert pa._reference_attn_allowed() is True
    monkeypatch.setenv(pa._ALLOW_REFERENCE_ATTN_ENV, "0")
    assert pa._reference_attn_allowed() is False

    monkeypatch.setenv(pa._ALLOW_REFERENCE_ATTN_ENV, "1")
    with patch("torch.version.hip", None), patch(_FA_AVAILABLE, return_value=True):
        config = _resolve("xpu", allow_reference=pa._reference_attn_allowed())
    assert config.backend_id == pa.BACKEND_XPU, "the switch must not force the reference path"


def test_config_is_immutable_and_per_instance():
    """No module-level 'last backend' state: each resolution is its own frozen value."""
    with patch("torch.version.hip", None), patch(_FA_AVAILABLE, return_value=True):
        first = _resolve("xpu")
        second = _resolve("cpu")

    assert first.backend_id == pa.BACKEND_XPU
    assert second.backend_id == pa.BACKEND_REFERENCE, "resolving again must not mutate the first"
    with pytest.raises(FrozenInstanceError):
        first.backend_id = pa.BACKEND_CUDA
    # The old design cached the last backend on the module, which two caches on
    # two devices would race over. Nothing may reintroduce that.
    assert not hasattr(pa, "ar_diffusion_paged_attention_backend")


# ── Execution ───────────────────────────────────────────────────────────────


def test_reference_backend_runs_the_dense_reference():
    """(4) One reference branch, shared by CPU and the diagnostic opt-in."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    with patch(_FA_FUNC, create=True) as kernel_mock:
        out = _call(q, kc, vc, bt, qsl, sl, backend_id=pa.BACKEND_REFERENCE)

    assert out.shape == q.shape
    kernel_mock.assert_not_called()


@pytest.mark.parametrize("backend_id", [pa.BACKEND_CUDA, pa.BACKEND_XPU])
def test_native_backend_forwards_paged_inputs_unchanged(backend_id):
    """(4) The kernel receives the block table and pools as-is -- no dense gather."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    seen: dict[str, object] = {}

    def fake_fa(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    with (
        patch(_FA_FUNC, side_effect=fake_fa, create=True),
        patch.object(pa, "_reference_paged_attention") as ref_mock,
    ):
        out = _call(q, kc, vc, bt, qsl, sl, backend_id=backend_id, fa_version=2)

    ref_mock.assert_not_called()
    assert out.shape == q.shape
    assert seen["block_table"] is bt
    assert seen["k"] is kc
    assert seen["v"] is vc
    assert seen["seqused_k"] is sl
    assert seen["fa_version"] == 2, "the resolved version is passed through, not recomputed"


def test_returned_tensor_wins_over_the_out_buffer():
    """The XPU wrapper returns its own tensor and may leave ``out`` untouched."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    owned = torch.full((q.shape[0], N_HEADS, HEAD_DIM), 7.0)

    with patch(_FA_FUNC, return_value=owned, create=True):
        out = _call(q, kc, vc, bt, qsl, sl, backend_id=pa.BACKEND_XPU, fa_version=2)
    assert torch.equal(out, owned)

    # A tuple return (CUDA) is unwrapped to its first element.
    with patch(_FA_FUNC, return_value=(owned, None), create=True):
        out = _call(q, kc, vc, bt, qsl, sl, backend_id=pa.BACKEND_CUDA, fa_version=3)
    assert torch.equal(out, owned)


def test_batched_query_keeps_its_shape():
    """A (B, L, H, D) query is flattened for the kernel and restored on the way out."""
    _, kc, vc, bt, qsl, sl = _make_paged_inputs()
    query = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)

    with patch(_FA_FUNC, side_effect=lambda **kw: torch.zeros_like(kw["q"]), create=True) as kernel_mock:
        out = _call(query, kc, vc, bt, qsl, sl, backend_id=pa.BACKEND_CUDA, fa_version=3)

    assert kernel_mock.call_args.kwargs["q"].shape == (BLOCK, N_HEADS, HEAD_DIM)
    assert out.shape == query.shape


@pytest.mark.parametrize("backend_id", [pa.BACKEND_CUDA, pa.BACKEND_XPU])
def test_kernel_errors_propagate(backend_id):
    """(5) Including a page-size refusal: no branch absorbs it into the reference.

    ``is_flash_attn_varlen_func_available()`` answers "is an entry point bound",
    not "can it service this geometry", so an unsupported page size can only
    surface from the kernel. Selection has already happened by then, and turning
    that error into a different backend mid-forward is what this replaces.
    """
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()

    def refusing_fa(**kwargs):
        raise RuntimeError(f"chunk_prefill: unsupported block_size={kc.shape[1]}")

    with (
        patch(_FA_FUNC, side_effect=refusing_fa, create=True),
        patch.object(pa, "_reference_paged_attention") as ref_mock,
    ):
        with pytest.raises(RuntimeError, match="unsupported block_size"):
            _call(q, kc, vc, bt, qsl, sl, backend_id=backend_id, fa_version=2)

    ref_mock.assert_not_called()


def test_forward_never_probes_capability_or_reads_the_environment(monkeypatch):
    """(5) The whole point of the refactor: no per-call setup work."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.setenv(pa._ALLOW_REFERENCE_ATTN_ENV, "1")
    allowed_mock = MagicMock(return_value=True)
    version_mock = MagicMock(return_value=3)

    with (
        patch(_FA_FUNC, side_effect=lambda **kw: torch.zeros_like(kw["q"]), create=True),
        patch(_FA_AVAILABLE, return_value=True) as available_mock,
        patch.object(pa, "_reference_attn_allowed", allowed_mock),
        patch.object(pa, "_resolve_fa_version", version_mock),
    ):
        _call(q, kc, vc, bt, qsl, sl, backend_id=pa.BACKEND_CUDA, fa_version=3)

    available_mock.assert_not_called()
    allowed_mock.assert_not_called()
    version_mock.assert_not_called()


def test_unknown_backend_is_rejected():
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    with pytest.raises(ValueError, match="Unknown attention backend"):
        _call(q, kc, vc, bt, qsl, sl, backend_id=99)
