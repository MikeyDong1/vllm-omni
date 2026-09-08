# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session end-of-life routing (RFC #4480, adapting to #5271).

The AR-Diffusion runner raises an explicit end-of-session signal on reset,
close, eviction, and failed forwards (``reset_ar_diffusion_session`` /
``close_ar_diffusion_session``). On the bespoke path those pop ``_states``; on
the opt-in manager path the session lives in the manager instead, so the hooks
must release it there or every closed session leaks its buffers.

All CPU, tiny tensors, no model.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero as pipeline_module
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
    DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION,
    MAX_DREAMZERO_SESSIONS,
    MAX_RESIDENT_DREAMZERO_SESSION_STATES,
    DreamZeroPipeline,
)
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.experimental.world_models.adapters.state_dreamzero_adapter import (
    DreamZeroStateAdapter,
)
from vllm_omni.experimental.world_models.session_state import (
    LatentBuffer,
    SessionStateManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -- SessionStateManager.drop_session ---------------------------------------


def test_drop_session_absent_returns_false() -> None:
    manager = SessionStateManager()
    assert manager.drop_session("missing") is False


def test_drop_session_removes_and_frees() -> None:
    manager = SessionStateManager()
    session = manager.get_or_create_session("s")
    buffer: LatentBuffer[torch.Tensor] = LatentBuffer()
    buffer.allocate(maxlen=None)
    buffer.append(torch.zeros(256, dtype=torch.float32))
    session.put("payload", buffer)
    assert manager.nbytes_by_device().get("cpu", 0) >= 1024

    assert manager.drop_session("s") is True
    assert "s" not in manager
    assert len(manager) == 0
    # The session is gone from the table and its buffers were reset, so the
    # manager reports no held bytes.
    assert manager.nbytes_by_device() == {}


def test_drop_session_recreates_fresh() -> None:
    manager = SessionStateManager()
    manager.get_or_create_session("s").attrs["k"] = 1
    manager.drop_session("s")
    assert manager.get_or_create_session("s").attrs == {}


# -- SessionStateManager.raise_max_sessions ----------------------------------


def test_raise_max_sessions_lifts_the_bound_and_stops_evicting() -> None:
    manager = SessionStateManager(max_sessions=2)

    assert manager.raise_max_sessions(4) is True
    assert manager.max_sessions == 4

    for index in range(4):
        manager.get_or_create_session(f"s{index}")
    assert len(manager) == 4
    assert manager.evictions == 0


def test_raise_max_sessions_refuses_to_lower() -> None:
    """Shrinking a live bound would silently strand accumulated history: the
    overflow path drops the table entry without resetting buffers."""
    manager = SessionStateManager(max_sessions=4)

    assert manager.raise_max_sessions(2) is False
    assert manager.raise_max_sessions(4) is False
    assert manager.max_sessions == 4


@pytest.mark.parametrize("capacity", [0, -1])
def test_raise_max_sessions_rejects_non_positive(capacity: int) -> None:
    manager = SessionStateManager(max_sessions=2)

    with pytest.raises(ValueError, match="max_sessions must be positive"):
        manager.raise_max_sessions(capacity)

    assert manager.max_sessions == 2


# -- pipeline hook routing ---------------------------------------------------


def _manager_pipe(manager: SessionStateManager, session_id: str) -> DreamZeroPipeline:
    """A pipeline built via __new__ whose state is a manager-backed adapter."""
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = manager
    pipe.state = DreamZeroStateAdapter(session_id, manager, vae_encoder_window=2)
    return pipe


@pytest.mark.parametrize("hook", ["close_ar_diffusion_session", "reset_ar_diffusion_session"])
def test_hook_releases_manager_session_and_clears_alias(hook: str) -> None:
    manager = SessionStateManager()
    pipe = _manager_pipe(manager, "sess")
    assert "sess" in manager

    getattr(DreamZeroPipeline, hook)(pipe, "sess")

    assert "sess" not in manager
    # The alias viewed the released session, so it is dropped and rebuilt lazily.
    assert pipe.state is None


def test_hook_keeps_alias_for_a_different_session() -> None:
    manager = SessionStateManager()
    pipe = _manager_pipe(manager, "default")
    manager.get_or_create_session("other")

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "other")

    assert "other" not in manager
    # The alias views "default", not the closed session, so it survives.
    assert isinstance(pipe.state, DreamZeroStateAdapter)
    assert pipe.state.session_id == "default"


def test_bespoke_hook_pops_states_and_clears_alias() -> None:
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = None
    state = DreamZeroState()
    pipe._states["sess"] = state
    pipe.state = state

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "sess")

    assert "sess" not in pipe._states
    assert pipe.state is None


# -- bounded ``_states`` ------------------------------------------------------
#
# The hooks above are the *only* removal path for ``_states``, and they are
# reached solely from the AR-Diffusion runner's release path. A stage that holds
# session state without hosting the engine -- the disaggregated encode stage,
# where ``engine_backend: ARDiffusionEngine`` is set on denoise only -- never
# receives that signal, so every finished session used to stay resident at
# ~603 MiB each. These cover the backstop bound that makes the pipeline
# memory-safe on its own.


def _bespoke_pipe(max_states: int) -> DreamZeroPipeline:
    """A pipeline via __new__ on the bespoke path with a known state bound."""
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = None
    pipe._max_session_states = max_states
    pipe._session_state_evict_warned = False
    pipe.state = None
    return pipe


def _seeded_state(marker: int) -> DreamZeroState:
    """A state carrying a device-side tensor and a call history to lose."""
    state = DreamZeroState()
    state.vae_encoder_out = torch.zeros(marker + 1, dtype=torch.float32)
    state.call_count = marker + 1
    return state


def test_insert_over_bound_evicts_oldest_and_keeps_newest() -> None:
    pipe = _bespoke_pipe(2)

    first = DreamZeroPipeline._get_or_create_state(pipe, "s0")
    second = DreamZeroPipeline._get_or_create_state(pipe, "s1")
    third = DreamZeroPipeline._get_or_create_state(pipe, "s2")

    assert list(pipe._states) == ["s1", "s2"]
    assert pipe._states["s1"] is second
    assert pipe._states["s2"] is third
    # The evicted session is gone, so it comes back as a distinct object.
    assert DreamZeroPipeline._get_or_create_state(pipe, "s0") is not first


def test_reuse_reorders_so_the_active_session_is_never_the_victim() -> None:
    """``move_to_end`` on a hit is what makes the bound an LRU and not a FIFO."""
    pipe = _bespoke_pipe(2)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    # Touch the oldest so the *other* entry becomes the eviction candidate.
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s2")

    assert list(pipe._states) == ["s0", "s2"]


def test_eviction_releases_the_evicted_state_buffers() -> None:
    """Popping the entry is not enough: GC is too late to free device memory."""
    pipe = _bespoke_pipe(1)
    doomed = _seeded_state(0)
    pipe._states["s0"] = doomed

    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert "s0" not in pipe._states
    # We still hold a reference, so this proves ``reset()`` ran rather than the
    # object merely becoming unreachable.
    assert doomed.vae_encoder_out is None
    assert doomed.call_count == 0


def test_eviction_clears_the_alias_when_it_viewed_the_evicted_state() -> None:
    pipe = _bespoke_pipe(1)
    doomed = _seeded_state(0)
    pipe._states["s0"] = doomed
    pipe.state = doomed

    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert pipe.state is None


def test_eviction_keeps_an_alias_pointing_at_a_survivor() -> None:
    pipe = _bespoke_pipe(2)
    pipe._states["s0"] = _seeded_state(0)
    survivor = DreamZeroPipeline._get_or_create_state(pipe, "s1")
    pipe.state = survivor

    DreamZeroPipeline._get_or_create_state(pipe, "s2")

    assert "s0" not in pipe._states
    assert pipe.state is survivor


def test_eviction_warns_once_naming_the_missing_close(monkeypatch) -> None:
    """Hitting the bound means a close was lost; say so, but only once."""
    messages: list[str] = []

    def _capture(msg, *args, **kwargs):
        messages.append(msg % args if args else msg)

    monkeypatch.setattr(pipeline_module.logger, "warning", _capture)

    pipe = _bespoke_pipe(1)
    for index in range(4):
        DreamZeroPipeline._get_or_create_state(pipe, f"s{index}")

    evict_warnings = [m for m in messages if "close_ar_diffusion_session()" in m]
    assert len(evict_warnings) == 1
    assert pipe._session_state_evict_warned is True
    assert len(pipe._states) == 1


def test_non_positive_bound_means_unbounded_not_evict_everything() -> None:
    pipe = _bespoke_pipe(0)

    for index in range(3):
        DreamZeroPipeline._get_or_create_state(pipe, f"s{index}")

    assert len(pipe._states) == 3


# -- capacity published by the runner ----------------------------------------


def test_published_capacity_raises_the_floor_and_suppresses_eviction() -> None:
    """A runner already reserves 603 MiB/session and evicts at its own bound,
    signalling us each time, so we must not drop state it still holds live."""
    pipe = _bespoke_pipe(2)

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 5)

    assert pipe._max_session_states == 5
    for index in range(5):
        DreamZeroPipeline._get_or_create_state(pipe, f"s{index}")
    assert len(pipe._states) == 5


def test_published_capacity_reaches_the_manager_too() -> None:
    """Regression: the manager is built in ``__init__`` with the configured cap,
    so raising only ``_max_session_states`` would leave it evicting at the lower
    bound -- and its overflow drops the entry without resetting buffers, so the
    next request would silently get a fresh session and lose its history."""
    manager = SessionStateManager(max_sessions=2)
    pipe = _bespoke_pipe(2)
    pipe._memory_manager = manager

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 5)

    assert pipe._max_session_states == 5
    assert manager.max_sessions == 5

    for index in range(5):
        manager.get_or_create_session(f"s{index}")
    assert len(manager) == 5
    assert manager.evictions == 0


def test_published_capacity_never_lowers_the_bound() -> None:
    pipe = _bespoke_pipe(4)
    manager = SessionStateManager(max_sessions=4)
    pipe._memory_manager = manager

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 1)

    assert pipe._max_session_states == 4
    assert manager.max_sessions == 4


@pytest.mark.parametrize("capacity", [0, -1])
def test_published_capacity_ignores_non_positive(capacity: int) -> None:
    pipe = _bespoke_pipe(4)

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, capacity)

    assert pipe._max_session_states == 4


# -- the bound has to be able to fire ----------------------------------------


def test_resident_state_bound_is_not_the_kv_slot_count() -> None:
    """Regression guard for the trap this fix exists to close.

    ``MAX_DREAMZERO_SESSIONS`` counts KV slots. Reusing it for model-owned state
    puts the bound at 64 x 603 MiB = 38.6 GiB -- larger than any single device --
    so it can never fire before the card OOMs, and a run under it looks exactly
    like a run with no bound at all.
    """
    assert MAX_RESIDENT_DREAMZERO_SESSION_STATES < MAX_DREAMZERO_SESSIONS
    resident_bytes = MAX_RESIDENT_DREAMZERO_SESSION_STATES * DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION
    assert resident_bytes < 8 * 1024**3


def test_manager_honours_the_same_small_bound() -> None:
    """The two stores must be sized together.

    ``_drop_ar_diffusion_session_state()`` returns early on the manager path
    without touching ``_states``, so a bound enforced on only one of them is
    dead code for whichever path a deployment actually takes.
    """
    manager = SessionStateManager(max_sessions=MAX_RESIDENT_DREAMZERO_SESSION_STATES)

    for index in range(MAX_RESIDENT_DREAMZERO_SESSION_STATES + 2):
        manager.get_or_create_session(f"s{index}")

    assert len(manager) == MAX_RESIDENT_DREAMZERO_SESSION_STATES
    assert manager.evictions == 2
    assert "s0" not in manager
