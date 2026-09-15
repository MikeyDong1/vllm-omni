# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-stage session lifecycle for the DreamZero disaggregated topology.

Uses fake stage workers, but drives the real coordinator, the real release-event
log and the real worker lifecycle dispatch, so the ordering guarantees are
exercised rather than mocked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.ar_diffusion.release_events import (
    ARDiffusionReleaseEvent,
    ARDiffusionReleaseEventLog,
)
from vllm_omni.experimental.ar_diffusion.stage_lifecycle import (
    DiffusionStageLifecycleCoordinator,
    DiffusionStageLifecycleTopology,
    SessionControls,
    SessionLifecycleError,
    SessionNotLiveError,
    read_session_controls,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ENCODE, DENOISE, DECODE = 0, 1, 2


class FakeStageWorker:
    """One stage's state-owning worker: sessions plus a real release-event log."""

    def __init__(self, stage_id: int, *, state_owning: bool = True, tp_size: int = 1) -> None:
        self.stage_id = stage_id
        self.state_owning = state_owning
        self.tp_size = tp_size
        self.sessions: dict[str, int] = {}
        self.generations: dict[str, int] = {}
        self.log = ARDiffusionReleaseEventLog(stage_id=stage_id)
        self.log.set_ready()
        self.capacity: int | None = None
        self.calls: list[tuple[str, tuple]] = []
        self.fail_close: set[str] = set()

    # -- model-side behavior ------------------------------------------------

    def touch_session(self, session_id: str) -> None:
        """Begin or continue a session, evicting LRU like the AR runner does."""
        if session_id not in self.sessions and self.capacity is not None:
            while len(self.sessions) >= self.capacity:
                victim = next(iter(self.sessions))
                self.release(victim, reason="lru_eviction")
        self.sessions[session_id] = self.sessions.get(session_id, 0) + 1

    def release(self, session_id: str, *, reason: str) -> None:
        self.sessions.pop(session_id, None)
        self.log.record(session_id, reason=reason)

    # -- RPC surface --------------------------------------------------------

    def _dispatch(self, method: str, args: tuple):
        self.calls.append((method, args))
        if method == "register_ar_diffusion_generation":
            session_id, generation = args
            self.generations[session_id] = generation
            self.log.register_generation(session_id, generation)
            return True
        if method in ("close_ar_diffusion_session", "reset_ar_diffusion_session"):
            (session_id,) = args
            if session_id in self.fail_close:
                raise RuntimeError(f"stage {self.stage_id} cannot clean up {session_id}")
            if not self.state_owning:
                # Stateless postprocess: idempotent no-op, no state created.
                return True
            with self.log.coordinated(session_id):
                self.sessions.pop(session_id, None)
            return True
        if method == "get_ar_diffusion_release_events":
            return self.log.pending()
        if method == "ack_ar_diffusion_release_events":
            (event_ids,) = args
            # The real worker returns how many records it dropped, not a flag.
            return self.log.acknowledge(event_ids)
        raise AssertionError(f"unexpected lifecycle RPC {method}")

    async def collective_rpc(self, method: str, args: tuple):
        """Mimic a stage pool: one result per TP rank."""
        results = []
        for _ in range(self.tp_size):
            try:
                results.append(self._dispatch(method, args))
            except Exception as exc:  # noqa: BLE001 - stage pools return, not raise
                results.append({"supported": False, "error": str(exc)})
        return results


class FakeTopologyRuntime:
    """The stage pools a coordinator talks to."""

    def __init__(self, workers: dict[int, FakeStageWorker]) -> None:
        self.workers = workers

    async def rpc(self, method: str, stage_id: int, args: tuple):
        # One physical replica per stage, matching the supported layout.
        return [await self.workers[stage_id].collective_rpc(method, args)]


def _coordinator(workers: dict[int, FakeStageWorker], **kwargs) -> DiffusionStageLifecycleCoordinator:
    state_owning = tuple(stage_id for stage_id, worker in sorted(workers.items()) if worker.state_owning)
    topology = DiffusionStageLifecycleTopology(
        stage_ids=tuple(sorted(workers)),
        state_owning_stage_ids=state_owning,
    )
    return DiffusionStageLifecycleCoordinator(topology, FakeTopologyRuntime(workers).rpc, **kwargs)


def _edd_workers(*, capacity: int | None = None, tp_size: int = 1) -> dict[int, FakeStageWorker]:
    workers = {
        ENCODE: FakeStageWorker(ENCODE),
        DENOISE: FakeStageWorker(DENOISE, tp_size=tp_size),
        DECODE: FakeStageWorker(DECODE, state_owning=False),
    }
    workers[DENOISE].capacity = capacity
    return workers


async def _run_request(
    coordinator: DiffusionStageLifecycleCoordinator,
    workers: dict[int, FakeStageWorker],
    request_id: str,
    session_id: str,
    *,
    reset: bool = False,
    close_session: bool = False,
    success: bool = True,
) -> None:
    """Admit, execute on the state-owning stages, then complete."""
    await coordinator.admit(request_id, SessionControls(session_id, reset=reset, close_session=close_session))
    if success:
        for stage_id in coordinator.topology.state_owning_stage_ids:
            workers[stage_id].touch_session(session_id)
    await coordinator.complete(request_id, success=success)


# -- topology ---------------------------------------------------------------


def test_topology_is_opt_in_and_excludes_the_stateless_stage():
    stage_configs = [
        SimpleNamespace(stage_id=0, stage_role="encode", model_stage="encode", coordinated_session_lifecycle=True),
        SimpleNamespace(stage_id=1, stage_role="denoise", model_stage="denoise", coordinated_session_lifecycle=True),
        SimpleNamespace(stage_id=2, stage_role="decode", model_stage="decode", coordinated_session_lifecycle=True),
    ]

    topology = DiffusionStageLifecycleTopology.from_stage_configs(stage_configs)

    assert topology.stage_ids == (0, 1, 2)
    # The trailing postprocess stage takes part in RPC but owns no state.
    assert topology.state_owning_stage_ids == (0, 1)


def test_topology_is_none_for_pipelines_that_did_not_opt_in():
    stage_configs = [SimpleNamespace(stage_id=0, stage_role="full", model_stage="diffusion")]

    assert DiffusionStageLifecycleTopology.from_stage_configs(stage_configs) is None
    assert DiffusionStageLifecycleTopology.from_stage_configs([]) is None


# -- controls ---------------------------------------------------------------


def test_read_session_controls_prefers_the_typed_tick():
    from vllm_omni.experimental.ar_diffusion.tick_protocol import AR_DIFFUSION_TICK_KEY

    params = SimpleNamespace(
        extra_args={
            "session_id": "flat",
            "reset": True,
            "close_session": False,
            AR_DIFFUSION_TICK_KEY: {
                "session_id": "typed",
                "request_id": "r0",
                "chunk_index": 0,
                "reset": False,
                "close_session": True,
            },
        }
    )

    controls = read_session_controls([params])

    assert controls == SessionControls("typed", reset=False, close_session=True)


def test_read_session_controls_returns_none_without_controls():
    assert read_session_controls([SimpleNamespace(extra_args={})]) is None
    assert read_session_controls([]) is None


# -- ordering ---------------------------------------------------------------


def test_begin_registers_one_generation_on_every_participant():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))

    generation = coordinator.generation_of("A")
    assert generation == 1
    assert all(worker.generations["A"] == generation for worker in workers.values())
    # The stateless postprocess stage recorded identity but no session state.
    assert workers[DECODE].sessions == {}


def test_continuation_of_a_session_with_no_state_is_rejected_before_mutation():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    with pytest.raises(SessionNotLiveError, match="explicit reset"):
        asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=False))

    assert workers[ENCODE].sessions == {}
    assert workers[DENOISE].sessions == {}
    # The rejection released the admission slot, so the next request is served.
    asyncio.run(_run_request(coordinator, workers, "r1", "A", reset=True))
    assert coordinator.is_active("A")


def test_reset_is_coordinated_once_and_not_repeated_downstream():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    first_generation = coordinator.generation_of("A")
    asyncio.run(_run_request(coordinator, workers, "r1", "A", reset=True))

    # A fresh generation, and the cleanup was fanned out exactly once per
    # state-owning stage rather than once per stage per request.
    assert coordinator.generation_of("A") == first_generation + 1
    for stage_id in (ENCODE, DENOISE):
        closes = [call for call in workers[stage_id].calls if call[0] == "close_ar_diffusion_session"]
        assert len(closes) == 1
    # The coordinator's own cleanup produced no release event to replay.
    assert workers[DENOISE].log.pending() == []


def test_denoise_eviction_of_a_retires_the_victim_on_the_encoder():
    """The reviewer's case: evicting A while running B must not leave encode
    holding A's VAE history."""
    workers = _edd_workers(capacity=2)
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    assert set(workers[ENCODE].sessions) == {"A", "B"}

    # C begins; the denoise runner is at capacity and evicts A on its own.
    asyncio.run(_run_request(coordinator, workers, "r2", "C", reset=True))

    assert "A" not in workers[DENOISE].sessions
    # The eviction was replayed onto the peer stage.
    assert "A" not in workers[ENCODE].sessions
    assert set(workers[ENCODE].sessions) == {"B", "C"}
    assert not coordinator.is_active("A")
    assert coordinator.is_active("B") and coordinator.is_active("C")
    # Events were acknowledged only after the peer cleanup succeeded.
    assert workers[DENOISE].log.pending() == []


def test_continuing_an_evicted_session_fails_without_touching_healthy_ones():
    workers = _edd_workers(capacity=2)
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r2", "C", reset=True))
    healthy_before = (dict(workers[ENCODE].sessions), dict(workers[DENOISE].sessions))

    with pytest.raises(SessionNotLiveError):
        asyncio.run(_run_request(coordinator, workers, "r3", "A", reset=False))

    assert (workers[ENCODE].sessions, workers[DENOISE].sessions) == healthy_before

    # Recovery is an explicit new rollout, and it gets a fresh generation.
    old_generation = max(worker.generations["A"] for worker in workers.values())
    asyncio.run(_run_request(coordinator, workers, "r4", "A", reset=True))
    assert coordinator.generation_of("A") > old_generation


def test_a_failed_request_invalidates_its_generation_everywhere():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r2", "A", success=False))

    assert not coordinator.is_active("A")
    assert "A" not in workers[ENCODE].sessions
    assert "A" not in workers[DENOISE].sessions
    # An unrelated session survives.
    assert coordinator.is_active("B")
    assert "B" in workers[ENCODE].sessions


def test_close_session_is_acknowledged_after_the_final_postprocess():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "A", close_session=True))

    assert not coordinator.is_active("A")
    assert workers[ENCODE].sessions == {}
    assert workers[DENOISE].sessions == {}


def test_close_twice_and_reset_after_close_are_idempotent():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(coordinator.close("A"))
    asyncio.run(coordinator.close("A"))

    assert not coordinator.is_active("A")
    assert workers[ENCODE].sessions == {}

    asyncio.run(_run_request(coordinator, workers, "r1", "A", reset=True))
    assert coordinator.is_active("A")


def test_only_one_request_is_in_flight_across_the_topology():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    order: list[str] = []

    async def scenario():
        await coordinator.admit("r0", SessionControls("A", reset=True))
        order.append("r0-admitted")

        async def second():
            await coordinator.admit("r1", SessionControls("B", reset=True))
            order.append("r1-admitted")
            await coordinator.complete("r1", success=True)

        task = asyncio.create_task(second())
        # Give the second request every chance to slip in early.
        for _ in range(5):
            await asyncio.sleep(0)
        order.append("r0-still-alone")
        await coordinator.complete("r0", success=True)
        await task

    asyncio.run(scenario())

    assert order == ["r0-admitted", "r0-still-alone", "r1-admitted"]


def test_a_cleanup_failure_blocks_admission_instead_of_reporting_success():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    workers[ENCODE].fail_close.add("A")

    with pytest.raises(SessionLifecycleError):
        asyncio.run(_run_request(coordinator, workers, "r1", "A", success=False))

    assert coordinator.blocked_reason is not None
    with pytest.raises(SessionLifecycleError, match="blocked pending recovery"):
        asyncio.run(coordinator.admit("r2", SessionControls("B", reset=True)))

    workers[ENCODE].fail_close.clear()
    coordinator.clear_block()
    asyncio.run(_run_request(coordinator, workers, "r3", "B", reset=True))
    assert coordinator.is_active("B")


def test_unsupported_participant_is_a_failure_not_a_silent_success():
    workers = _edd_workers()

    class Unsupported(FakeStageWorker):
        def _dispatch(self, method, args):
            if method == "close_ar_diffusion_session":
                return {"supported": False, "error": "no lifecycle support"}
            return super()._dispatch(method, args)

    workers[ENCODE] = Unsupported(ENCODE)
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    with pytest.raises(SessionLifecycleError, match="no lifecycle support"):
        asyncio.run(_run_request(coordinator, workers, "r1", "A", success=False))


def test_release_events_from_every_tp_rank_agree_and_collapse():
    workers = _edd_workers(capacity=1, tp_size=4)
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))

    # One eviction, reported by four ranks, retires the victim once per stage.
    assert not coordinator.is_active("A")
    closes = [call for call in workers[ENCODE].calls if call == ("close_ar_diffusion_session", ("A",))]
    assert len(closes) == 1


def test_conflicting_tp_rank_reports_are_an_error():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))

    conflicting = [
        ARDiffusionReleaseEvent(event_id="rel-1-0", session_id="A", reason="lru_eviction", stage_id=1).to_dict(),
        ARDiffusionReleaseEvent(event_id="rel-1-0", session_id="A", reason="forward_exception", stage_id=1).to_dict(),
    ]
    workers[DENOISE]._dispatch = lambda method, args: (  # type: ignore[method-assign]
        conflicting if method == "get_ar_diffusion_release_events" else True
    )

    with pytest.raises(SessionLifecycleError, match="disagrees across ranks"):
        asyncio.run(_run_request(coordinator, workers, "r1", "A"))


def test_a_stage_that_failed_its_own_cleanup_is_not_acknowledged_as_clean():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))

    async def scenario():
        await coordinator.admit("r1", SessionControls("A"))
        workers[ENCODE].touch_session("A")
        # The denoise worker dropped the session but its own cleanup only
        # partly succeeded, so peers cannot assume the state is gone.
        workers[DENOISE].log.record("A", reason="forward_exception", cleanup_failed=True)
        await coordinator.complete("r1", success=True)

    with pytest.raises(SessionLifecycleError, match="failed to clean up"):
        asyncio.run(scenario())
    # Unacknowledged, so a retry still sees it.
    assert workers[DENOISE].log.pending_count() == 1
    assert coordinator.blocked_reason is not None


def test_a_dead_participant_invalidates_every_live_session():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))

    asyncio.run(coordinator.invalidate_all(reason="stage-1 replica-0 died"))

    assert coordinator.live_sessions == {}
    with pytest.raises(SessionNotLiveError):
        asyncio.run(coordinator.admit("r2", SessionControls("A", reset=False)))


def test_more_than_one_replica_per_stage_is_rejected():
    workers = _edd_workers()
    coordinator = _coordinator(workers, replica_count=lambda stage_id: 2 if stage_id == DENOISE else 1)

    with pytest.raises(SessionLifecycleError, match="one replica per stage"):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))


def test_a_stage_with_no_live_replica_is_rejected():
    workers = _edd_workers()
    coordinator = _coordinator(workers, replica_count=lambda stage_id: 0 if stage_id == ENCODE else 1)

    with pytest.raises(SessionLifecycleError, match="no live replica"):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))


def test_live_registry_stays_bounded_and_retires_its_victim_properly():
    workers = _edd_workers()
    coordinator = _coordinator(workers, max_live_sessions=2)

    for index in range(4):
        asyncio.run(_run_request(coordinator, workers, f"r{index}", f"S{index}", reset=True))

    assert len(coordinator.live_sessions) == 2
    assert set(coordinator.live_sessions) == {"S2", "S3"}
    # The registry's own eviction went through the same fan-out, so no stage is
    # left holding a session the coordinator forgot.
    assert set(workers[ENCODE].sessions) == {"S2", "S3"}


def test_generations_never_reuse_an_earlier_value_after_close():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    seen = []
    for index in range(5):
        asyncio.run(_run_request(coordinator, workers, f"r{index}", "A", reset=True, close_session=True))
        seen.append(max(worker.generations["A"] for worker in workers.values()))

    assert seen == sorted(set(seen))
    assert len(set(seen)) == 5


def test_a_hundred_begin_close_cycles_return_to_baseline():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    for index in range(100):
        asyncio.run(_run_request(coordinator, workers, f"r{index}", f"S{index}", reset=True, close_session=True))

    assert coordinator.live_sessions == {}
    for worker in workers.values():
        assert worker.sessions == {}
        assert worker.log.pending() == []
        assert worker.log.pending_count() == 0


# -- release-event log ------------------------------------------------------


def test_release_events_recorded_before_readiness_are_discarded():
    """Startup warmup drives real rollouts; those are not user sessions."""
    log = ARDiffusionReleaseEventLog(stage_id=1)

    assert log.record("__ardiffusion_warmup__", reason="warmup_complete") is None
    assert log.pending() == []

    log.set_ready()
    assert log.pending() == []
    assert log.record("A", reason="lru_eviction") is not None
    assert [event["session_id"] for event in log.pending()] == ["A"]


def test_reading_release_events_does_not_consume_them():
    log = ARDiffusionReleaseEventLog(stage_id=1)
    log.set_ready()
    log.record("A", reason="lru_eviction")

    first = log.pending()
    assert log.pending() == first
    assert log.pending_count() == 1

    assert log.acknowledge([first[0]["event_id"]]) == 1
    assert log.pending() == []
    # Acknowledging twice is harmless and reports nothing was still pending.
    assert log.acknowledge([first[0]["event_id"]]) == 0


def test_release_events_carry_the_registered_generation():
    log = ARDiffusionReleaseEventLog(stage_id=2)
    log.set_ready()
    log.register_generation("A", 9)

    log.record("A", reason="forward_exception")
    log.record("B", reason="forward_exception")

    by_session = {event["session_id"]: event for event in log.pending()}
    assert by_session["A"]["generation"] == 9
    assert by_session["A"]["stage_id"] == 2
    # A session the coordinator never registered reports generation 0.
    assert by_session["B"]["generation"] == 0

    log.forget_generation("A")
    log.record("A", reason="close")
    assert [event["generation"] for event in log.pending() if event["reason"] == "close"] == [0]


def test_coordinated_releases_are_suppressed_but_nested_use_is_safe():
    log = ARDiffusionReleaseEventLog(stage_id=1)
    log.set_ready()

    with log.coordinated("A"):
        with log.coordinated("A"):
            assert log.record("A", reason="close") is None
        # The inner block must not clear the outer suppression.
        assert log.record("A", reason="close") is None
    assert log.record("A", reason="close") is not None
    # Suppression is per session.
    with log.coordinated("A"):
        assert log.record("B", reason="close") is not None


def test_a_full_release_log_reports_overflow_instead_of_dropping_silently():
    log = ARDiffusionReleaseEventLog(stage_id=1, max_pending=2)
    log.set_ready()

    log.record("A", reason="lru_eviction")
    log.record("B", reason="lru_eviction")
    assert log.overflowed is False

    assert log.record("C", reason="lru_eviction") is None
    assert log.overflowed is True
    assert log.pending_count() == 2

    log.acknowledge([event["event_id"] for event in log.pending()])
    assert log.overflowed is False


def test_release_event_round_trips_through_its_wire_form():
    event = ARDiffusionReleaseEvent(
        event_id="rel-1-0",
        session_id="A",
        reason="lru_eviction",
        generation=4,
        stage_id=1,
        cleanup_failed=True,
    )

    assert ARDiffusionReleaseEvent.from_dict(event.to_dict()) == event
    assert ARDiffusionReleaseEvent.from_dict(event) is event
    with pytest.raises(ValueError, match="event_id"):
        ARDiffusionReleaseEvent.from_dict({"session_id": "A"})
    with pytest.raises(TypeError):
        ARDiffusionReleaseEvent.from_dict(["not", "a", "dict"])


# -- remaining ordering cases ----------------------------------------------


def test_touching_b_makes_a_the_eviction_victim():
    """Capacity two: begin A, begin B, touch B, begin C -> A is the victim."""
    workers = _edd_workers(capacity=2)
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r2", "B"))
    asyncio.run(_run_request(coordinator, workers, "r3", "C", reset=True))

    assert not coordinator.is_active("A")
    assert coordinator.is_active("B") and coordinator.is_active("C")
    for stage_id in (ENCODE, DENOISE):
        assert "A" not in workers[stage_id].sessions
        assert set(workers[stage_id].sessions) == {"B", "C"}


def test_unknown_continuation_while_full_releases_nothing():
    workers = _edd_workers(capacity=2)
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))

    before = {stage_id: dict(worker.sessions) for stage_id, worker in workers.items()}
    closes_before = sum(1 for call in workers[ENCODE].calls if call[0] == "close_ar_diffusion_session")

    with pytest.raises(SessionNotLiveError):
        asyncio.run(_run_request(coordinator, workers, "r2", "unknown"))

    assert {stage_id: dict(worker.sessions) for stage_id, worker in workers.items()} == before
    closes_after = sum(1 for call in workers[ENCODE].calls if call[0] == "close_ar_diffusion_session")
    assert closes_after == closes_before
    assert set(coordinator.live_sessions) == {"A", "B"}


def test_a_late_release_event_for_a_reused_id_is_ignored():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    stale_generation = coordinator.generation_of("A")
    asyncio.run(coordinator.close("A"))
    # The id is reused for a brand-new rollout.
    asyncio.run(_run_request(coordinator, workers, "r1", "A", reset=True))
    fresh_generation = coordinator.generation_of("A")
    assert fresh_generation != stale_generation

    async def scenario():
        await coordinator.admit("r2", SessionControls("A"))
        # A release event from the retired generation arrives late.
        workers[DENOISE].log.register_generation("A", stale_generation)
        workers[DENOISE].log.record("A", reason="lru_eviction")
        workers[DENOISE].log.register_generation("A", fresh_generation)
        with pytest.raises(SessionLifecycleError, match="generation"):
            await coordinator.complete("r2", success=True)

    asyncio.run(scenario())
    # The new rollout was not retired by the stale event.
    assert coordinator.generation_of("A") == fresh_generation


# -- strict acknowledgement and transactional admission ---------------------


class RegistrationRefusingWorker(FakeStageWorker):
    """A worker whose registration returns the real ``False`` contract."""

    def _dispatch(self, method, args):
        if method == "register_ar_diffusion_generation":
            self.calls.append((method, args))
            return False
        return super()._dispatch(method, args)


class RegistrationErroringWorker(FakeStageWorker):
    def _dispatch(self, method, args):
        if method == "register_ar_diffusion_generation":
            self.calls.append((method, args))
            raise RuntimeError("registration exploded")
        return super()._dispatch(method, args)


class RegistrationSilentWorker(FakeStageWorker):
    """Returns None, the shape a worker with no lifecycle support produces."""

    def _dispatch(self, method, args):
        if method == "register_ar_diffusion_generation":
            self.calls.append((method, args))
            return None
        return super()._dispatch(method, args)


def _closes(worker: FakeStageWorker, session_id: str) -> int:
    return sum(1 for call in worker.calls if call == ("close_ar_diffusion_session", (session_id,)))


@pytest.mark.parametrize(
    "worker_cls",
    [RegistrationRefusingWorker, RegistrationErroringWorker, RegistrationSilentWorker],
)
def test_a_declined_registration_fails_admission_and_publishes_nothing(worker_cls):
    """`False`, a raised error and `None` are all failures, not acknowledgements."""
    workers = _edd_workers()
    healthy_denoise = workers[DENOISE]
    workers[DENOISE] = worker_cls(DENOISE)
    coordinator = _coordinator(workers)

    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))

    assert coordinator.live_sessions == {}
    assert not coordinator.is_active("A")
    # Rolled back on every state-owning participant, including the one that
    # acknowledged before the failure.
    assert _closes(workers[ENCODE], "A") == 1
    # The gate was released, so the topology still serves other work.
    workers[DENOISE] = healthy_denoise
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    assert coordinator.is_active("B")


def test_a_failed_admission_cannot_be_continued():
    workers = _edd_workers()
    workers[DENOISE] = RegistrationRefusingWorker(DENOISE)
    coordinator = _coordinator(workers)

    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))

    with pytest.raises(SessionNotLiveError):
        asyncio.run(coordinator.admit("r1", SessionControls("A")))


def test_a_rollback_that_cannot_be_confirmed_blocks_the_topology():
    workers = _edd_workers()
    workers[DENOISE] = RegistrationRefusingWorker(DENOISE)
    workers[ENCODE].fail_close.add("A")
    coordinator = _coordinator(workers)

    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))

    assert coordinator.blocked_reason is not None
    assert "rollback" in coordinator.blocked_reason
    with pytest.raises(SessionLifecycleError, match="blocked pending recovery"):
        asyncio.run(coordinator.admit("r1", SessionControls("B", reset=True)))


def test_a_failed_reset_does_not_restore_the_old_generation():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    old_generation = coordinator.generation_of("A")

    # The reset retires A, then its replacement registration is refused.
    workers[DENOISE] = RegistrationRefusingWorker(DENOISE)
    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.admit("r1", SessionControls("A", reset=True)))

    # Neither generation is continuable; the old one is not resurrected.
    assert coordinator.live_sessions == {}
    assert coordinator.generation_of("A") != old_generation
    with pytest.raises(SessionNotLiveError):
        asyncio.run(coordinator.admit("r2", SessionControls("A")))


def test_the_generation_counter_never_rewinds_after_a_failure():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    first = coordinator.generation_of("A")

    failing = RegistrationRefusingWorker(DENOISE)
    healthy = workers[DENOISE]
    workers[DENOISE] = failing
    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.admit("r1", SessionControls("B", reset=True)))

    workers[DENOISE] = healthy
    asyncio.run(_run_request(coordinator, workers, "r2", "B", reset=True))

    # The failed attempt consumed an id rather than handing it to B.
    assert coordinator.generation_of("B") > first + 1


def test_an_unexpected_transport_error_still_reaches_the_other_participants():
    workers = _edd_workers()

    class ExplodingRPC(FakeTopologyRuntime):
        async def rpc(self, method, stage_id, args):
            if method == "register_ar_diffusion_generation" and stage_id == DENOISE:
                raise ConnectionResetError("transport died")
            return await super().rpc(method, stage_id, args)

    topology = DiffusionStageLifecycleTopology(
        stage_ids=(ENCODE, DENOISE, DECODE), state_owning_stage_ids=(ENCODE, DENOISE)
    )
    coordinator = DiffusionStageLifecycleCoordinator(topology, ExplodingRPC(workers).rpc)

    with pytest.raises(SessionLifecycleError, match="ConnectionResetError"):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))

    assert coordinator.live_sessions == {}
    assert _closes(workers[ENCODE], "A") == 1


def test_capacity_eviction_followed_by_failure_does_not_resurrect_the_victim():
    workers = _edd_workers()
    coordinator = _coordinator(workers, max_live_sessions=2)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))

    # C is admitted and evicts A, then its own request fails.
    asyncio.run(_run_request(coordinator, workers, "r2", "C", reset=True, success=False))

    assert not coordinator.is_active("A")
    assert not coordinator.is_active("C")
    assert coordinator.is_active("B")
    with pytest.raises(SessionNotLiveError):
        asyncio.run(coordinator.admit("r3", SessionControls("A")))


def test_a_failed_release_event_ack_keeps_the_events_pending():
    workers = _edd_workers(capacity=1)

    class AckRefusingWorker(FakeStageWorker):
        def _dispatch(self, method, args):
            if method == "ack_ar_diffusion_release_events":
                self.calls.append((method, args))
                return {"supported": False, "error": "ack transport down"}
            return super()._dispatch(method, args)

    workers[DENOISE] = AckRefusingWorker(DENOISE)
    workers[DENOISE].capacity = 1
    coordinator = _coordinator(workers)

    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    with pytest.raises(SessionLifecycleError, match="ack transport down"):
        asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))

    # Unacknowledged, so recovery still sees them.
    assert workers[DENOISE].log.pending_count() >= 1
    assert coordinator.blocked_reason is not None


def test_an_empty_stage_reply_is_not_an_acknowledgement():
    workers = _edd_workers()

    class EmptyReplyRuntime(FakeTopologyRuntime):
        async def rpc(self, method, stage_id, args):
            if method == "register_ar_diffusion_generation" and stage_id == DECODE:
                # No live replica answered.
                return []
            return await super().rpc(method, stage_id, args)

    topology = DiffusionStageLifecycleTopology(
        stage_ids=(ENCODE, DENOISE, DECODE), state_owning_stage_ids=(ENCODE, DENOISE)
    )
    coordinator = DiffusionStageLifecycleCoordinator(topology, EmptyReplyRuntime(workers).rpc)

    with pytest.raises(SessionLifecycleError, match="empty reply"):
        asyncio.run(coordinator.admit("r0", SessionControls("A", reset=True)))
    assert coordinator.live_sessions == {}


def test_cancellation_during_registration_leaves_no_orphan_and_frees_the_gate():
    workers = _edd_workers()

    class HangingRuntime(FakeTopologyRuntime):
        async def rpc(self, method, stage_id, args):
            if method == "register_ar_diffusion_generation" and stage_id == DENOISE:
                await asyncio.sleep(10)
            return await super().rpc(method, stage_id, args)

    topology = DiffusionStageLifecycleTopology(
        stage_ids=(ENCODE, DENOISE, DECODE), state_owning_stage_ids=(ENCODE, DENOISE)
    )
    coordinator = DiffusionStageLifecycleCoordinator(topology, HangingRuntime(workers).rpc)

    async def scenario():
        task = asyncio.create_task(coordinator.admit("r0", SessionControls("A", reset=True)))
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The gate is free and nothing orphaned.
        assert coordinator.live_sessions == {}
        await coordinator.admit("r1", SessionControls("B", reset=True))
        await coordinator.complete("r1", success=True)

    asyncio.run(scenario())
    assert coordinator.is_active("B")


# -- terminal ordering: settle before publishing ----------------------------


def test_settle_keeps_the_gate_until_admission_is_released():
    """The next generation must not start while this outcome is undecided."""
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    order: list[str] = []

    async def scenario():
        await coordinator.admit("r0", SessionControls("A", reset=True))
        workers[ENCODE].touch_session("A")
        workers[DENOISE].touch_session("A")
        await coordinator.settle("r0", success=True)
        order.append("settled")

        async def second():
            await coordinator.admit("r1", SessionControls("B", reset=True))
            order.append("r1-admitted")
            await coordinator.complete("r1", success=True)

        task = asyncio.create_task(second())
        for _ in range(5):
            await asyncio.sleep(0)
        # Settled but not released: nobody else got in.
        order.append("still-gated")
        await coordinator.release_admission("r0")
        await task

    asyncio.run(scenario())

    assert order == ["settled", "still-gated", "r1-admitted"]


def test_settle_raises_so_a_caller_can_withhold_a_terminal_success():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    async def scenario():
        await coordinator.admit("r0", SessionControls("A", reset=True, close_session=True))
        workers[ENCODE].touch_session("A")
        workers[DENOISE].touch_session("A")
        workers[ENCODE].fail_close.add("A")
        with pytest.raises(SessionLifecycleError):
            await coordinator.settle("r0", success=True)
        # Still gated after a failed settle, so nothing races the failure.
        assert coordinator.is_inflight("r0")
        await coordinator.release_admission("r0")

    asyncio.run(scenario())
    assert coordinator.blocked_reason is not None


def test_settling_the_same_request_twice_is_a_no_op():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    async def scenario():
        await coordinator.admit("r0", SessionControls("A", reset=True))
        workers[ENCODE].touch_session("A")
        workers[DENOISE].touch_session("A")
        await coordinator.settle("r0", success=True)
        await coordinator.release_admission("r0")
        # Re-entering cleanup for the same id does nothing and does not double
        # release the gate.
        await coordinator.settle("r0", success=True)
        await coordinator.release_admission("r0")
        assert not coordinator.is_inflight("r0")
        await coordinator.admit("r1", SessionControls("A"))
        await coordinator.complete("r1", success=True)

    asyncio.run(scenario())
    assert coordinator.is_active("A")


def test_is_inflight_tracks_only_the_admitted_request():
    workers = _edd_workers()
    coordinator = _coordinator(workers)

    async def scenario():
        assert coordinator.is_inflight("r0") is False
        await coordinator.admit("r0", SessionControls("A", reset=True))
        assert coordinator.is_inflight("r0") is True
        assert coordinator.is_inflight("r1") is False
        await coordinator.complete("r0", success=True)
        assert coordinator.is_inflight("r0") is False

    asyncio.run(scenario())


def test_a_close_cannot_be_confirmed_while_the_topology_is_blocked():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))
    asyncio.run(_run_request(coordinator, workers, "r1", "B", reset=True))
    workers[ENCODE].fail_close.add("A")

    with pytest.raises(SessionLifecycleError):
        asyncio.run(coordinator.close("A"))
    assert coordinator.blocked_reason is not None

    # A later close of an absent session is not a clean success while unresolved.
    with pytest.raises(SessionLifecycleError, match="blocked pending recovery"):
        asyncio.run(coordinator.close("B"))


def test_a_begin_is_refused_while_the_same_id_is_being_closed():
    workers = _edd_workers()
    coordinator = _coordinator(workers)
    asyncio.run(_run_request(coordinator, workers, "r0", "A", reset=True))

    async def scenario():
        # The close is requested and parked behind the gate that admit holds.
        await coordinator.admit("r1", SessionControls("A"))
        close_task = asyncio.create_task(coordinator.close("A"))
        for _ in range(5):
            await asyncio.sleep(0)
        await coordinator.complete("r1", success=True)
        await close_task

    asyncio.run(scenario())
    assert not coordinator.is_active("A")

    # And a begin while a close is pending is refused outright.
    coordinator.request_close("C")
    with pytest.raises(SessionLifecycleError, match="close in progress"):
        asyncio.run(coordinator.admit("r2", SessionControls("C", reset=True)))
