# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Topology-wide session lifecycle ordering for a disaggregated AR pipeline.

A session split across stages has its state in more than one place: the encode
stage owns the VAE observation history, the denoise stage owns paged KV and its
own window bookkeeping, and a trailing postprocess stage owns nothing. Each of
those was previously released independently, so an eviction inside the denoise
runner left the encoder still holding history for a session whose KV was gone --
the next continuation then encoded against stale history.

This coordinator owns the ordering instead:

* one end-to-end request in flight across the topology, so a request cannot
  enter encode while another is evicting its session in denoise;
* a coordinator-issued, globally increasing generation per begin/reset, carried
  by every participant so late payloads and late release events from an earlier
  generation are rejected rather than applied;
* explicit reset/close fanned out once, at the topology boundary, instead of
  each stage repeating a global cleanup;
* self-initiated releases (LRU eviction, failed forward) drained back from the
  workers at every request boundary and replayed onto the peer stages.

The class is transport-agnostic: it is handed an async ``rpc`` callable and
never imports the orchestrator, so it is exercised directly on CPU.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.release_events import ARDiffusionReleaseEvent

logger = init_logger(__name__)

# Roles holding per-session state. A trailing postprocess stage is stateless: it
# still takes part in lifecycle RPC (so an unsupported participant is visible)
# but owns nothing to retire.
_STATE_OWNING_ROLES = frozenset({"full", "encode", "denoise", "denoise_decode"})

DEFAULT_MAX_LIVE_SESSIONS = 64

# Async RPC into one stage: (method, stage_id, args) -> per-replica results.
StageRPC = Callable[[str, int, tuple[Any, ...]], Awaitable[Any]]


class SessionLifecycleError(RuntimeError):
    """A coordinated lifecycle operation could not be completed safely."""


class SessionStateLostError(SessionLifecycleError):
    """A continuation refers to a session whose state no longer exists.

    Recovery is an explicit new rollout: the caller must send a begin/reset.
    Reconstructing VAE history from a later chunk is not possible, so this is
    raised before any stage mutates counters, history or KV.
    """


@dataclass(frozen=True)
class SessionControls:
    """The lifecycle intent one request carries, normalized once."""

    session_id: str
    reset: bool = False
    close_session: bool = False


def read_session_controls(sampling_params_list: Sequence[Any]) -> SessionControls | None:
    """Read session controls from a request's per-stage sampling params.

    Returns ``None`` for a request that carries none, which is how a
    non-session workload sharing the same engine opts out.

    A typed ``ar_diffusion_tick`` is authoritative over the flat controls,
    matching the runner: a typed ``False`` is not overridden by a stale flat
    ``True``.
    """
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

    for params in sampling_params_list or ():
        extra_args = getattr(params, "extra_args", None)
        if not isinstance(extra_args, Mapping):
            continue
        tick = ARDiffusionTickRequest.from_extra_args(extra_args)
        if tick is not None:
            return SessionControls(
                session_id=str(tick.session_id),
                reset=bool(tick.reset),
                close_session=bool(tick.close_session),
            )
        session_id = extra_args.get("session_id")
        if session_id is None:
            continue
        return SessionControls(
            session_id=str(session_id),
            reset=bool(extra_args.get("reset", False)),
            close_session=bool(extra_args.get("close_session", False)),
        )
    return None


@dataclass(frozen=True)
class DiffusionStageLifecycleTopology:
    """Which stages take part, and which of them own session state."""

    stage_ids: tuple[int, ...]
    state_owning_stage_ids: tuple[int, ...]

    @classmethod
    def from_stage_configs(cls, stage_configs: Sequence[Any]) -> DiffusionStageLifecycleTopology | None:
        """Build the topology from declared stage configs, or None if not opted in."""
        from vllm_omni.config.stage_config import resolve_diffusion_stage_role

        stage_ids: list[int] = []
        state_owning: list[int] = []
        for index, stage_config in enumerate(stage_configs or ()):
            if not bool(getattr(stage_config, "coordinated_session_lifecycle", False)):
                continue
            stage_id = int(getattr(stage_config, "stage_id", index))
            stage_ids.append(stage_id)
            role = resolve_diffusion_stage_role(
                getattr(stage_config, "stage_role", None),
                getattr(stage_config, "model_stage", None),
            )
            if str(role.value) in _STATE_OWNING_ROLES:
                state_owning.append(stage_id)
        if not stage_ids:
            return None
        return cls(stage_ids=tuple(stage_ids), state_owning_stage_ids=tuple(state_owning))


@dataclass
class _InFlight:
    request_id: str
    session_id: str
    generation: int
    close_session: bool
    # Sessions this request's admission already retired, so a release event for
    # them is expected and must not look like a surprise eviction.
    coordinated_sessions: set[str] = field(default_factory=set)


class DiffusionStageLifecycleCoordinator:
    """Serialize and order session lifecycle across a declared topology."""

    def __init__(
        self,
        topology: DiffusionStageLifecycleTopology,
        rpc: StageRPC,
        *,
        max_live_sessions: int = DEFAULT_MAX_LIVE_SESSIONS,
        replica_count: Callable[[int], int] | None = None,
    ) -> None:
        if max_live_sessions <= 0:
            raise ValueError(f"max_live_sessions must be positive, got {max_live_sessions}")
        self.topology = topology
        self._rpc = rpc
        self._replica_count = replica_count
        self._max_live_sessions = int(max_live_sessions)
        # Bounded live registry; the generation counter is global and monotonic,
        # so a reused id never inherits an earlier generation and no tombstone
        # dictionary is needed.
        self._live: OrderedDict[str, int] = OrderedDict()
        self._next_generation = 0
        self._gate = asyncio.Lock()
        self._inflight: _InFlight | None = None
        self._blocked_reason: str | None = None

    # -- observability ---------------------------------------------------

    @property
    def live_sessions(self) -> dict[str, int]:
        return dict(self._live)

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def is_active(self, session_id: str) -> bool:
        return str(session_id) in self._live

    def generation_of(self, session_id: str) -> int | None:
        return self._live.get(str(session_id))

    def clear_block(self) -> None:
        """Allow admission again after an operator-level recovery."""
        self._blocked_reason = None

    # -- RPC helpers -----------------------------------------------------

    async def _call(self, method: str, stage_ids: Iterable[int], *args: Any) -> dict[int, Any]:
        results: dict[int, Any] = {}
        for stage_id in stage_ids:
            results[stage_id] = await self._rpc(method, stage_id, tuple(args))
        return results

    @staticmethod
    def _collect_support(value: Any, *, errors: list[str], supported: list[bool]) -> None:
        """Flatten per-replica/per-rank RPC results into support and error lists."""
        if isinstance(value, bool):
            supported.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                DiffusionStageLifecycleCoordinator._collect_support(item, errors=errors, supported=supported)
            return
        if isinstance(value, Mapping):
            if value.get("supported") is False:
                errors.append(str(value.get("error") or "stage does not support this lifecycle RPC"))
                return
            if "result" in value:
                DiffusionStageLifecycleCoordinator._collect_support(value["result"], errors=errors, supported=supported)
            return
        if value is None:
            # A worker that returned nothing did not report success.
            supported.append(False)

    async def _fan_out(self, method: str, session_id: str, stage_ids: Sequence[int]) -> None:
        """Run one lifecycle method on every listed stage, failing loudly."""
        results = await self._call(method, stage_ids, session_id)
        errors: list[str] = []
        supported: list[bool] = []
        for stage_id, value in results.items():
            stage_errors: list[str] = []
            self._collect_support(value, errors=stage_errors, supported=supported)
            errors.extend(f"stage {stage_id}: {error}" for error in stage_errors)
        if errors or not supported or not all(supported):
            detail = "; ".join(errors) or "at least one participant does not support it"
            raise SessionLifecycleError(
                f"{method} for session {session_id!r} did not complete on every participant: {detail}"
            )

    def _require_single_replica_layout(self) -> None:
        """Session-affine routing does not exist yet, so refuse a fan-out layout."""
        if self._replica_count is None:
            return
        for stage_id in self.topology.stage_ids:
            count = int(self._replica_count(stage_id))
            if count > 1:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle supports one replica per stage until "
                    f"session-affine routing exists, but stage {stage_id} has {count}. "
                    "TP ranks inside one replica are supported; multiple replicas are not."
                )
            if count == 0:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle stage {stage_id} has no live replica; "
                    "the session's state cannot be reached."
                )

    # -- registry --------------------------------------------------------

    def _new_generation(self, session_id: str) -> int:
        self._next_generation += 1
        generation = self._next_generation
        self._live[str(session_id)] = generation
        self._live.move_to_end(str(session_id))
        return generation

    def _retire_locally(self, session_id: str) -> None:
        self._live.pop(str(session_id), None)

    async def _evict_oldest_if_needed(self) -> None:
        """Keep the live registry bounded by retiring the oldest session properly."""
        while len(self._live) > self._max_live_sessions:
            victim, _ = next(iter(self._live.items()))
            logger.warning(
                "Coordinated session registry is full (%d); retiring least-recently-used session %s",
                self._max_live_sessions,
                victim,
            )
            self._retire_locally(victim)
            await self._fan_out(
                "close_ar_diffusion_session",
                victim,
                self.topology.state_owning_stage_ids,
            )

    # -- admission -------------------------------------------------------

    async def admit(self, request_id: str, controls: SessionControls) -> int:
        """Order one request against the topology and return its generation.

        Blocks until the previous end-to-end request has finished and its
        lifecycle acknowledgements are in. Raises before any stage is mutated
        when the request cannot be served safely; the caller must then report
        the error and must not submit the request.
        """
        await self._gate.acquire()
        try:
            if self._blocked_reason is not None:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle is blocked pending recovery: {self._blocked_reason}"
                )
            self._require_single_replica_layout()
            session_id = str(controls.session_id)
            coordinated: set[str] = set()

            if controls.reset:
                # Older work is already drained by the gate. Retire every
                # participant once, here, so no downstream stage repeats it.
                if session_id in self._live:
                    await self._retire_session(session_id, reason="coordinated_reset")
                    coordinated.add(session_id)
                generation = self._new_generation(session_id)
                await self._evict_oldest_if_needed()
            else:
                generation = self._live.get(session_id, 0)
                if not generation:
                    raise SessionStateLostError(
                        f"DreamZero session {session_id!r} has no live state on this topology; "
                        "its history was released (explicit close, eviction, or a failed request) "
                        "and a continuation cannot rebuild it. Start a new rollout with an "
                        "explicit reset."
                    )
                self._live.move_to_end(session_id)

            await self._register_generation(session_id, generation)
            self._inflight = _InFlight(
                request_id=str(request_id),
                session_id=session_id,
                generation=generation,
                close_session=bool(controls.close_session),
                coordinated_sessions=coordinated,
            )
            return generation
        except BaseException:
            self._inflight = None
            self._gate.release()
            raise

    async def _register_generation(self, session_id: str, generation: int) -> None:
        """Bind the generation on every participant before payload execution.

        The stateless postprocess stage records identity only; nothing resident
        is allocated for it.
        """
        results = await self._call("register_ar_diffusion_generation", self.topology.stage_ids, session_id, generation)
        for stage_id, value in results.items():
            errors: list[str] = []
            supported: list[bool] = []
            self._collect_support(value, errors=errors, supported=supported)
            if errors:
                raise SessionLifecycleError(
                    f"Registering generation {generation} for session {session_id!r} failed on stage "
                    f"{stage_id}: {'; '.join(errors)}"
                )

    async def _retire_session(self, session_id: str, *, reason: str) -> None:
        """Clear a session on every state-owning participant and mark it inactive."""
        self._retire_locally(session_id)
        await self._fan_out(
            "close_ar_diffusion_session",
            session_id,
            self.topology.state_owning_stage_ids,
        )
        logger.info("Coordinated session %s retired across the topology (%s)", session_id, reason)

    # -- completion ------------------------------------------------------

    async def complete(self, request_id: str, *, success: bool) -> None:
        """Close out one admitted request, then let the next one in.

        Runs for a returned error and a raised exception alike: the release
        events a worker recorded on either path must still be drained before
        another request can reuse affected session state.
        """
        inflight = self._inflight
        if inflight is None or inflight.request_id != str(request_id):
            # Not the request holding the gate (already completed, or never
            # admitted); nothing to release.
            return
        self._inflight = None
        try:
            failures: list[str] = []
            try:
                await self._drain_release_events(inflight)
            except SessionLifecycleError as exc:
                failures.append(str(exc))

            if not success:
                # The affected generation is invalid on every participant; it is
                # not retried as a continuation.
                try:
                    await self._retire_session(inflight.session_id, reason="request_failure")
                except SessionLifecycleError as exc:
                    failures.append(str(exc))
            elif inflight.close_session and inflight.session_id in self._live:
                try:
                    await self._retire_session(inflight.session_id, reason="explicit_close")
                except SessionLifecycleError as exc:
                    failures.append(str(exc))

            if failures:
                # Cleanup did not complete: block reuse rather than report a
                # synchronized success we did not achieve.
                self._blocked_reason = "; ".join(failures)
                raise SessionLifecycleError(self._blocked_reason)
        finally:
            self._gate.release()

    async def close(self, session_id: str) -> None:
        """Explicit close with no inference: clear participants and mark inactive."""
        async with self._gate:
            await self._retire_session(str(session_id), reason="explicit_close")

    async def invalidate_all(self, *, reason: str) -> None:
        """Drop every live session and clean whatever peer state is reachable.

        Used when a participant died: its KV did not survive the restart, so no
        session that spanned it may be continued. This deliberately does not take
        the admission gate -- it runs while the request that lost its worker is
        still being torn down -- and it blocks admission when peer cleanup could
        not be completed, rather than reporting a cleanup it did not achieve.
        """
        sessions = list(self._live)
        self._live.clear()
        if not sessions:
            return
        logger.error(
            "Invalidating %d coordinated session(s) after %s; a continuation now requires an explicit reset",
            len(sessions),
            reason,
        )
        failures: list[str] = []
        for session_id in sessions:
            try:
                await self._fan_out(
                    "close_ar_diffusion_session",
                    session_id,
                    self.topology.state_owning_stage_ids,
                )
            except SessionLifecycleError as exc:
                failures.append(str(exc))
        if failures:
            self._blocked_reason = f"{reason}: " + "; ".join(failures)

    # -- reverse path ----------------------------------------------------

    async def _drain_release_events(self, inflight: _InFlight) -> None:
        """Replay worker-initiated releases onto the peer stages, then ack them."""
        stage_ids = self.topology.state_owning_stage_ids
        if not stage_ids:
            return
        raw = await self._call("get_ar_diffusion_release_events", stage_ids)

        conflicts: list[str] = []
        # session -> generation seen, and stage -> event ids to acknowledge
        victims: dict[str, int] = {}
        acks: dict[int, list[str]] = {}
        for stage_id, value in raw.items():
            events, decode_errors = _decode_release_events(value)
            conflicts.extend(f"stage {stage_id}: {error}" for error in decode_errors)
            # TP ranks of one replica report the same releases; agreeing records
            # collapse, a disagreeing generation is an error.
            for event in events:
                acks.setdefault(stage_id, []).append(event.event_id)
                if event.cleanup_failed:
                    conflicts.append(
                        f"stage {stage_id} failed to clean up session {event.session_id!r} "
                        f"({event.reason}); peer state cannot be assumed released"
                    )
                registered = self._live.get(event.session_id)
                if event.generation and registered is not None and event.generation != registered:
                    conflicts.append(
                        f"stage {stage_id} released session {event.session_id!r} at generation "
                        f"{event.generation} but generation {registered} is registered"
                    )
                    continue
                if event.generation and registered is None:
                    # The session is already retired here: an expected event
                    # from a cleanup this coordinator drove, or a late duplicate.
                    continue
                previous = victims.get(event.session_id)
                if previous is not None and event.generation and previous and previous != event.generation:
                    conflicts.append(
                        f"session {event.session_id!r} was released at conflicting generations "
                        f"{previous} and {event.generation}"
                    )
                victims[event.session_id] = event.generation or (previous or 0)

        retire_errors: list[str] = []
        for session_id in list(victims):
            if session_id in inflight.coordinated_sessions:
                # Already handled at admission; do not rebroadcast.
                continue
            if session_id == inflight.session_id:
                logger.warning(
                    "Coordinated session %s was released by a worker while its own request was in flight",
                    session_id,
                )
            try:
                await self._retire_session(session_id, reason="worker_release_event")
            except SessionLifecycleError as exc:
                retire_errors.append(str(exc))

        if conflicts or retire_errors:
            # Do not acknowledge: the events must survive for a retry.
            raise SessionLifecycleError("; ".join(conflicts + retire_errors))

        for stage_id, event_ids in acks.items():
            if event_ids:
                await self._call("ack_ar_diffusion_release_events", [stage_id], event_ids)


def _decode_release_events(value: Any) -> tuple[list[ARDiffusionReleaseEvent], list[str]]:
    """Flatten one stage's RPC result into release events, deduped by event id."""
    events: dict[str, ARDiffusionReleaseEvent] = {}
    errors: list[str] = []

    def add(event: ARDiffusionReleaseEvent) -> None:
        # TP ranks of one replica report the same releases under the same ids;
        # agreeing records collapse, disagreeing ones are a synchronization bug.
        existing = events.get(event.event_id)
        if existing is not None and existing != event:
            errors.append(
                f"release event {event.event_id} disagrees across ranks: {existing.to_dict()} vs {event.to_dict()}"
            )
            return
        events[event.event_id] = event

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, ARDiffusionReleaseEvent):
            add(node)
            return
        if isinstance(node, Mapping):
            if node.get("supported") is False:
                errors.append(str(node.get("error") or "stage does not support release-event reporting"))
                return
            if "event_id" in node:
                try:
                    event = ARDiffusionReleaseEvent.from_dict(node)
                except (TypeError, ValueError) as exc:
                    errors.append(f"malformed release event: {exc}")
                    return
                add(event)
                return
            if "result" in node:
                walk(node["result"])
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        errors.append(f"unexpected release-event payload of type {type(node).__name__}")

    walk(value)
    return list(events.values()), errors
