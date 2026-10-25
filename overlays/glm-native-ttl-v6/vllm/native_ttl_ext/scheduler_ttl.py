# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU-testable scheduler adapter; no automatic serving activation.

Admission accepts only server-verified token endpoints. A future API/Core RPC
integration must call admission before generation, translate TTLRejected into
a client error, and emit truthful cache usage. That integration is not enabled
by this module. No sampler, dtype, draft model, or tensor is modified here.
"""
from dataclasses import dataclass

from .native_ttl import TTLInvariantError, TTLRejected, TTLCapacityRejected
from .ttl_protocol import expand_boundaries


@dataclass(frozen=True)
class EngineContract:
    kv_dtype: str
    mtp_tokens: int
    tensor_parallel: int
    groups: int
    block_size: int
    identity: str

    def validate(self, pool):
        if (self.kv_dtype != 'fp8_ds_mla' or self.mtp_tokens != 3
                or self.tensor_parallel != 8 or self.groups != 1
                or self.block_size != pool.hash_block_size or not self.identity):
            raise TTLRejected('Candidate requires the bound FP8 MLA / MTP=3 / TP=8 single-group engine')


@dataclass
class Plan:
    ends: dict[bytes, int]
    expires: float
    expired: bool = False
    complete: bool = False


class SchedulerTTL:
    def __init__(self, controller, contract):
        contract.validate(controller.pool)
        self.controller = controller
        self.contract = contract
        self.plans = {}

    def admit(self, request, verified_boundaries, *, started_at, identity):
        """request.block_hashes are native precomputed hashes, not user input."""
        if identity != self.contract.identity:
            raise TTLRejected('Renderer/model/cache identity mismatch')
        if request.request_id in self.plans:
            raise TTLRejected('Duplicate TTL scheduler plan')
        if len(self.plans) >= self.controller.max_reservations:
            raise TTLCapacityRejected('Too many outstanding TTL requests')
        block_size = self.contract.block_size
        prompt_blocks = request.num_prompt_tokens // block_size
        hashes = request.block_hashes[:prompt_blocks]
        # Only GLM's single cache group is qualified by this candidate.
        grouped = [bytes(h) + (0).to_bytes(4, 'big') for h in hashes]
        spans = expand_boundaries(verified_boundaries, grouped, block_size,
                                  prompt_tokens=request.num_prompt_tokens)
        self.controller.reserve(request.request_id, spans, started_at=started_at)
        ends = {key: (i + 1) * block_size for i, key in enumerate(grouped)}
        self.plans[request.request_id] = Plan(
            {key: ends[key] for key, _ in spans},
            max(started_at + seconds for _, seconds in spans))

    def after_completed_step(self, request, manager, *, processed_step_seq):
        """Call after the engine's completion fence, before freeing KV blocks."""
        owner = request.request_id
        if owner not in self.plans:
            return
        self.tick()
        plan = self.plans[owner]
        if plan.expired:
            # Long generation may legitimately outlast the requested TTL.
            # Expiry releases the cache promise, not the model request.
            return
        if plan.complete:
            return
        if request.last_sched_seq > processed_step_seq:
            raise TTLRejected('GPU step is still in flight; cannot promise cache')
        completed = min(request.num_prompt_tokens,
                        max(0, request.num_computed_tokens - request.num_in_flight_tokens))
        if manager.block_pool is not self.controller.pool or manager.num_kv_cache_groups != 1:
            raise TTLInvariantError('Scheduler/pool identity mismatch')
        groups = manager.get_blocks(owner).blocks
        if len(groups) != 1:
            raise TTLInvariantError('Unqualified KV group layout')
        entries = []
        remaining = self.controller.requests[owner].deadlines
        for key, end in plan.ends.items():
            if key in remaining and key not in self.controller.requests[owner].published and end <= completed:
                index = end // self.contract.block_size - 1
                if index >= len(groups[0]):
                    raise TTLRejected('Missing completed prefix block')
                block = groups[0][index]
                if self.controller.publication_ready(key, block, end):
                    entries.append((key, block, end))
        self.controller.publish(owner, entries, completed_prompt_tokens=completed)
        reservation = self.controller.requests[owner]
        plan.complete = reservation.published == reservation.deadlines.keys()

    def finish(self, request):
        self.tick()
        plan = self.plans.get(request.request_id)
        if plan is None:
            raise TTLRejected('Unknown scheduler TTL plan')
        if plan.expired:
            del self.plans[request.request_id]
            return {'retention': 'expired_during_request'}
        self.controller.finish(request.request_id)
        self.plans.pop(request.request_id, None)
        return {'retention': 'committed'}

    def cancel(self, request):
        self.controller.cancel(request.request_id)
        self.plans.pop(request.request_id, None)

    def tick(self):
        """Also required while idle, to avoid retaining expired entries forever."""
        self.controller.reap()
        for owner, plan in self.plans.items():
            if not plan.expired and self.controller.last_time >= plan.expires:
                plan.expired = True
                plan.ends.clear()
        # The small, bounded completion records remain until finish/cancel.
        # Engine lifecycle hooks must call cancel in all error/disconnect paths.

    def close(self):
        self.controller.close()
        self.plans.clear()
