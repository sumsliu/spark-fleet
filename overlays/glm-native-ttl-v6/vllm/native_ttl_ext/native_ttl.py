# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, scheduler-thread-owned leases for native vLLM full KV blocks.

No CUDA operations, extra KV allocations, wall-clock deadlines, or background
threads. A lease owns exactly one additional native pool reference. Requests
must reserve before admission, publish only completed immutable prompt blocks,
then finish/cancel their reservation. This is NOT an installed serving hook.
"""
from dataclasses import dataclass, field
import math
import threading
import time


class TTLRejected(ValueError):
    """Admission/capability failure to be returned before starting generation."""


class TTLCapacityRejected(TTLRejected):
    """A bounded retention quota is full; the generation may still fit."""


class TTLInvariantError(RuntimeError):
    """Stop candidate qualification; never pretend an invalid lease is live."""


@dataclass
class Lease:
    block: object | None = None
    until: float = 0.0
    owners: set[str] = field(default_factory=set)


@dataclass
class Reservation:
    deadlines: dict[bytes, float]
    published: set[bytes] = field(default_factory=set)


class NativeTTL:
    def __init__(self, pool, *, capacity_blocks, working_set_blocks,
                 max_reservations=128, clock=time.monotonic):
        for value in (capacity_blocks, working_set_blocks, max_reservations):
            if type(value) is not int or value <= 0:
                raise TTLRejected("TTL bounds must be positive integers")
        if not pool.enable_caching:
            raise TTLRejected("TTL requires prefix caching")
        if capacity_blocks + working_set_blocks > pool.num_gpu_blocks - 1:
            raise TTLRejected("TTL capacity would consume the working-set reserve")
        self.pool = pool
        if not hasattr(pool, 'native_ttl') or pool.native_ttl is not None:
            raise TTLRejected("An isolated TTL-aware pool without a controller is required")
        self.capacity = capacity_blocks
        self.working_set = working_set_blocks
        self.max_reservations = max_reservations
        self.clock = clock
        self.leases: dict[bytes, Lease] = {}
        self.requests: dict[str, Reservation] = {}
        self.thread = threading.get_ident()
        self.last_time = -math.inf
        self.closed = False
        self.next_expiry = math.inf
        self.stats = {"admitted": 0, "rejected": 0, "published": 0,
                      "expired": 0, "cancelled": 0, "finished": 0,
                      "expiry_sweeps": 0, "opportunistic_bypassed": 0,
                      "publication_failed": 0}
        pool.native_ttl = self

    def _now(self):
        if threading.get_ident() != self.thread:
            raise TTLInvariantError("TTL mutations must run on the scheduler thread")
        now = self.clock()
        if not isinstance(now, (float, int)) or not math.isfinite(now) or now < self.last_time:
            raise TTLInvariantError("TTL requires a non-decreasing monotonic clock")
        self.last_time = now
        return now

    def _check_identity(self, block):
        if (block.is_null or block.pool is not self.pool or block.ref_cnt < 0
                or not 0 <= block.block_id < len(self.pool.blocks)
                or self.pool.blocks[block.block_id] is not block):
            raise TTLInvariantError("Foreign, null or invalid cache block")

    def _check_block(self, key, block):
        self._check_identity(block)
        end = block.block_hash_num_tokens
        # cache_full_blocks records the cumulative token end on full blocks too.
        # None remains valid for native legacy entries; a partial end is not.
        if (block.block_hash != key
                or (end is not None and (type(end) is not int or end <= 0
                                          or end % self.pool.hash_block_size))
                or self.pool.cached_block_hashes_by_block.get(block.block_id, set())
                or not self.pool.cached_block_hash_to_block.contain(key, block.block_id)):
            raise TTLInvariantError("Invalid retained full-block cache identity")

    def publication_ready(self, key, block, end):
        self._check_identity(block)
        # An active request must own the block independently of any TTL lease.
        if block.ref_cnt < 1:
            raise TTLInvariantError("Publication requires an active request reference")
        if block.block_hash is None:
            return False
        if (block.block_hash != key
                or block.block_hash_num_tokens not in (None, end)
                or self.pool.cached_block_hashes_by_block.get(block.block_id, set())):
            raise TTLRejected("Prefix block is not the requested immutable full block")
        # The hash may have been assigned before its lookup registration.
        return self.pool.cached_block_hash_to_block.contain(key, block.block_id)

    def _release(self, key, lease, *, purge):
        block = lease.block
        if block is not None:
            self._check_block(key, block)
            if block.ref_cnt < 1:
                raise TTLInvariantError("Lost TTL-owned reference")
            if purge:
                # Explicit invalidation (close): remove every duplicate physical
                # entry for this logical prefix, leaving active request
                # references untouched.
                while (duplicate := self.pool.cached_block_hash_to_block.get_one_block(key)) is not None:
                    self.pool.evict_blocks({duplicate.block_id})
            # Expiry only ends the guarantee: dropping our reference returns the
            # block to the native LRU free queue with its hash intact, so a later
            # request may still hit it until allocation pressure reuses it.
            self.pool.free_blocks([block])
            lease.block = None
            lease.until = 0.0

    def reap(self):
        """Call on scheduler ticks AND before cache lookup/allocation/reset."""
        now = self._now()
        # Native prefix lookup invokes this once per block. Most calls must be
        # O(1), not an O(prefix_length * retained_blocks) scan on every request.
        if now < self.next_expiry:
            return now
        self.stats["expiry_sweeps"] += 1
        # Pending reservations also expire; a stalled client must not hold quota.
        for owner, reservation in list(self.requests.items()):
            for key, deadline in list(reservation.deadlines.items()):
                if deadline <= now:
                    self.leases[key].owners.discard(owner)
                    del reservation.deadlines[key]
                    reservation.published.discard(key)
            if not reservation.deadlines:
                del self.requests[owner]
        for key, lease in list(self.leases.items()):
            if lease.block is not None and lease.until <= now:
                self._release(key, lease, purge=False)
                self.stats["expired"] += 1
            if lease.block is None and not lease.owners:
                del self.leases[key]
        self.next_expiry = min(
            [lease.until for lease in self.leases.values() if lease.block is not None]
            + [deadline for r in self.requests.values() for deadline in r.deadlines.values()],
            default=math.inf)
        return now

    def reserve(self, owner, spans, *, started_at):
        """Atomically reserve distinct hashes; no pin/renew on quota failure.

        spans is an ordered sequence of (full_block_hash_with_group, seconds).
        Hashes MUST be computed by the engine over the real token prefix with
        a stable authenticated namespace/cache_salt. Never accept client hashes.
        started_at is converted once to this engine's monotonic clock domain.
        """
        now = self.reap()
        if self.closed:
            raise TTLRejected("TTL controller is closed")
        if not isinstance(owner, str) or not owner or len(owner) > 256 or owner in self.requests:
            raise TTLRejected("Unique bounded request id required")
        if type(started_at) not in (int, float) or not math.isfinite(started_at) or started_at > now:
            raise TTLRejected("Invalid monotonic request start")
        desired = {}
        for key, seconds in spans:
            if not isinstance(key, bytes) or not 5 <= len(key) <= 132:
                raise TTLRejected("Engine block hash with group id required")
            if type(seconds) is not int or seconds not in (300, 3600):
                raise TTLRejected("TTL must be 5m or 1h")
            deadline = started_at + seconds
            if deadline <= now:
                raise TTLRejected("Cache deadline expired before admission")
            desired[key] = max(desired.get(key, 0), deadline)
            if len(desired) > self.capacity:
                self.stats["rejected"] += 1
                raise TTLCapacityRejected("TTL prefix exceeds capacity")
        if not desired:
            raise TTLRejected("No complete cacheable blocks")
        if (len(self.requests) >= self.max_reservations
                or len(self.leases.keys() | desired.keys()) > self.capacity):
            self.stats["rejected"] += 1
            raise TTLCapacityRejected("TTL reservation capacity exhausted")
        self.requests[owner] = Reservation(desired)
        for key in desired:
            lease = self.leases.setdefault(key, Lease())
            lease.owners.add(owner)
            if lease.block is not None:
                # Accepted reuse renews retention at request start. Do not
                # invalidate a shared hash while that request waits for GPU.
                lease.until = max(lease.until, desired[key])
        self.next_expiry = min(self.next_expiry, min(desired.values()))
        self.stats["admitted"] += 1

    def publish(self, owner, entries, *, completed_prompt_tokens):
        """Protect actual completed prompt blocks (never speculative draft KV).

        entries contains (engine_hash, native_block, prompt_token_end). Caller
        supplies only immutable completed prompt blocks, not scheduled tokens.
        Validation is all-or-nothing before any new reference is acquired.
        """
        self.reap()
        if owner not in self.requests:
            raise TTLRejected("Missing or expired TTL reservation")
        if type(completed_prompt_tokens) is not int or completed_prompt_tokens < 0:
            raise TTLRejected("Completed prompt token count required")
        reservation = self.requests[owner]
        selected = {}
        block_ids = set()
        for key, block, end in entries:
            if (key not in reservation.deadlines or type(end) is not int
                    or end <= 0 or end > completed_prompt_tokens
                    or end % self.pool.hash_block_size):
                raise TTLRejected("Unreserved, partial or not-yet-computed prefix")
            if not self.publication_ready(key, block, end):
                raise TTLRejected("Prefix block is not registered yet")
            if key in selected or block.block_id in block_ids:
                raise TTLRejected("Duplicate publish entry")
            selected[key] = block
            block_ids.add(block.block_id)
        for key, block in selected.items():
            lease = self.leases[key]
            if lease.block is None:
                self.pool.touch([block])
                lease.block = block
            else:
                # A duplicate physical block is allowed; keep the already
                # retained copy. Both must have this same complete prefix hash.
                self._check_block(key, lease.block)
            lease.until = max([lease.until, reservation.deadlines[key]] +
                              [self.requests[o].deadlines[key] for o in lease.owners])
            reservation.published.add(key)
            self.stats["published"] += 1

    def finish(self, owner):
        """Require all promised blocks published before claiming success."""
        self.reap()
        reservation = self.requests.get(owner)
        if reservation is None:
            raise TTLRejected("Reservation expired before completion")
        if reservation.published != reservation.deadlines.keys():
            raise TTLRejected("Cannot claim TTL: prefix not completely published")
        self._drop_owner(owner)
        self.stats["finished"] += 1

    def _drop_owner(self, owner):
        reservation = self.requests.pop(owner, None)
        if reservation is not None:
            for key in reservation.deadlines:
                lease = self.leases[key]
                lease.owners.discard(owner)
                if lease.block is None and not lease.owners:
                    del self.leases[key]

    def cancel(self, owner):
        self.reap()
        self._drop_owner(owner)
        self.stats["cancelled"] += 1

    def close(self):
        """Explicit shutdown/invalidation only, never a silent quota fallback."""
        self._now()
        for key, lease in self.leases.items():
            self._release(key, lease, purge=True)
        self.requests.clear()
        self.leases.clear()
        self.closed = True
        self.pool.native_ttl = None

    def snapshot(self):
        self.reap()
        return {"capacity_blocks": self.capacity, "working_set_blocks": self.working_set,
                "reserved_unique_blocks": len(self.leases),
                "retained_blocks": sum(x.block is not None for x in self.leases.values()),
                "pending_requests": len(self.requests), **self.stats}
