# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Offline candidate integration. All mutations occur on the engine thread."""
from collections import OrderedDict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

from .native_ttl import NativeTTL, TTLRejected, TTLCapacityRejected
from .scheduler_ttl import EngineContract, SchedulerTTL
from .ttl_protocol import Marker, expand_boundaries

PLAN_KEY = '_native_ttl_v2'
CONFIG_ENV = 'VLLM_NATIVE_TTL_CONFIG'
logger = logging.getLogger(__name__)


def config():
    path = os.environ.get(CONFIG_ENV)
    if path is None:
        local = Path(__file__).with_name('deployment.json')
        path = str(local) if local.is_file() else None
    if not path:
        return None
    with open(path) as f:
        value = json.load(f)
    if value.get('schema') != 2 or not value.get('identity'):
        raise TTLRejected('Invalid offline TTL candidate configuration')
    return value


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def reject_external(args):
    if any(k.startswith('_native_ttl') for k in (args or {})):
        raise TTLRejected('Reserved internal TTL request field')


def attach(scheduler, vllm_config):
    cfg = config()
    if cfg is None:
        return None
    parallel = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    manager = scheduler.kv_cache_manager
    if (parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or scheduler.connector is not None or scheduler.ec_connector is not None
            or spec is None or spec.method != 'mtp'
            or vllm_config.cache_config.block_size != manager.block_pool.hash_block_size):
        raise TTLRejected('TTL candidate requires local, single-group FP8 MLA/MTP engine')
    contract = EngineContract(vllm_config.cache_config.cache_dtype,
                             spec.num_speculative_tokens, parallel.tensor_parallel_size,
                             manager.num_kv_cache_groups, manager.block_pool.hash_block_size,
                             cfg['identity'])
    contract.validate(manager.block_pool)
    minimum = math.ceil((vllm_config.model_config.max_model_len + max(spec.num_speculative_tokens, vllm_config.max_in_flight_tokens))
                        / contract.block_size)
    if 'capacity_tokens' in cfg:
        if type(cfg['capacity_tokens']) is not int or cfg['capacity_tokens'] <= 0:
            raise TTLRejected('TTL token capacity must be a positive integer')
        cfg = dict(cfg, capacity_blocks=cfg['capacity_tokens'] // contract.block_size,
                   working_set_blocks=minimum)
    if cfg['working_set_blocks'] < minimum:
        raise TTLRejected('TTL must leave enough unpinned blocks for a full-context request')
    controller = NativeTTL(manager.block_pool, capacity_blocks=cfg['capacity_blocks'],
                           working_set_blocks=cfg['working_set_blocks'],
                           max_reservations=cfg.get('max_requests', 128))
    return Runtime(controller, contract)


class Runtime(SchedulerTTL):
    def __init__(self, controller, contract):
        super().__init__(controller, contract)
        self.receipts = OrderedDict()
        self.publications = {}
        self.ttls = {}

    def admit_wire(self, request):
        plan = (request.sampling_params.extra_args or {}).get(PLAN_KEY)
        if not isinstance(plan, dict) or plan.get('schema') != 2:
            raise TTLRejected('Invalid TTL admission plan')
        if (getattr(request, 'resumable', False) or getattr(request, 'mm_features', None)
                or getattr(request, 'lora_request', None)
                or getattr(request, 'kv_transfer_params', None)
                or getattr(request, 'ec_transfer_params', None)):
            raise TTLRejected('TTL candidate supports text, non-resumable local requests only')
        if plan['digest'] != token_digest(request.prompt_token_ids):
            raise TTLRejected('Rendered prompt changed before engine admission')
        boundaries = [(Marker((), seconds), end) for end, seconds in plan['ends']]
        if plan['identity'] != self.contract.identity:
            raise TTLRejected('Renderer/model/cache identity mismatch')
        grouped = [bytes(h) + b'\0'*4 for h in request.block_hashes[:request.num_prompt_tokens // self.contract.block_size]]
        spans = expand_boundaries(boundaries, grouped, self.contract.block_size,
                                  prompt_tokens=request.num_prompt_tokens)
        if not spans:
            # Like an APC prefix below its cache granularity, this remains a
            # valid generation request, with no physical TTL promise or quota.
            return False
        self.admit(request, boundaries, started_at=plan['started_at'], identity=plan['identity'])
        owner = request.request_id
        self.publications[owner] = {}
        self.ttls[owner] = {k: int(round(deadline - plan['started_at']))
                           for k, deadline in self.controller.requests[owner].deadlines.items()}
        return True

    def completed(self, request, manager):
        owner = request.request_id
        if owner not in self.plans or self.plans[owner].complete:
            return
        # update_from_output has observed this GPU result and subtracted rejected
        # MTP tokens. Later in-flight tokens are excluded by the adapter itself.
        try:
            self.after_completed_step(request, manager, processed_step_seq=request.last_sched_seq)
        except TTLRejected as exc:
            self._publication_failed(request, exc)
            return
        reservation = self.controller.requests.get(owner)
        if reservation:
            for key in reservation.published:
                self.publications[owner][self.plans[owner].ends[key]] = self.ttls[owner][key]

    def _publication_failed(self, request, exc):
        # Expected retention failures must not abort generation/EngineCore.
        # Pool ownership/refcount invariants deliberately remain fatal.
        self.cancel_admission(request)
        self.controller.stats['publication_failed'] += 1
        logger.warning("TTL retention cancelled for one request: %s", exc)

    def finished(self, request):
        owner = request.request_id
        if owner not in self.plans:
            return
        reason = str(request.get_finished_reason())
        # Abort/error must not claim a completed reservation. Already published
        # cache remains retained, as for a cancelled request after prefill.
        if reason in ('stop', 'length'):
            try:
                result = self.finish(request)
            except TTLRejected as exc:
                self._publication_failed(request, exc)
                return
        else:
            self.cancel(request)
            result = {'retention': 'cancelled'}
        result['published'] = sorted(self.publications.pop(owner, {}).items())
        result['block_size'] = self.contract.block_size
        result['finished_at'] = self.controller.last_time
        self.ttls.pop(owner, None)
        self.receipts[owner] = result
        while len(self.receipts) > self.controller.max_reservations:
            self.receipts.popitem(last=False)

    def cancel_admission(self, request):
        self.cancel(request)
        self.publications.pop(request.request_id, None)
        self.ttls.pop(request.request_id, None)

    def take_receipt(self, owner):
        self.tick()
        return self.receipts.pop(owner, None)

    def tick(self):
        super().tick()
        if hasattr(self, 'receipts'):
            for owner, receipt in list(self.receipts.items()):
                if receipt['finished_at'] + 300 <= self.controller.last_time:
                    del self.receipts[owner]

    def can_reset(self):
        self.tick()
        # Reset may be retried after TTL expiry. Never silently break a lease.
        return not self.controller.leases and not self.plans


def admit_core(core, prepared):
    runtime = getattr(core.scheduler, 'native_ttl', None)
    if runtime is None:
        return {'ok': False, 'status': 400, 'message': 'Native TTL is not enabled'}
    native = None
    owned = False
    try:
        native, wave = prepared
        if native.request_id in core.scheduler.requests:
            raise TTLRejected('Duplicate engine request ID')
        owned = True
        try:
            runtime.admit_wire(native)
        except TTLCapacityRejected:
            plan = (native.sampling_params.extra_args or {}).get(PLAN_KEY, {})
            if plan.get('strict') is not False:
                raise
            # Claude Code's ordinary ephemeral hints are opportunistic locally.
            # Preserve generation/APC when bounded retention is full. Explicit
            # ttl always keeps strict admission; no false creation receipt.
            runtime.controller.stats['opportunistic_bypassed'] += 1
        core.add_request(native, wave)
        return {'ok': True}
    except Exception as exc:
        if owned:
            if native.request_id in core.scheduler.requests:
                core.abort_requests([native.request_id])
            runtime.cancel_admission(native)
        if isinstance(exc, (TTLRejected, ValueError, KeyError, TypeError)):
            return {'ok': False, 'status': 429 if isinstance(exc, TTLCapacityRejected) else 400,
                    'message': str(exc)}
        return {'ok': False, 'status': 500, 'message': 'Native TTL admission failed'}
