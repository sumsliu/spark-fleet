# Operations, ownership and rollback

## Authority belongs to the target operator

This repository carries no standing authorization for any machine. Supply your own authenticated SSH/Docker access and operator-approved scope. Normal user traffic is legitimate. Drain through the existing gateway/service owner before stopping a serving model. Plan refuses a busy GPU or occupied lock; it does not treat those as a reason to kill processes.

A change to one model must not touch another model's nodes, containers, gateways or tasks. Use disjoint topology files and maintenance scopes. Do not run multiple operators' switch workflows at once.

## Four shared locks and guards

The node agent holds these exclusive locks for the release lifetime under `paths.state/locks`:

```
model-switch.lock
phaseB.lock
stepA.lock
llm-switch.lock
```

All operators and launchers must coordinate on the **same files**. If an existing fleet uses different lock paths, integrate the path/lock contract before adopting this controller; an unrelated lock directory does not protect it. v0.1 does not automatically adopt external watchdogs or legacy services.

The configured envelope preserves warning/abort at 85 °C, trip ceiling 90 °C and at least 8 GiB available host/unified memory. The controller intentionally aborts at 85 °C rather than waiting for 90 °C. GPU temperature is used for intervention; platform thermal sensors are recorded for monitoring, not silently reinterpreted as GPU limits. Polling is bounded, not a replacement for a hardware or OS protection mechanism.

Admission records the architecture/GPU, boot ID, Docker image ID, overlay hashes, optional private env-file hash, memory and temperatures. The plan expires in one hour. New launch scopes require new plans; old `used` markers and incident records are never replay templates.

## Lifecycle

`run` holds persistent SSH channels to standalone node agents. Each agent holds four locks, starts one exact named/labeled container and emits guard status. Controller EOF or STOP requests its release stop. Any node fault causes coordinated cleanup. A secondary exact-ownership stop path covers a node-agent crash after Docker creation.

Container ownership requires matching managed/cluster/release labels, observed image ID, entrypoint and command. Stop uses an exact container ID after verification. There is no `docker stop $(docker ps…)`, `pkill`, fleet-wide pruning or “free the GPU” command. A Docker daemon failure is unconfirmed, not evidence that the container is gone.

Private receipts contain release, source, image, boot, container and start identities. `STOPPED.json` reports whether every exact stop was confirmed. Keep the controller foreground under an understood service manager. Do not use a client UI closing as your service shutdown strategy.

## Incident fences

A node fault writes persistent `paths.state/incident.json`. A subsequent run refuses that node. Inspect the exact owned container, thermal/memory condition, control channel and recovery image. Preserve the incident and stop receipts. Once the operator has investigated and confirmed recovery conditions, archive the fence outside the active path and create a fresh plan. There is no automatic fence-clearing or fabricated cooling observation.

If a stop cannot be confirmed, inspect the target locally through existing administration access. Do not assume remote cleanup succeeded during a network partition. Locks and observation alone cannot create hardware fencing across an unreachable host.

## Keep the fastest verified version

Save a private `best-verified.json` containing:

- Complete model recipe and compatible image digest/local IDs.
- Weight revision and manifests; immutable overlay payload/hashes.
- Acceptance, long-context, concurrency, quality and paired performance receipts.
- Known limitations, current route configuration and recovery instructions.

This is an operator-maintained record, not a hidden auto-promotion flag. A candidate replaces it only after a stable positive comparison and correctness/resource checks. Retain the previous image/overlays/weights and enough disk space for recovery.

## A maintenance window

1. Identify the exact model and nodes; announce/drain via your existing process. Wait for active work rather than cancelling it by default.
2. Preserve current recipe, route policy, source/image identities, health and best-verified evidence.
3. Stop the current release through its own manager; check its exact-stop confirmation and cooldown/admission observations.
4. Prepare a fresh plan for the candidate and start the guarded scope.
5. Perform local readiness, API, correctness and benchmark checks while the public route remains disabled or directed to the retained service.
6. Publish the accepted route only when the wider matrix passes. Keep the old recipe recoverable.
7. On failure or uncertain speed benefit, stop only the candidate and recover the best-verified recipe with a **new** plan.

## Gateway and client upgrades

A valid atomic gateway policy reload changes discovery/admission for new requests while active streams drain. Device revocation or expiry deliberately terminates that device's existing transports. Invalid policy fails closed; prepare and validate policy before replacement. A listener or binary change should use a separately checked blue/green process with the existing path retained until connections finish.

Desktop client upgrades are a separate compatibility surface. Keep the original installer/version identity, isolated local integration source, reversible changes and client smoke tests. Do not distribute proprietary app binaries or modify unrelated OS permissions as a troubleshooting shortcut.
