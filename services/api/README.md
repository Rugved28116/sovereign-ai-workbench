# Sovereign API

This service is the first local backend slice: FastAPI, registry validation, sovereign model eligibility, deterministic routing, and a development-only mock provider.

## Setup

Requires Python 3.12 and `uv`.

```bash
uv sync
```

## Run

```bash
SOVEREIGN_ENV=development uv run uvicorn sovereign_api.main:app --reload
```

## Test

```bash
uv run pytest
```

## Environment

`SOVEREIGN_ENV` is mandatory and accepts exactly `development`, `on-prem`, or `air-gapped`. Missing or unknown values fail application startup.

`POST /v1/generate` accepts prompts up to 32,768 characters. Callers may explicitly supply 1–16 unique `required_capabilities`, with capability names up to 64 characters; explicit capabilities bypass automatic classification. When the field is omitted, deterministic classifier rules infer one or more required capabilities. Empty prompts, explicit `null`, empty capability lists or names, and duplicate capabilities are rejected. The endpoint also has a 256 KiB request-body limit.

## Deterministic task classification

Automatic classification is provider-neutral and uses normalized prompt text with primary-class precedence: `vision` → `coding` → `document` → `reasoning` → `general`. A frozen task-requirements value can contain one or more capabilities. Explicit combination rules cover document/vision/reasoning, vision/coding, document/coding, vision/reasoning, document/reasoning, document/vision, and coding/reasoning requests; signals are not blindly unioned. Ambiguous requests default to `general`/`chat`; classification never selects a model or provider and cannot bypass Stage 1 eligibility or environment restrictions.

Inferred capabilities use AND semantics and are emitted in deterministic `document`, `vision`, `coding`, `reasoning`, `chat` order as applicable. Stage 1 requires one eligible model to support all of them; capabilities are never dropped, and decomposition across multiple models is not supported. Explicit client capabilities remain authoritative, retain caller order, and bypass inference entirely.

The classifier is currently a conservative keyword-and-phrase ruleset. It performs no model calls, embeddings, semantic routing, network access, or learned scoring. In `development`, response routing metadata identifies whether capabilities were `explicit` or `inferred` and includes the inferred task class and required capabilities. These classifier details are omitted from successful responses in other environments.

## Task requirement planning

`TaskRequirements` describe what the current request needs and continue to route through Stage 1 with AND semantics. For inferred requests, the deterministic `TaskRequirementPlanner` also creates an immutable `TaskPlan` describing how those capabilities may later be decomposed into ordered `generate`, `code`, `document`, `vision`, or `reason` stages with deterministic `stage-N` IDs.

Plans are metadata only and are not executed. `/v1/generate` still routes the complete current requirement set to one model; it does not invoke stages, split work across models, or weaken capabilities. Development responses expose inferred plans for inspection, while non-development responses omit them. Any future stage execution must independently re-enter sovereign eligibility and routing before a model is selected. Unsupported or non-canonical capability sequences fail closed rather than being dropped, invented, or reordered.

## Agent execution state

The provider-neutral execution domain can now convert a `TaskPlan` into an immutable `AgentTask` snapshot containing ordered `AgentStep` values. Task and step state changes use validated transition methods that return new snapshots, with explicit caller-supplied timestamps and a revision incremented exactly once per transition. Results are limited to immutable text or serialized structured content, and errors are valid only on failed steps. Invalid transitions and inconsistent terminal states fail with typed execution-state errors. When a step fails, the task fails atomically: other running steps become `cancelled`, pending steps become `skipped`, and completed terminal steps are preserved.

The domain exposes a compare-and-swap validation contract that rejects an expected revision which does not match the current snapshot. It performs no storage operation and cannot prevent stale writes by itself; future persistence must compare the expected and current revisions and commit the replacement atomically. This is a state model only. No agent stages, models, tools, files, or network operations are executed, and `/v1/generate` is unchanged. A future orchestrator may operate through these values, but each eventual stage must independently pass normal sovereign eligibility and routing before execution.

## Minimal task orchestration

The provider-neutral `SequentialTaskOrchestrator` can now process a `TaskPlan` in order through a pluggable asynchronous `StageExecutor`. It uses only the latest immutable `AgentTask` returned by the execution-state transition API, so each task or step transition advances the task revision without direct field mutation. Known executor failures become failed task snapshots using the existing atomic failure disposition; unexpected exceptions receive a fixed safe error message, and execution stops immediately.

`MockStageExecutor` supplies deterministic, network-free results and optional deterministic failure injection for development and tests only. No public orchestration endpoint, persistence, parallel execution, live cancellation, or tool execution is included, and `/v1/generate` remains unchanged.

`ModelStageExecutor` is an explicit alternative that routes every stage independently through the existing Stage 1 sovereign eligibility filter and Stage 2 optimizer, then invokes the approved provider through `ModelProvider`. Composition supplies the configured router, provider mapping, and a frozen `TaskExecutionInput` for each task run. The original prompt uses the same whitespace normalization and 1–32,768 character limits as generation requests.

Stage context contains the original prompt, ordered earlier successful results labelled with stage ID/type, and the current stage type. Current and future results are excluded. The entire composed context is limited to 65,536 characters including labels; overflow fails with a typed stage error before routing, without truncation. This is a character bound, not a tokenizer-specific context-window guarantee. Responses become text `StepResult` values without adding model/provider metadata. Provider content is preserved as returned (including the existing mock provider's synthetic text).

Expected routing failures, unavailable providers, provider exceptions, and malformed responses become a fixed safe `StageExecutionError`; the orchestrator then fails the task and dispositions remaining work. Unexpected router or optimizer defects propagate to the orchestrator's separate, sanitized unexpected-failure path. There are no retries or fallback providers. This executor adds no direct network implementation: future eligible runtime adapters use their existing transport and sovereignty controls. No runtime or model is installed, and the shared registry is unchanged.

## Internal execution provenance

Internal routed-stage provenance is stored in optional immutable `AgentStep.provenance`. Successful model executions return a `StageExecutionRecord` containing the unchanged `StepResult` and separate `ExecutionProvenance`; routed failures carry provenance through the existing safe failure path. The orchestrator preserves these facts through normal state transitions. The record contains the approved model/provider identity, stage ID, exact requested capabilities, routing environment, timezone-aware start/completion timestamps, success flag, and successful-result digest. It contains no prompt, response payload, endpoint, configuration, or exception details.

The digest is SHA-256 over UTF-8 JSON containing `content` and `output_type`, with sorted keys, compact separators, and unescaped Unicode. Content strings are preserved exactly, including serialized structured content; this does not normalize semantically equivalent structured strings. Failed executions have no digest. No provenance is fabricated when routing fails before approval, or for skipped stages and `MockStageExecutor`. Model executor clocks can be injected for deterministic tests, and completion before start is rejected. Provenance is internal only: it is neither persisted nor exposed through a public API. A future audit subsystem can consume it; hashes prepare for integrity checks and are not encryption.

## Two-stage routing

Routing first applies the mandatory, fail-closed sovereign eligibility filter. Only models approved for enablement, the exact active environment, mock-provider restrictions, and all requested capabilities reach the Stage 2 optimizer.

Stage 2 uses a provider-neutral optimizer interface over immutable candidate projections; registry models and provider configuration are not exposed to it. Its current deterministic implementation selects by ascending numeric priority and then ascending lexicographic model ID, preserving existing behavior. Optimizer output is checked against a private snapshot of the exact Stage 1-approved state; returning an unknown, rejected, or altered candidate raises a typed routing error instead of falling back or bypassing eligibility. No learned classification, semantic routing, telemetry, scoring, or load balancing is implemented.

### Local OpenAI-compatible provider

Registry entries may use the provider key `local-openai-compatible`. When an enabled entry is eligible for the active environment, provider-private configuration must set:

```bash
SOVEREIGN_LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

The URL may use HTTP or HTTPS with exact `localhost`, IPv4 loopback, RFC1918 private IPv4, or IPv6 loopback. Exact `localhost` is converted to `127.0.0.1` before requests, so accepted endpoints never depend on DNS or hosts-file resolution. Public IP addresses, other DNS hostnames, control characters, and malformed URLs are rejected during configuration. Configuration is not stored in `models/registry.yaml`, redirects are not followed, and environment proxy settings are ignored.

Provider responses are streamed and limited to 4 MiB, including unsuccessful responses. Oversized declared or streamed bodies fail with a typed provider response error. No local runtime or model is installed by this provider adapter.

The generic `local-openai-compatible` provider remains supported for backward compatibility with local compatible servers.

### vLLM provider

Registry entries may use the runtime-specific provider key `vllm`. When an enabled vLLM entry is eligible for the active environment, set its provider-private endpoint configuration with:

```bash
SOVEREIGN_VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

The vLLM adapter reuses the same sovereign endpoint validation, DNS avoidance, proxy and redirect controls, and 4 MiB streamed response limit described above. This milestone implements only the basic text-generation protocol adapter; it does not install a vLLM server or model, and it does not support client streaming, tools, vision, embeddings, or model management.

## Current limitations

Only deterministic routing, the offline mock provider, and local HTTP protocol adapters are implemented. The supplied mock models are `development`-only, so generation fails closed in `on-prem` and `air-gapped`. No real inference runtime or model is included. Authentication, orchestration, retrieval, tools, and other later architecture are out of scope.
