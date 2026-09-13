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

Planning does not itself execute stages. `/v1/generate` still routes the complete current requirement set to one model; it does not invoke stages, split work across models, or weaken capabilities. Development responses expose inferred plans for inspection, while non-development responses omit them. The internal one-stage coordinator described below independently re-enters sovereign eligibility and routing for each model stage. Unsupported or non-canonical capability sequences fail closed rather than being dropped, invented, or reordered.

## Agent execution state

The provider-neutral execution domain can now convert a `TaskPlan` into an immutable `AgentTask` snapshot containing ordered `AgentStep` values. Task and step state changes use validated transition methods that return new snapshots, with explicit caller-supplied timestamps and a revision incremented exactly once per transition. Results are limited to immutable text or serialized structured content, and errors are valid only on failed steps. Invalid transitions and inconsistent terminal states fail with typed execution-state errors. When a step fails, the task fails atomically: other running steps become `cancelled`, pending steps become `skipped`, and completed terminal steps are preserved.

The internal `AgentTaskState` projection separately represents future multi-stage control state derived exactly from a `TaskPlan`, including the validated original prompt, immutable capability requirements, current stage, output references, selected model identifiers, and controlled safe error fields. Its `StageExecutionState` values preserve unique plan-stage IDs, order, and requirements, and all task and stage transitions return new snapshots or fail closed with typed transition errors. Stage failure atomically fails its task, while cancelling active work atomically marks both the stage and task cancelled; terminal tasks cannot retain running stages. This projection has no arbitrary metadata or provider configuration, is not persisted or exposed by an API, and does not itself route or execute any stage.

The domain exposes a compare-and-swap validation contract that rejects an expected revision which does not match the current snapshot. It performs no storage operation and cannot prevent stale writes by itself; future persistence must compare the expected and current revisions and commit the replacement atomically. The state model itself executes no stages or tools. The internal one-stage coordinator described below operates through its transitions; `/v1/generate` is unchanged.

## Minimal task orchestration

The provider-neutral `SequentialTaskOrchestrator` can now process a `TaskPlan` in order through a pluggable asynchronous `StageExecutor`. It uses only the latest immutable `AgentTask` returned by the execution-state transition API, so each task or step transition advances the task revision without direct field mutation. Known executor failures become failed task snapshots using the existing atomic failure disposition; unexpected exceptions receive a fixed safe error message, and execution stops immediately.

`MockStageExecutor` supplies deterministic, network-free results and optional deterministic failure injection for development and tests only. No public orchestration endpoint, persistence, parallel execution, live cancellation, or tool execution is included, and `/v1/generate` remains unchanged.

`ModelStageExecutor` is an explicit alternative that routes every stage independently through the existing Stage 1 sovereign eligibility filter and Stage 2 optimizer, then invokes the approved provider through `ModelProvider`. Composition supplies the configured router, provider mapping, and a frozen `TaskExecutionInput` for each task run. The original prompt uses the same whitespace normalization and 1–32,768 character limits as generation requests.

Stage context contains the original prompt, ordered earlier successful results labelled with stage ID/type, and the current stage type. Current and future results are excluded. The entire composed context is limited to 65,536 characters including labels; overflow fails with a typed stage error before routing, without truncation. This is a character bound, not a tokenizer-specific context-window guarantee. Responses become text `StepResult` values without adding model/provider metadata. Provider content is preserved as returned (including the existing mock provider's synthetic text).

Expected routing failures, unavailable providers, provider exceptions, and malformed responses become a fixed safe `StageExecutionError`; the orchestrator then fails the task and dispositions remaining work. Unexpected router or optimizer defects propagate to the orchestrator's separate, sanitized unexpected-failure path. There are no retries or fallback providers. This executor adds no direct network implementation: future eligible runtime adapters use their existing transport and sovereignty controls. No runtime or model is installed, and the shared registry is unchanged.

## One-stage agent-state coordination

`StageExecutionCoordinator.execute_one` advances exactly one next-pending `TaskStage` per call and returns a new immutable `AgentTaskState`. It verifies exact plan-stage identity and order, rejects terminal or already-active tasks, starts the task/stage through state transitions, records the outcome, and completes the task only after every stage has completed. It does not loop, retry, skip stages, or accept model-generated tool calls. Existing `ToolExecutor` and workspace tools are not invoked by this coordinator because the current planner defines only model-oriented stage types.

The pluggable `StageExecutor` contract returns an immutable `StageExecutionResult` containing a stage ID, terminal status, bounded text for separate storage, optional selected model ID, and controlled failure fields. `RoutedAgentStageExecutor` handles the current `generate`, `code`, `document`, `vision`, and `reason` stage labels: it routes each stage's exact capability set through the existing Stage 1/Stage 2 router, then calls the approved `ModelProvider` once. This does not add document/image payload handling or make unavailable capabilities routable.

Successful model-stage text is stored outside `AgentTaskState` through the internal provider-neutral `StageOutputStore` contract. The first implementation, `InMemoryStageOutputStore`, returns an opaque UUID-based reference; state holds only that reference, not model text or a fabricated digest. The store accepts only `text/plain` UTF-8 output, at most 256 KiB encoded bytes per output, with an 8 MiB total text capacity and a 1,024-entry ceiling. Capacity overflow fails closed without eviction. References are checked against the expected task and stage on retrieval. This implementation is process-local and non-durable: restart loses outputs and later stages then fail closed. It is unsuitable for multi-process production coordination; a future durable, ownership-preserving store is required.

The first stage receives the validated original task prompt. A later stage receives a deterministic prompt containing that original prompt, only the immediately preceding completed stage's retrieved output, and the current stage type. The prior output is JSON-quoted and delimited as **untrusted data**, never inserted as a system message or parsed as instructions or tool calls. No arbitrary output reference can be selected by a caller; missing, unknown, cross-task, or cross-stage output blocks execution before provider invocation. Composed prompts obey the existing 32,768-character generation limit and are never silently truncated. A storage failure after provider invocation returns a terminal failed task/stage snapshot, not the unchanged pre-execution state. The 256 KiB output bound does not guarantee every output will fit a later prompt; an oversized composed prompt fails closed. There is still no full-plan loop, retry, public agent endpoint, model-driven tool invocation, or durable output recovery.

Registry model IDs are validated before routing: they must be 1–256 characters without surrounding whitespace or control characters. The routed result and execution-state model-ID fields use the same rule; IDs are never truncated. Once a stage records a selected model ID, its completion/failure transitions may preserve that ID but cannot replace it.

Expected routing/provider failures and malformed provider responses produce safe failed stage/task snapshots. Unexpected executor faults are converted to fixed safe errors without retaining raw diagnostics in state. The coordinator validates and reserves all UTC transition timestamps before calling its executor; a clock failure therefore causes zero provider calls, and no clock access after invocation can discard the outcome. These are logical state-transition timestamps and may precede actual provider completion time. A routed model ID is recorded through the controlled stage transition after routing; there is no provider fallback or direct network implementation in the coordinator. There is no public agent endpoint, persistence, live cancellation, tool-stage planning, or autonomous full-plan execution in this path. Callers must serialize access to a task: without persistence or an atomic revision check, concurrent calls using the same stale snapshot could execute a stage twice. Existing provider adapters retain their own local transport behavior and environment restrictions.

## Internal execution provenance

Internal routed-stage provenance is stored in optional immutable `AgentStep.provenance`. Successful model executions return a `StageExecutionRecord` containing the unchanged `StepResult` and separate `ExecutionProvenance`; routed failures carry provenance through the existing safe failure path. The orchestrator preserves these facts through normal state transitions. The record contains the approved model/provider identity, stage ID, exact requested capabilities, routing environment, timezone-aware start/completion timestamps, success flag, and successful-result digest. It contains no prompt, response payload, endpoint, configuration, or exception details.

The digest is SHA-256 over UTF-8 JSON containing `content` and `output_type`, with sorted keys, compact separators, and unescaped Unicode. Content strings are preserved exactly, including serialized structured content; this does not normalize semantically equivalent structured strings. Failed executions have no digest. No provenance is fabricated when routing fails before approval, or for skipped stages and `MockStageExecutor`. Model executor clocks can be injected for deterministic tests, and completion before start is rejected. Provenance is internal only: it is neither persisted nor exposed through a public API. A future audit subsystem can consume it; hashes prepare for integrity checks and are not encryption.

## Tool contracts and permissions

Tools use provider-neutral immutable descriptors, requests, and results. The registry rejects duplicate and unknown IDs. Request arguments accept JSON values, defensively copied into read-only mappings and tuples; non-finite numbers, custom objects, cycles, and structures exceeding 32 levels or 10,000 value nodes are rejected. Results hold output references or bounded text and safe text fields, never raw exceptions or binary blobs. Callers must supply safe messages and keep credentials out of arguments.

The deterministic policy evaluator defaults to `DENY`. Trusted descriptors declare exact dotted permission identifiers (no wildcards); all must be explicitly granted, and empty declarations are denied. Network tools additionally require `network.access` even in development. Because destination scope is not modeled yet, air-gapped deployments deny all network tools; grants and approval cannot override this prohibition. After all denial checks, destructive tools and high/critical-risk tools supporting writes yield `REQUIRE_APPROVAL`; other permitted tools yield `ALLOW`. Approval is a separate future authorization step, never a permission grant, and no approval workflow exists yet.

Policy uses trusted registry descriptors and deployment/grant inputs, not agent-supplied descriptions or arguments. It evaluates the whole tool conservatively, not individual argument/resource scopes. The internal `PolicyEnforcedToolExecutor` looks up the trusted descriptor, evaluates explicit grants and the active environment, and only then resolves an implementation from a separately registered, immutable `ExecutableToolRegistry`. `DENY` and `REQUIRE_APPROVAL` raise typed outcomes without invoking a tool; approval cannot override denial. Missing or mismatched implementations fail closed without fallback. Known safe tool errors remain typed, while unexpected implementation errors receive a fixed safe execution error. Only `workspace.read_file` and `workspace.write_artifact` are composed for execution; there is no public tool endpoint or agent-state/routing integration.

The first executable tool is the explicitly registered `workspace.read_file`. Set `SOVEREIGN_WORKSPACE_ROOT` to an existing absolute directory before constructing it; there is no default workspace, repository, home, or filesystem root. Execution walks each requested component relative to an already-open verified workspace directory descriptor. Absolute paths, drive-qualified paths, `.`/`..` segments, duplicate separators, backslashes, every symlink component, missing paths, and non-regular files are rejected. Paths are treated as filesystem paths and are never URL-decoded.

`workspace.read_file` requires the explicit `filesystem.read` permission in `development`, `on-prem`, and `air-gapped`. The policy-enforced executor stops on `DENY` or `REQUIRE_APPROVAL` before invoking any tool. The reader accepts exactly `{"path": "relative/file.txt"}`, reads UTF-8 strictly, and permits at most 1 MiB, checking opened-file metadata and reading no more than 1 MiB plus one detection byte. It opens directories and the final file read-only with descriptor-relative `O_NOFOLLOW` operations, verifies regular-file and device/inode identity, and returns bounded inline text. It performs no writes, network access, or subprocess execution. Platforms without the required descriptor-relative no-follow primitives fail closed with a typed safety error. These checks block ancestor replacement during the walk; deployment filesystem isolation remains required for hostile shared workspaces because standard-library operations cannot provide a kernel-level sandbox.

`workspace.write_artifact` writes only beneath a separately configured, existing absolute `SOVEREIGN_ARTIFACT_ROOT`; it never uses the read workspace root as a default. It accepts exactly `{"path": "reports/output.txt", "content": "..."}` and requires explicit `artifact.write` permission in every environment. The current medium-risk write policy allows it when that permission is granted; `DENY` and `REQUIRE_APPROVAL` still cause no write. Content is strict UTF-8 and at most 1 MiB after encoding. Parent directories must exist; the tool creates no directories. The root and existing parent directories are opened relative to verified directory descriptors without following symlinks; the final target is checked without following symlinks. Existing regular files are atomically replaced through an exclusive, randomly named temporary file in the same parent directory; existing symlinks, directories, and special files are rejected. It does not append, expose a host path, or offer general filesystem writes. The result contains only a relative artifact reference. No public write endpoint exists.

The artifact root must be a private application-owned directory tree: the root and every traversed parent must be owned by the effective service user and have neither group nor world write permission. These properties are checked from opened directory descriptors at construction and again during writes; shared writable directories such as `/tmp` are invalid artifact roots. Deployment must also protect the root's ancestors from untrusted rename authority and isolate the service identity in hostile multi-user environments. Descriptor-relative traversal prevents path/symlink substitution during the walk, but it cannot stop a process with rename authority from relocating an already-open parent outside the root. Temporary names are random and exclusively created, and the opened temporary inode is checked before replacement, but POSIX pathname rename cannot bind its source to that inode: an attacker with directory write access could swap the temporary name before rename. The private-directory requirement is the mitigation for both races, not a kernel-enforced guarantee against a malicious same-UID process or a compromised host. Such processes are outside this tool's supported trust model.

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

Runtime-facing capabilities remain deterministic routing, the offline mock provider, local HTTP protocol adapters, the bounded workspace file reader, and the scoped artifact writer; the one-stage agent-state coordinator is internal only. The supplied mock models are `development`-only, so generation fails closed in `on-prem` and `air-gapped`. No real inference runtime or model is included. Authentication, public agent/tool integration, retrieval, general write or execution tools, and other later architecture are out of scope.
