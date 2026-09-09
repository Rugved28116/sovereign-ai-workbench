# ADR 0001: Foundational Architecture

- **Status:** Accepted
- **Date:** 2026-09-08
- **Decision owners:** Sovereign AI Workbench maintainers
- **Scope:** Initial product architecture; no application implementation is authorized by this ADR

## Context

Sovereign AI Workbench must provide model-agnostic, local-first AI workflows for confidential enterprise and industrial work. It must operate in `development`, `on-prem`, and `air-gapped` environments without making internet access, hosted AI services, external telemetry, or cloud control planes runtime requirements.

The GitHub landscape research supports reusing mature local data-plane components while retaining ownership of the sovereign control plane. Inference engines, workflow mechanics, document parsers, vector search, identity infrastructure, policy evaluators, and file-format libraries can reduce implementation risk, but none of them supplies the complete environment eligibility, authorization, audit, provenance, and zero-egress guarantees required by this project.

This ADR turns that finding into an initial component model. It preserves the existing dependency rule: business logic depends on project-owned, provider-neutral contracts, never on a model family, provider SDK, inference runtime, vector database, agent framework, or sandbox engine.

## Decision Summary

The project will build and own its sovereign control plane and will reuse mature local components behind narrow interfaces. `WRAP` means an upstream component is used only through a project-owned abstraction. `REUSE` means an upstream implementation is selected as a replaceable infrastructure primitive. `EVALUATE LATER` records an intentional deferral, not an implicit dependency.

| Component | Decision | Technology/Approach | Why |
|---|---|---|---|
| Workbench frontend | BUILD | Purpose-built browser application | Product workflows, governance, provenance, and security UX should not inherit the coupling or licensing constraints of a generic local ChatGPT UI. |
| Application API | BUILD | FastAPI for the first implementation slice, behind application contracts | Establishes a local API boundary while authorization remains server-side and framework-specific types stay out of domain contracts. |
| Model provider abstraction | BUILD | Provider-neutral request, response, streaming, capability, error, and usage contracts | Keeps business and orchestration logic independent of model vendors, model families, and serving engines. |
| Model registry | BUILD | Project-owned declarative registry | Exact environment eligibility, status, capabilities, deterministic priority, and neutral routing metadata are sovereign control-plane concerns. |
| Sovereign eligibility filter | BUILD | Deterministic, fail-closed Stage 1 router | Security and deployment restrictions must be authoritative, explainable, and unaffected by probabilistic optimization. |
| Advisory model optimizer | BUILD | Pluggable Stage 2 scoring with deterministic fallback | Quality, latency, resource, specialization, context, and load signals are useful only within the eligible candidate set. |
| Primary GPU inference | WRAP | vLLM behind a dedicated provider adapter | Reuses mature GPU serving while containing telemetry, runtime configuration, wire-format, and model-specific behavior. |
| Lightweight/local inference | WRAP | llama.cpp behind a separate provider adapter | Supports CPU, edge, GGUF, and constrained deployments without leaking llama.cpp concepts into shared contracts. |
| Additional inference runtimes | EVALUATE LATER | Benchmark candidates such as SGLang | A third runtime is not justified until representative local workloads show material benefit. |
| Agent orchestration | WRAP | LangGraph mechanics behind a project-owned orchestrator abstraction | Reuses graph, checkpoint, retry, resume, and interrupt mechanics without delegating routing, authorization, audit, or durable application state. |
| Document intelligence | WRAP | Docling in a constrained document-processing worker | Reuses structured conversion while enforcing local artifacts, processing limits, provenance, and no remote services. |
| Specialist OCR | WRAP | PaddleOCR and OCRmyPDF through modular processor adapters | Provides replaceable OCR and searchable-PDF capabilities without creating a competing business-facing document model. |
| Governed retrieval and citations | BUILD | Project-owned retrieval pipeline and canonical provenance contracts | Authorization, ACL-safe filtering, composition, citation lineage, and verification must not be delegated to a storage engine. |
| Vector database | REUSE | Hardened, locally operated Qdrant behind a vector-store interface | Supplies mature dense, sparse, hybrid, and metadata-filtering primitives without becoming an agent-facing API. |
| Reranking | WRAP | Eligible local reranker through provider or dedicated worker adapters | Reranking models remain governed by the registry and local model contracts. |
| Tool permission framework | BUILD | Capability manifests, grants, policy enforcement points, and approval service | Tool descriptions and model output are not authorization; least privilege and approvals are product responsibilities. |
| Tool protocol compatibility | EVALUATE LATER | MCP-style contracts and selected local protocol/SDK pieces | MCP may improve interoperability, but server metadata is not authority and cloud-hosted or runtime-installed servers are out of scope. |
| Sandbox broker | BUILD | Project-owned admission, workspace, resource, network, secret, lifecycle, and audit boundary | Isolation engines do not supply end-to-end authorization or safe orchestration. Agents must not control the host runtime. |
| Sandbox isolation runtime | EVALUATE LATER | gVisor as the leading initial candidate; preserve a pluggable backend | gVisor is promising, but compatibility and threat-model validation should precede adoption. |
| Policy evaluation engine | EVALUATE LATER | Local OPA proof of concept behind project-owned policy interfaces | OPA may externalize policy evaluation, but the application must own policy semantics and enforce every decision. |
| Authentication infrastructure | EVALUATE LATER | Locally operated Keycloak for identity, sessions, federation, and coarse RBAC | Mature identity can be reused later; resource and action authorization remain inside application boundaries. |
| Artifact serialization | REUSE | Local `python-docx`, `python-pptx`, and `openpyxl` libraries | Mature offline file-format implementations avoid rebuilding serializers. |
| Artifact governance | BUILD | Templates, validation, policy, approvals, provenance, and release controls | Format libraries do not provide trustworthy content governance or traceability. |
| Audit and provenance | BUILD | Local append-oriented correlated event model and governed storage | Cross-component lineage, redaction, integrity, retention, and offline export are core product requirements. |
| Offline provisioning | BUILD | Mirrored and verified models, packages, container images, frontend assets, and dependencies | Air-gapped installation and operation require a product-owned promotion and verification process. |

## Architectural Boundaries

### Business Logic, Router, Provider, and Runtime

The conceptual dependency and call boundary is:

```text
Business/application logic
        |
        v
Provider-neutral model request and capability contracts
        |
        v
Model router (eligibility, then optimization)
        |
        v
Provider contract
        |
        v
Concrete provider adapter
        |
        v
Locally operated inference runtime
```

- **Business logic** describes the required capability and constraints. It does not select vLLM, llama.cpp, a model family, a device, or a runtime-specific request format.
- **The model router** selects only from models admitted by the registry and trusted policy inputs. It produces an explainable route decision and invokes a provider through the internal provider contract.
- **A model provider** implements the project-owned contract, normalizes responses and errors, and translates requests for one runtime. Provider identity is infrastructure configuration, not business behavior.
- **An inference runtime** loads and executes local model artifacts. It is an internal data-plane service and must not be directly exposed to users or agents.

OpenAI-compatible endpoints, where an engine provides them, are only adapter-level wire protocols. They do not justify importing hosted-provider SDKs or assuming identical tool calling, templates, tokenization, structured output, multimodal behavior, or security semantics.

vLLM is the primary GPU runtime. Its adapter and deployment profile must use pre-staged artifacts, disable usage reporting and remote-code trust, prevent runtime downloads, require internal service authentication where applicable, and run under enforced egress restrictions. llama.cpp is a separate adapter for lightweight, CPU/edge, GGUF, or otherwise constrained deployments; it must use pinned local artifacts and offline operation. Neither is a routing authority or a public API.

### Model Registry

The project owns the shared model registry. Each entry must declaratively contain:

- stable model ID and provider key;
- status, including whether it is enabled;
- exact eligible deployment environments;
- provider-neutral capabilities and modalities;
- priority, where lower numeric values win;
- provider-neutral context and hardware compatibility constraints; and
- routing metadata needed for deterministic selection or advisory scoring.

The canonical environment identifiers are `development`, `on-prem`, and `air-gapped`. Eligibility requires an exact match; missing, unknown, or unauthorized environment configuration fails closed. Mock models are eligible only for `development`.

Equal-priority candidates are ordered by ascending lexicographic model ID. This `(priority, model_id)` ordering is the deterministic baseline and fallback when advisory signals are absent, disabled, invalid, or tied.

Runtime endpoints, command-line flags, tensor parallelism, device mappings, runtime-specific quantization settings, credentials, model server authentication, mock fixture responses, and similar provider details must remain in provider-private configuration. The shared registry may express neutral requirements such as capability, context length, accelerator class, memory class, or artifact reference; it must not become a runtime configuration file.

### Two-Stage Model Routing

Routing has two ordered stages.

#### Stage 1: Sovereign Eligibility Filter

Stage 1 is a trusted, deterministic, fail-closed policy enforcement point. It excludes any candidate that does not satisfy all of the following:

1. enabled status;
2. exact active deployment environment;
3. every required capability and modality;
4. caller, task, data, and model permissions or policy;
5. hardware compatibility and locally available approved artifacts; and
6. applicable security restrictions, including data classification, locality, license, and runtime constraints.

Missing or indeterminate facts result in exclusion, not permissive fallback. The filter records structured exclusion reasons for audit without exposing sensitive configuration to the caller. Disabled and ineligible models never reach Stage 2.

#### Stage 2: Advisory Optimizer

Stage 2 ranks only the candidates returned by Stage 1. It may consider:

- measured or evaluated quality;
- predicted latency;
- resource usage and availability;
- request context requirements;
- task or modality specialization; and
- current load and health.

Optimization is advisory. It cannot restore an excluded candidate, weaken a permission, alter environment eligibility, or bypass a security restriction. Learned or model-based signals are untrusted ranking inputs, never authorization inputs. The final decision, candidate set, signal versions, fallback behavior, and reason must eventually be auditable. Initial development uses only deterministic `(priority, model_id)` ordering; adaptive optimization is deferred until measurements, evaluation data, and rollback controls exist.

### Agent Orchestration

LangGraph will initially supply graph execution, typed state transitions, retries, checkpoint/resume, and interrupt mechanics, subject to a focused local proof of concept and dependency review. It will be wrapped behind a project-owned orchestrator contract so workflows, business services, policy decisions, model routing, audit events, and durable task state do not depend on LangGraph types.

An interrupt is a control-flow event, not permission. Human approval must be issued and validated by the trusted permission framework, scoped to the actor, task, operation, arguments or resource, policy version, and expiry. Checkpoints remain local, versioned, and must not deserialize untrusted executable objects.

### Document Intelligence

Docling will be the primary reusable document conversion component where its formats and structured representation fit. PaddleOCR and OCRmyPDF may be selected through modular processor interfaces for specialist OCR and searchable-PDF preprocessing. Processors run as constrained, non-root workers with pre-staged models and binaries, no network access, bounded file/page/time/memory/output limits, and read-only inputs.

The application owns a canonical, processor-neutral document representation and provenance contract. It should retain source hashes and versions, access-control labels, page and region coordinates, reading order, table cells, images, OCR confidence, and processor/model versions where available. Specialist processor output maps into this representation rather than defining parallel domain models.

### Knowledge and RAG

Qdrant is the initial vector database and must be locally operated behind a repository-neutral storage interface. Its deployment profile must disable telemetry and unnecessary integrations, restrict network binding, enable authentication, and use TLS when traffic crosses a trust boundary.

Agents never query Qdrant directly. A project-owned governed retrieval layer performs authorization and ACL-safe metadata filtering as part of the query, followed by dense and/or sparse retrieval, hybrid fusion, local reranking, citation assembly, and verification as appropriate. Provenance must survive ingestion, chunking, embedding, retrieval, reranking, and answer generation so citations can identify the authorized source version, page, region, table cell, or chunk. Post-search ACL filtering alone is insufficient.

### Sandbox Architecture

The project will build a sandbox broker as a trusted control-plane service. Agents and tools request a declared operation; the broker validates policy and approval, chooses an isolation backend, prepares a narrowly mounted ephemeral workspace, applies limits, starts the workload, captures bounded results and audit events, and verifies teardown.

Every sandbox must use non-root execution where practical, an immutable or read-only base, explicit filesystem mounts, CPU/memory/PID/time/output limits, and an ephemeral writable workspace. Network access defaults to disabled. Any future network-enabled profile requires an explicit destination allowlist, permission, observability, and incompatibility with `air-gapped` deployment where it implies external access.

Agents must never receive direct or indirect access to `/var/run/docker.sock`, containerd sockets, host runtime credentials, arbitrary host paths, devices, or unrestricted shell execution. gVisor is the leading later isolation candidate, but it does not replace the broker, policy enforcement, resource controls, or network policy. Its compatibility, performance, host integration, and threat-model fit must be validated before adoption.

### Policy and Permissions

The application owns the authorization domain model, tool capability manifests, grants, approval records, and policy enforcement points at the API, router, retriever, tool broker, sandbox broker, and artifact release boundary. Sensitive or indeterminate operations default to deny. Prompts, model output, tool annotations, workflow configuration, and protocol negotiation cannot grant authority.

OPA will be evaluated as a local policy decision point when policy complexity justifies it. If adopted, policies and data must be local, versioned, tested, integrity-checked, and unavailable or undefined decisions must fail closed. OPA evaluates; application-owned enforcement points enforce. Human approval gates must support scoped, attributable, expiring decisions and cannot be represented only as an orchestrator pause.

### Authentication

Keycloak will be evaluated later as locally operated identity, session, federation, MFA, and coarse RBAC infrastructure. Enterprise SSO integration is not part of initial development. Whether or not Keycloak is adopted, the application validates identity at its trusted boundary and owns authorization for models, tools, knowledge, sandboxes, resources, and artifacts. Authentication claims are inputs to authorization, not authorization decisions by themselves.

### Tool Protocol

The project will study MCP-style, provider-neutral schemas, capability negotiation, and local process transports. Any support must sit beneath the project-owned tool contract, capability manifest, permission framework, secret broker, sandbox policy, and audit path. Only pinned and approved local servers may be enabled; automatic marketplace installation, runtime package downloads, ambient credentials, and cloud-hosted MCP services cannot be runtime dependencies. Remote transports are disabled unless explicitly allowed by deployment policy, and external ones are prohibited in `air-gapped` mode.

### Artifact Generation

Local libraries such as `python-docx`, `python-pptx`, and `openpyxl` will provide DOCX, PPTX, and XLSX serialization. They remain replaceable implementation details behind project-owned artifact contracts. The project builds the template catalog, schema validation, content and malware checks, deterministic post-generation validation, provenance, access policy, approval, storage, and release workflow. Generated artifacts are untrusted until validation and policy checks complete.

### Audit and Provenance

Audit and provenance are project-owned. Every significant model request and route, retrieval event, tool execution, policy decision, human approval, sandbox lifecycle event, verification result, and generated artifact should eventually emit a correlated local event. Events should identify actor, task, component and version, policy/configuration version, timestamps, outcome, and hashes or references needed for lineage.

Audit storage should be append-oriented and integrity-protected. Redaction, access control, retention, and export rules must avoid storing secrets or confidential payloads by default while preserving enough evidence for investigation and reproducibility. No external exporter or telemetry destination is enabled by default, and none is required.

### Frontend

The project will build its own browser workbench around governed tasks, workspaces, sources, approvals, artifacts, route explanations, and audit visibility. Existing local AI interfaces may inform UX research, but a generic ChatGPT-like UI will not be forked as the core architecture. The frontend is never the sole authorization or validation boundary.

### Air-Gapped Deployment

Offline installation and operation are first-class design targets. Runtime operation must not require DNS, package repositories, model hubs, CDNs, license servers, update services, hosted identity, hosted telemetry, or any other internet endpoint. Application flags such as “offline mode” are defense-in-depth settings, not network boundaries.

The project must eventually support controlled offline provisioning of model weights and tokenizers, OCR assets, Python and frontend packages, container images, policies, templates, licenses, SBOMs, and all static frontend assets. A connected staging process may mirror approved inputs, but promotion to a disconnected environment must verify hashes/signatures, licenses, vulnerability decisions, and completeness. Runtime downloads and auto-installers are prohibited in `on-prem` and `air-gapped` production profiles; deployment-enforced egress denial remains required.

## Request Flow

The intended future request path is:

```text
User
  -> API
  -> Authentication/Authorization
  -> Task classification
  -> Sovereign eligibility filter
  -> Model optimization/routing
  -> Agent orchestrator
  -> Tools / Knowledge / Sandbox
  -> Verification
  -> Artifact/output
  -> Audit trail
```

The API authenticates the user and establishes trusted task context. Task classification proposes required capabilities but cannot grant permissions. The sovereign filter admits models before the optimizer ranks them. The orchestrator may coordinate further model calls and governed capabilities, but every subsequent call re-enters the appropriate authorization, routing, retrieval, tool, or sandbox boundary. Verification checks outputs and artifacts against task-specific rules. Audit events are emitted throughout the flow; “Audit trail” represents final correlation and persistence, not a single end-of-request logging action.

## Trust Boundaries

All data crossing these boundaries is validated and treated as untrusted unless a stronger guarantee is explicitly established.

| Boundary | Trust decision and required controls |
|---|---|
| User to application | Authenticate locally, authorize server-side, validate and bound inputs/uploads, rate and resource limit, and never treat user or prompt claims as authority. |
| Application control plane to model runtime | Use an authenticated internal provider adapter; allow only eligible models and bounded requests; do not expose runtimes publicly; treat model output and model artifacts as untrusted. |
| Application/orchestrator to tools | Resolve a pinned tool identity and manifest, validate arguments and resource scope, enforce least privilege and approvals, inject only scoped secrets, bound execution, and audit the result. |
| Application/tool broker to sandbox | Only the trusted broker launches workloads; apply an approved image, mounts, limits, identity, network policy, and teardown; the sandbox cannot control its broker. |
| Sandbox to host operating system | Use a replaceable isolation runtime, non-root identity, syscall/filesystem/process/resource restrictions, no host runtime sockets or devices, explicit mounts, and defense-in-depth host controls. |
| Application/retrieval layer to knowledge stores | Enforce tenant/resource ACL filters within retrieval, authenticate storage access, preserve provenance, limit result data, and never expose database credentials or APIs to agents. |
| Document/artifact workers to application and stores | Treat parsers, documents, generated files, and metadata as hostile; use isolated workers, quotas, validation, scanning, controlled storage, and provenance. |
| Model runtime, tools, sandbox, and stores to external networks | Default to no egress; permit only explicit local or allowlisted destinations under policy; prohibit external access in air-gapped operation. |

The application control plane is trusted to enforce policy but is not assumed immune to malformed inputs. Model runtimes, document processors, tool processes, and sandboxes are separate, less-trusted data-plane components. The host operating system and operator-controlled infrastructure form the final administrative boundary; agents receive no host authority.

## Explicit Non-Goals for Initial Development

- Kubernetes.
- Distributed inference orchestration.
- Training our own foundation model.
- Multi-node cluster management.
- Cloud integrations.
- Enterprise SSO.
- Full production hardening.

These non-goals limit the initial implementation scope; they do not relax deny-by-default behavior, environment filtering, secret hygiene, local-only operation, or other foundational security controls.

## First Implementation Slice

The first implementation slice, to be implemented in a later change, is:

```text
Browser
  -> FastAPI
  -> Model Router
  -> Mock Provider
  -> Mock Models
```

It contains only:

- a health endpoint;
- model registry loading and validation;
- exact environment filtering with fail-closed unknown or missing environments;
- deterministic routing by ascending numeric priority and then ascending lexicographic model ID;
- deterministic mock inference through the provider-neutral contract; and
- tests, including negative cases for disabled, incapable, non-development, missing-environment, and unknown-environment models.

Mock models and the mock provider are development-only and must never route in `on-prem` or `air-gapped`. Stage 2 is initially the deterministic baseline rather than a learned optimizer. This ADR does not implement or add dependencies for the slice.

## Consequences

### Positive

- Business behavior remains independent of vLLM, llama.cpp, LangGraph, Qdrant, and other replaceable infrastructure.
- Eligibility and permissions remain deterministic, fail closed, and auditable before advisory optimization or agent action.
- Mature local components reduce implementation effort without transferring sovereign governance to upstream projects.
- Control-plane services can evolve independently from less-trusted inference, parsing, retrieval, tool, and sandbox data planes.
- Air-gapped packaging, telemetry denial, and runtime download prevention are architecture requirements from the outset.

### Costs and Risks

- Project-owned abstractions, governance, route explanations, provenance, and offline promotion require meaningful engineering and testing.
- Wrappers introduce compatibility work when upstream APIs or model-specific semantics change.
- LangGraph, Docling, Qdrant, vLLM, llama.cpp, OPA, Keycloak, gVisor, OCR engines, models, and file libraries each require license, supply-chain, upgrade, and offline-package review before adoption.
- Single-node initial designs will need measured migration criteria if future scale requires distributed operation.
- `on-prem` does not automatically mean zero egress; deployment policy must explicitly define and enforce permitted local network paths.

## Reconciliation Notes

This decision follows the research recommendation to reuse mature data-plane engines and own the sovereign control plane. It deliberately strengthens several upstream defaults: vLLM usage reporting and Qdrant telemetry must be disabled; all runtime downloads must be removed from production profiles; Docling and OCR processing must be isolated and bounded; LangGraph interrupts are not approval decisions; Qdrant filtering does not replace application authorization; OPA is not an enforcement point; MCP declarations are not permissions; and gVisor alone is not a complete sandbox.

The research recommends adopting gVisor and OPA and adopting Keycloak for their respective narrow roles. This accepted decision intentionally defers all three until a threat model, integration boundary, offline dependency review, and focused proof of concept justify adoption. Deferral changes timing, not the research conclusion that they are leading candidates. Similarly, LangGraph is selected only behind a wrapper and remains conditional on a local proof of concept.

The canonical environment names in this ADR follow `AGENTS.md`: `development`, `on-prem`, and `air-gapped`. Foundational documentation uses these names consistently. The current registry contains only `development` entries, so it does not require modification for this decision.

This accepted ADR is the decision point for the selections above. The roadmap and architecture overview are aligned with it, and architecture decision records are standardized under `docs/adr/`.

## Unresolved Architectural Questions

The following require focused follow-up decisions or proofs of concept; none blocks the documented first slice:

1. What exact provider-neutral request, streaming, usage, tool-call, multimodal, embedding, and reranking contracts are needed for the first real-runtime adapter?
2. What evidence and acceptance criteria must the LangGraph proof of concept meet, particularly for local checkpoint storage, serialization safety, cancellation, replay, and abstraction leakage?
3. What canonical document representation will preserve Docling, PaddleOCR, and OCRmyPDF provenance without binding the domain model to any one processor?
4. What Qdrant topology, authentication/TLS profile, backup policy, sparse retrieval approach, and tenant/ACL indexing strategy are appropriate for the first on-prem deployment?
5. Which hardware inventory and admission service supplies trustworthy compatibility and load facts to the router without leaking provider-specific settings into the registry?
6. When does policy complexity justify OPA, and what will the application-owned policy input/output schema, failure behavior, and signed local policy lifecycle be?
7. What sandbox threat model and compatibility benchmark would justify adopting gVisor, and what safer interim execution behavior is allowed before a sandbox backend exists?
8. What audit event schema, integrity mechanism, payload-redaction policy, retention model, and local export format meet enterprise traceability without retaining excessive confidential content?
9. What verification gates are required for generated answers and each artifact class before release?
10. What offline bill of materials, signing roots, model-license process, vulnerability exception process, and connected-to-disconnected promotion workflow will be supported?
