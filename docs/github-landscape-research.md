# GitHub Landscape Research

## Executive Summary

This research used the connected GitHub plugin in read-only mode on 2026-09-08. It inspected repository-owned evidence—READMEs, licenses, dependency manifests, Docker/Compose and configuration files, security documents, source paths, releases, recent commits, and issue/PR activity. Thirty repositories entered the formal discovery funnel and fifteen were shortlisted for detailed analysis. Star counts are a point-in-time activity signal, not a quality ranking. GitHub contributor totals were unavailable through the connected interface; recent commit authors, releases, and issue/PR activity were used instead, and no contributor number is invented.

The central conclusion is to **reuse mature data-plane engines but own the sovereign control plane**:

- **WRAP** [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp) behind the existing provider-neutral model contract; evaluate [SGLang](https://github.com/sgl-project/sglang) only where benchmarks justify another backend.
- **ADOPT behind isolation** [Docling](https://github.com/docling-project/docling), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and [Qdrant](https://github.com/qdrant/qdrant), with all models and artifacts pre-staged and all telemetry disabled.
- **WRAP** [LangGraph](https://github.com/langchain-ai/langgraph) for graph/checkpoint/interrupt mechanics if it passes a focused proof of concept; keep authorization, audit, model routing, and persistence contracts outside it.
- **ADOPT** [gVisor](https://github.com/google/gvisor) as a sandbox isolation primitive and [OPA](https://github.com/open-policy-agent/opa) as an optional policy-decision engine. Neither replaces our trusted execution broker or policy-enforcement points.
- **BUILD OUR OWN** model registry and router, tool/capability authorization, approval service, audit/provenance schema, artifact governance layer, and offline supply-chain/deployment profiles. These are the places where the project’s exact-environment, fail-closed, zero-egress requirements are differentiating.

The most important architectural discovery is that model routing must be split in two. A deterministic trusted stage first enforces environment eligibility, enablement, capability, permissions, locality, artifact availability, and resource constraints. Only then may an advisory stage optimize quality, latency, load, or learned semantic fit among eligible models. [vLLM Semantic Router](https://github.com/vllm-project/semantic-router), [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm), [LiteLLM](https://github.com/BerriAI/litellm), and [RouteLLM](https://github.com/lm-sys/RouteLLM) each contain useful ideas, but none implements the required trusted boundary end to end.

The recurring security finding is equally important: **self-hosted, local-model-capable, and offline are not synonyms for zero egress or safe agent execution**. Examples include default-on usage reporting in [vLLM](https://github.com/vllm-project/vllm/blob/main/docs/usage/usage_stats.md), default-on telemetry in [Qdrant](https://github.com/qdrant/qdrant/blob/master/config/config.yaml), runtime downloads in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), an unauthenticated single-tenant OSS deployment in [OpenHands](https://github.com/OpenHands/OpenHands/blob/main/helm/agent-canvas/README.md), and a privileged Docker-socket sandbox manager in [RAGFlow](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose-base.yml). Zero egress therefore has to be enforced by deployment policy and network isolation, not by application flags alone.

Claim labels used below:

- **VERIFIED**: corroborated in configuration, source, tests, or deployment artifacts.
- **PARTIALLY VERIFIED**: meaningful support exists, but important conditions or gaps remain.
- **CLAIMED ONLY**: found in project documentation without adequate implementation evidence in this pass.
- **NOT SUPPORTED**: inspected evidence contradicts the claim or lacks the required feature.

### Discovery Funnel

The thirty formal candidates were: Open WebUI, AnythingLLM, LibreChat, Dify, PrivateGPT, LocalAI, LangGraph, Haystack, AutoGen, OpenHands, goose, aider, vLLM, llama.cpp, SGLang, Text Generation Inference, LiteLLM, RouteLLM, vLLM Semantic Router, RAGFlow, Qdrant, Weaviate, Milvus, Docling, PaddleOCR, OCRmyPDF, gVisor, Firecracker, MCP, and OPA.

Fifteen were shortlisted below. Near-shortlist projects and narrower supporting dependencies were not ignored; they are not added to the formal candidate count:

| Candidate | Finding and disposition |
|---|---|
| [LocalAI](https://github.com/mudler/LocalAI) | MIT, active, broad local backend and hardware-aware/distributed routing ideas; **STUDY ONLY** because normal flows pull backends/models, authentication defaults off in inspected CLI configuration, and the control plane is much broader than the required provider adapter. |
| [SGLang](https://github.com/sgl-project/sglang) | Apache-2.0, active v0.5.19; OpenAI-compatible text/vision/embedding, structured output, batching, prefix cache, quantization and distributed parallelism are **VERIFIED**. Local cache use is supported, but no complete zero-egress profile or root security policy was found. **DEFER**, then WRAP only if workload benchmarks beat vLLM materially. |
| [LiteLLM](https://github.com/BerriAI/litellm) | Local Ollama/vLLM plus rich retry, fallback, latency/cost/least-busy routing are **VERIFIED**. Core is MIT while `enterprise/` is commercially restricted; its large cloud-provider surface and lack of exact-environment eligibility make it **STUDY ONLY**. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Apache-2.0; valuable threshold calibration/evaluation pattern, but key routers require hosted embeddings, it is mainly two-model routing, and activity stopped in 2024. **STUDY ONLY**. |
| [RAGFlow](https://github.com/infiniflow/ragflow) | Apache-2.0; strong document, hybrid retrieval, reranking, and citation ideas. **STUDY ONLY** because of a large operational stack and unsafe sandbox/deployment defaults. |
| [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) | MPL-2.0; mature local Tesseract/Ghostscript searchable-PDF preprocessing. **ADOPT as an isolated subprocess**, with MPL obligations reviewed if its files are modified. |
| [Firecracker](https://github.com/firecracker-microvm/firecracker) | Apache-2.0; strong microVM isolation. **DEFER** until threat model or density requirements justify KVM, image, networking, and orchestration complexity beyond gVisor. |
| [Zarf](https://github.com/zarf-dev/zarf) | Apache-2.0; disconnected Kubernetes packaging, image/URL rewriting, SBOMs and signing are **VERIFIED**. **ADOPT WITH HARDENING** because signature verification is optional and package actions can execute shell commands in the Zarf process context. |
| [Extism](https://github.com/extism/extism) | BSD-3-Clause; Wasmtime-based plugins with explicit hosts, filesystem preopens, memory/output limits and timeouts. **WRAP/ADOPT** for constrained tools, not arbitrary Python/shell. |
| [Keycloak](https://github.com/keycloak/keycloak) | Apache-2.0; mature self-hosted OIDC/identity. **ADOPT** for authentication/federation, not tool authorization. |
| [OpenBao](https://github.com/openbao/openbao) | MPL-2.0; mature local secrets, PKI, audit devices and integrated Raft storage. **ADOPT as a separately operated secrets service**, subject to license and operational review. |
| [OpenFGA](https://github.com/openfga/openfga) | Apache-2.0; useful relationship authorization. **DEFER** until resource-sharing requirements outgrow application RBAC plus OPA. |
| [Langfuse](https://github.com/langfuse/langfuse) | MIT core with separately licensed enterprise directories; self-hosted tracing is useful, but telemetry is opt-out. **DEFER/STUDY ONLY** until a zero-egress observability profile is proven. |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | MIT and useful local evaluation patterns. Its own security guidance makes clear that telemetry opt-out is not a firewall. **WRAP only in an egress-denied evaluation environment**. |
| [python-docx](https://github.com/python-openxml/python-docx), [PptxGenJS](https://github.com/gitbrent/PptxGenJS), [XlsxWriter](https://github.com/jmcnamara/XlsxWriter), [WeasyPrint](https://github.com/Kozea/WeasyPrint) | Mature permissively licensed local artifact primitives. **ADOPT behind our artifact contract**; they do not supply templates, approvals, provenance, or content-policy enforcement. |

## Top Projects

| Project | GitHub | Category | License | Local Models | Agentic | Multimodal | RAG | Sandbox | Air-Gap Potential | Activity | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Open WebUI | [repo](https://github.com/open-webui/open-webui) | Workbench | Custom source-available | VERIFIED | VERIFIED | VERIFIED | VERIFIED | No | Medium–High after hardening | High; 151k★, v0.11.3 Aug 2026 | STUDY ONLY |
| AnythingLLM | [repo](https://github.com/Mintplex-Labs/anything-llm) | Workbench/router | MIT | VERIFIED | VERIFIED | Partial | VERIFIED | Partial FS containment | Medium | High; 65.8k★, v1.16.1 Aug 2026 | STUDY ONLY |
| LangGraph | [repo](https://github.com/langchain-ai/langgraph) | Agent runtime | MIT | Via adapters | VERIFIED | Via nodes | Via nodes | No | High for core | High; 41.3k★, releases Aug 2026 | WRAP |
| OpenHands ecosystem | [repo](https://github.com/OpenHands/OpenHands) | Coding agent/control plane | MIT | PARTIAL | VERIFIED | Partial | No | NOT SUFFICIENT | Low–Medium | High; 86.9k★, v1.16 Aug 2026 | STUDY ONLY |
| goose | [repo](https://github.com/aaif-goose/goose) | Coding/general agent | Apache-2.0 | VERIFIED | VERIFIED | Partial | No | External only | Medium–High if isolated | High; 54.0k★, v1.49 Sep 2026 | WRAP as agent pack |
| vLLM | [repo](https://github.com/vllm-project/vllm) | GPU model serving | Apache-2.0 | VERIFIED | Tool API | VERIFIED | No | No | High after hardening | High; 91.3k★, v0.28 Aug 2026 | WRAP |
| llama.cpp | [repo](https://github.com/ggml-org/llama.cpp) | CPU/edge model serving | MIT | VERIFIED | Tool API | VERIFIED | No | No | High | High; 127.5k★, rolling releases | WRAP |
| vLLM Semantic Router | [repo](https://github.com/vllm-project/semantic-router) | Model routing | Apache-2.0 | VERIFIED | N/A | Signals | N/A | No | Medium | High; 5.7k★, v0.3 Jun 2026 | STUDY ONLY |
| Haystack | [repo](https://github.com/deepset-ai/haystack) | RAG/agent pipelines | Apache-2.0 | VERIFIED | VERIFIED | Via components | VERIFIED | No | High after telemetry off | High; 26.4k★, v3.1.1 Sep 2026 | WRAP selected parts |
| Qdrant | [repo](https://github.com/qdrant/qdrant) | Vector/hybrid retrieval | Apache-2.0 | N/A | No | Multivectors | VERIFIED | No | High after hardening | High; 34.4k★, v1.19.1 Sep 2026 | ADOPT |
| Docling | [repo](https://github.com/docling-project/docling) | Document intelligence | MIT; model licenses vary | N/A | No | VERIFIED | Feeds RAG | Worker needed | High with pre-bundling | High; 66.2k★, v2.126 Sep 2026 | ADOPT isolated |
| PaddleOCR | [repo](https://github.com/PaddlePaddle/PaddleOCR) | OCR/layout/tables | Apache-2.0 | Local CV models | No | VERIFIED | Feeds RAG | Worker needed | High with pinned weights | Healthy; 89.1k★, v3.7 Jun 2026 | WRAP |
| gVisor | [repo](https://github.com/google/gvisor) | Sandbox runtime | Apache-2.0 | N/A | N/A | N/A | N/A | VERIFIED primitive | High with explicit net policy | Active; 19k★ | ADOPT |
| MCP | [repo](https://github.com/modelcontextprotocol/modelcontextprotocol) | Tool/plugin protocol | Mixed MIT/Apache-2.0 transition | N/A | VERIFIED | Tool-dependent | Tool-dependent | No | Medium with allowlist | High; active Sep 2026 | WRAP protocol only |
| OPA | [repo](https://github.com/open-policy-agent/opa) | Policy/governance | Apache-2.0 | N/A | N/A | N/A | N/A | No | High with local bundles | High; 12.2k★, v1.20.2 Sep 2026 | ADOPT as PDP |

Activity is based on observations on 2026-09-08 and will change. “Air-gap potential” means technically adaptable to disconnected operation; it does not mean the upstream defaults enforce zero egress.

### Maintenance, Issues, and Documentation Signal

GitHub's repository `open_issues_count` combines issues and pull requests, so queue sizes below are health signals rather than defect counts. Contributor totals were unavailable through the connector. Every shortlisted project had recent code, release, issue/PR, license, documentation, and configuration evidence checked where available.

| Project | Documentation/architecture quality | Activity and issue-health observation |
|---|---|---|
| Open WebUI | Good product/deployment docs; architecture is less cleanly separated than its feature docs | High activity; 283 open issues/PRs observed, recent release and same-week issue movement |
| AnythingLLM | Good technical overview and readable subsystem source | High activity; 312 open issues/PRs, same-day fixes and recent release |
| LangGraph | Strong API/concept docs and explicit checkpoint contracts | High activity; 764 open issues/PRs with continuing releases and fixes |
| OpenHands ecosystem | Good architecture/spec documents; chart candidly documents serious limitations | High activity; 686 open issues/PRs in Canvas, active SDK releases |
| goose | Good modular source/docs and explicit security guidance | High activity; 334 open issues/PRs, same-week feature work and release |
| vLLM | Extensive performance, serving, security and feature documentation | Very high activity; large 7,776 issue/PR queue reflects scale and rapid change, not automatic poor health |
| llama.cpp | Extensive practical server/model/backend docs, with some fast-moving behavior | Very high rolling activity; 2,446 open issues/PRs and continuous build releases |
| vLLM Semantic Router | Good example configuration and unusually useful router threat model; young APIs | Rapid activity; 531 open issues/PRs and a recent release, so integration churn risk is high |
| Haystack | Strong component/pipeline documentation and modular packages | High activity; same-day commits and a recent v3 release; active issue/PR handling observed |
| Qdrant | Strong operator/query docs and inspectable security configuration | High activity; same-day work and v1.19.1 release in the same week |
| Docling | Strong pipeline/model/offline usage docs; no root security policy found | High activity; 895 open issues/PRs, same-day commits and a recent release |
| PaddleOCR | Broad documentation and model catalog, though sprawling and version-sensitive | Healthy but slower than the fastest projects; v3.7 in June and later 2026 commits |
| gVisor | Excellent security, architecture, filesystem, network and operations docs | High activity; same-day commit and release within the prior week |
| MCP | Living specification with explicit transport/authorization security considerations | High activity and active issue discussion; fast specification evolution creates compatibility risk |
| OPA | Mature policy, security, deployment and decision-log documentation | High activity; CNCF-graduated project, same-day commit and current release |

## Detailed Analysis

The serving engines were assessed against the requested technical dimensions:

| Engine | OpenAI-compatible API | Tool calling | Multimodal | Structured output | Batching | Quantization | Multi-GPU | Offline finding |
|---|---|---|---|---|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | VERIFIED | VERIFIED; model/parser-specific | VERIFIED | VERIFIED | Continuous | VERIFIED | Tensor/pipeline and other modes | PARTIAL by default; local-cache support verified, usage telemetry must be disabled |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | VERIFIED | VERIFIED with compatible Jinja/chat template | VERIFIED | JSON-schema verified | Continuous server batching | Native GGUF quantization | VERIFIED across supported backends | VERIFIED with local artifacts and `--offline` |
| [SGLang](https://github.com/sgl-project/sglang) | VERIFIED for text/vision/embeddings | VERIFIED | VERIFIED | VERIFIED | Continuous | VERIFIED | Tensor/pipeline/expert/data parallelism | PARTIAL; local cache honored, complete zero-egress profile not found |

All three still require an authenticated internal gateway, pinned model/template artifacts, compatibility tests, resource admission, and deployment-enforced network policy. OpenAI compatibility is a wire-level convenience, not proof of identical tool, schema, tokenization, or multimodal semantics.

### Open WebUI

**Repository:** [open-webui/open-webui](https://github.com/open-webui/open-webui)  
**License:** [Open WebUI License](https://github.com/open-webui/open-webui/blob/main/LICENSE), a custom BSD-derived license with branding restrictions above a stated user threshold unless permission or an enterprise license is obtained. This is not equivalent to MIT/BSD/Apache and needs commercial review.  
**Primary purpose:** Full-featured self-hosted AI chat/workbench.  
**Last meaningful activity:** A substantive commit was observed on 2026-09-04 and v0.11.3 was released on 2026-08-31; issue activity remained current.  
**Architecture summary:** Svelte frontend plus FastAPI/SQLAlchemy backend, with Ollama/OpenAI-compatible endpoints, tools/functions/skills, RAG, authentication/access-control concepts, vector-store adapters, and Docker/Kubernetes deployment. Its [Python manifest](https://github.com/open-webui/open-webui/blob/main/pyproject.toml) shows a very broad hosted and local integration surface.

**Relevant components:** frontend/workbench, model adapters, tools, local RAG, multimodal UX, RBAC concepts, deployment.

**Strengths:** The strongest workbench UX reference in the pool; local Ollama composition is **VERIFIED** in its [Compose file](https://github.com/open-webui/open-webui/blob/main/docker-compose.yaml). Access grants, model visibility, RAG flows, and administration UX are worth studying.

**Weaknesses:** Monolithic dependency surface, provider-specific concerns mixed into the application, and licensing unsuitable for an unreviewed product foundation. Plugin requirements can invoke package installation when connected.

**Cloud dependencies:** Many are optional rather than mandatory, but hosted provider/storage/search/document SDKs are present. `OFFLINE_MODE` sets `HF_HUB_OFFLINE` and disables version checks in [environment code](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/env.py); this is **PARTIALLY VERIFIED offline behavior**, not a network boundary.

**Air-gap suitability:** Medium–High only after rebuilding with local adapters, pre-staged assets, offline flags, disabled plugin installation, and enforced outbound denial. Enforced zero egress is **NOT SUPPORTED** by the application alone.

**Security observations:** Telemetry opt-out defaults were present in inspected container/example settings. Tool creation is permission-checked, but uniform trusted-boundary enforcement across every plugin/tool path was not proven. Initial signup behavior and privileged administration require deployment review.

**What we should learn from it:** Workbench information architecture, model grants, knowledge UX, admin surfaces, and offline-state visibility.  
**What we should reuse:** No core code before license review; protocol-level interaction ideas only.  
**What we should NOT copy:** The monolith, hosted-provider dependency breadth, or an application flag presented as zero-egress enforcement.  
**Recommendation:** **STUDY ONLY**.

### AnythingLLM

**Repository:** [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm)  
**License:** [MIT](https://github.com/Mintplex-Labs/anything-llm/blob/master/LICENSE).  
**Primary purpose:** Self-hosted workbench combining chat, RAG, agents, local providers, flows, and routing.  
**Last meaningful activity:** Multiple substantive fixes were observed on 2026-09-08; v1.16.1 was released on 2026-08-27.  
**Architecture summary:** React/Vite frontend, Node/Express server, separate document collector, Prisma persistence, LanceDB default storage, multiple LLM/embedding/vector adapters, AIbitat agent runtime, flow builder, and a model-router subsystem. See its [technical overview](https://github.com/Mintplex-Labs/anything-llm/blob/master/README.md).

**Relevant components:** workbench, RAG, agent runtime, filesystem tools, model routing.

**Strengths:** Local Ollama, LocalAI, llama.cpp-compatible and native embedding options are **VERIFIED**. The [router service](https://github.com/Mintplex-Labs/anything-llm/blob/master/server/utils/router/index.js) implements ordered rules, calculated and LLM-classified conditions, sticky routing, fallback, cached classification, and cooldown—the closest application-level analogue to our router.

**Weaknesses:** Its router does not provide exact-environment allowlisting, declarative provider-neutral capabilities, hardware admission, or audit-grade decision traces. The whole application remains a tightly integrated product rather than modular infrastructure.

**Cloud dependencies:** Hosted-provider packages coexist with local providers. PostHog telemetry is enabled unless `DISABLE_TELEMETRY=true`, and documented flows download models/assets from CDNs or GitHub; see [telemetry code](https://github.com/Mintplex-Labs/anything-llm/blob/master/server/models/telemetry.js).

**Air-gap suitability:** Medium with telemetry disabled, assets pre-staged, local-only provider configuration, and an egress-denied deployment. Default zero egress is **NOT SUPPORTED**.

**Security observations:** Its [filesystem manager](https://github.com/Mintplex-Labs/anything-llm/blob/master/server/utils/agents/aibitat/plugins/filesystem/lib.js) checks real paths and symlinks under an allowed root—useful containment work. However, broad SQL/filesystem powers rely substantially on underlying credentials/host permissions, and no general approval gate was found.

**What we should learn from it:** Deterministic router rules, sticky session decisions, cooldown/fallback behavior, and filesystem canonicalization.  
**What we should reuse:** Concepts and tests, not the application or router as our source of truth.  
**What we should NOT copy:** Default telemetry/downloads or reliance on host scoping for tool authorization.  
**Recommendation:** **STUDY ONLY**.

### LangGraph

**Repository:** [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)  
**License:** [MIT](https://github.com/langchain-ai/langgraph/blob/main/LICENSE).  
**Primary purpose:** Stateful graph execution for agents and durable workflows.  
**Last meaningful activity:** Substantive fix observed 2026-09-03; SDK releases continued through 2026-08-27.  
**Architecture summary:** Pregel-inspired state graphs, checkpoint contracts, replay/resume, node-level interrupts, retries, and composable state. The [core package](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/pyproject.toml) depends on `langchain-core` but not directly on a hosted model SDK.

**Relevant components:** agent runtime, workflow graph, durable tasks, memory/checkpoints, human interruption.

**Strengths:** Durable execution is **VERIFIED** through the [checkpoint contract](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py); human-in-the-loop primitives are **VERIFIED** in its [interrupt types](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py). These are mature mechanics that should not be rebuilt casually.

**Weaknesses:** It does not supply model routing, permissions, sandboxing, audit, or sovereign storage. Examples and the production story create LangSmith/cloud gravity even though the open-source core is usable locally.

**Cloud dependencies:** None required by the core graph runtime; local model nodes are **PARTIALLY VERIFIED** because they are supplied through adapters rather than by LangGraph itself.

**Air-gap suitability:** High for the core when LangSmith and hosted integrations are excluded and checkpoint stores remain local.

**Security observations:** Its [serializer warning](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py) says untrusted Python objects must not be deserialized. Interrupts are control-flow features, not authorization gates; trusted enforcement must remain ours.

**What we should learn from it:** Explicit state machines, resumability, idempotent nodes, checkpoint versioning, and interrupts.  
**What we should reuse:** The core runtime behind an internal neutral interface, subject to a proof of concept.  
**What we should NOT copy:** Provider-specific examples, cloud observability assumptions, or unsafe serialization choices.  
**Recommendation:** **WRAP**.

### OpenHands Ecosystem

**Repository:** [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) and its execution-side [software-agent-sdk](https://github.com/OpenHands/software-agent-sdk).  
**License:** [MIT](https://github.com/OpenHands/OpenHands/blob/main/LICENSE).  
**Primary purpose:** Coding-agent control plane, persistent conversations/automations, and agent execution SDK.  
**Last meaningful activity:** Both repositories were active on 2026-09-08; Canvas v1.16.0 was released 2026-08-27 and SDK v1.45.0 on 2026-09-07.  
**Architecture summary:** The main repository is now primarily a React/TypeScript Agent Canvas. Execution is separated behind REST/ACP into agent servers and workspaces, supporting local, Docker, Kubernetes, VM, and remote backends. This control-plane/execution split is a useful architecture reference.

**Relevant components:** coding agents, persistent tasks, frontend/control plane, tools, execution backends.

**Strengths:** Clear conversational event model, swappable backends, persistent task UX, and explicit separation between Canvas and execution. The SDK supports tools, conversations, workspaces, and local/ephemeral Docker/Kubernetes workspaces.

**Weaknesses:** The open-source deployment does not supply enterprise isolation. The [Helm documentation](https://github.com/OpenHands/OpenHands/blob/main/helm/agent-canvas/README.md) explicitly describes an experimental, unauthenticated, single-tenant shared pod/PVC with no per-agent sandbox.

**Cloud dependencies:** Local use is possible, but ordinary startup/build flows use package registries, GHCR, downloadable server versions, and optional remote providers. Turnkey air gap is **NOT SUPPORTED**.

**Air-gap suitability:** Low–Medium after substantial rebuilding and pre-staging.

**Security observations:** Host mode grants full host filesystem/environment/network access. The SDK’s inspected Docker workspace command did not establish `cap-drop`, `no-new-privileges`, read-only root, PID/CPU/memory limits, seccomp, or `network=none`; the server also warns about unauthenticated network exposure. Optional Helm values can grant namespace-admin or cluster-admin authority—unacceptable for our agent workload.

**What we should learn from it:** Control-plane/backend separation, task UX, event streaming, and ACP-style backend substitution.  
**What we should reuse:** At most isolated protocol/SDK ideas after a security review.  
**What we should NOT copy:** Host execution, shared workspaces, unauthenticated services, privileged Kubernetes bindings, or default Docker workspace posture.  
**Recommendation:** **STUDY ONLY**.

### goose

**Repository:** [aaif-goose/goose](https://github.com/aaif-goose/goose)  
**License:** [Apache-2.0](https://github.com/aaif-goose/goose/blob/main/LICENSE).  
**Primary purpose:** Local-capable coding/general agent with providers, MCP extensions, recipes, sessions, and subagents.  
**Last meaningful activity:** Meaningful local/model changes were observed on 2026-09-05; v1.49.0 was released 2026-09-03.  
**Architecture summary:** Rust core with CLI, desktop, API/ACP server, provider trait/registry, MCP extensions, YAML recipes, sessions, and custom-distribution support. The [provider trait](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/providers/base.rs) is a useful modularity reference.

**Relevant components:** coding-agent pack, provider abstraction, tool/plugin system, permission UX, evaluation.

**Strengths:** Ollama/local inference is **VERIFIED**. Explicit `always_allow`, `ask_before`, and `never_allow` states are **VERIFIED** in the [permission manager](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/config/permission.rs); malformed permission YAML fails startup rather than granting access. Telemetry is opt-in in inspected [PostHog code](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/posthog.rs).

**Weaknesses:** The agent can execute on the host and supplies no hardened isolation boundary; the [security policy](https://github.com/aaif-goose/goose/blob/main/SECURITY.md) tells operators to use a VM/container. Enterprise RBAC, audit, tenant isolation, and trusted approval enforcement are absent.

**Cloud dependencies:** Optional cloud providers, updates, and runtime shim downloads exist. Local operation is real but a fully offline package requires disabling and pre-staging these paths.

**Air-gap suitability:** Medium–High only inside our external sandbox, with local providers and allowlisted, preinstalled extensions.

**Security observations:** Model-based “smart approval” must never authorize actions. Keyring fallback and extension/download behavior need a controlled secrets and artifact policy.

**What we should learn from it:** Provider traits, packaged recipes, MCP extension registration, and allow/ask/deny UX.  
**What we should reuse:** Potentially wrap it as an optional coding-agent pack within our broker and sandbox.  
**What we should NOT copy:** Host authority, model-mediated authorization, or runtime downloads.  
**Recommendation:** **WRAP** as an optional isolated agent pack.

### vLLM

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)  
**License:** [Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE).  
**Primary purpose:** High-throughput GPU inference serving.  
**Last meaningful activity:** Active on 2026-09-08; v0.28.0 released 2026-08-26.  
**Architecture summary:** Engine plus OpenAI-compatible server with continuous batching, prefix caching, quantization, structured output, tool parsers, multimodal models, heterogeneous accelerators, and distributed parallelism. Capabilities are documented in its [README](https://github.com/vllm-project/vllm/blob/main/README.md), [tool-calling guide](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md), and [parallelism guide](https://github.com/vllm-project/vllm/blob/main/docs/serving/parallelism_scaling.md).

**Relevant components:** inference, text/vision models, tool calling, structured generation, multi-GPU serving.

**Strengths:** Batching, quantization, tensor/pipeline parallelism, OpenAI compatibility, structured outputs, and multimodal support are **VERIFIED**. Offline resolution honors local files/`HF_HUB_OFFLINE`, including [offline tests](https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/llm/offline_mode/test_offline_mode.py). Its security process is comparatively mature.

**Weaknesses:** It is an inference engine, not a tenant-security or routing control plane. Model-specific tool parsers/templates still require compatibility validation.

**Cloud dependencies:** Model hub paths exist. More importantly, zero egress is **NOT SUPPORTED by default**: usage reporting targets `stats.vllm.ai` unless `VLLM_NO_USAGE_STATS=1` or `DO_NOT_TRACK=1` is set, per [usage documentation](https://github.com/vllm-project/vllm/blob/main/docs/usage/usage_stats.md).

**Air-gap suitability:** High after pre-staging models/tokenizers, forcing offline mode, disabling usage statistics, disabling remote-code trust, requiring service authentication, and network-isolating the server.

**Security observations:** Never expose the OpenAI server directly. The internal gateway should authenticate, authorize, constrain sampling/schema requests, and redact/audit metadata. Treat model weights and custom code as untrusted supply-chain inputs.

**What we should learn from it:** Scheduler/caching/parallelism capability descriptors and model-specific tool-parser registration.  
**What we should reuse:** The serving engine behind our adapter.  
**What we should NOT copy:** vLLM request/config types into business contracts or its telemetry defaults.  
**Recommendation:** **WRAP** as the primary GPU backend.

### llama.cpp

**Repository:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
**License:** [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).  
**Primary purpose:** Efficient GGUF inference across CPUs, GPUs, workstations, and edge hardware.  
**Last meaningful activity:** Active on 2026-09-08 with rolling build releases.  
**Architecture summary:** Portable C/C++ inference core plus `llama-server`, quantization tooling, CPU/GPU hybrid offload, multiple accelerator backends, and multi-GPU operation.

**Relevant components:** local inference, constrained hardware, quantized models, embeddings/reranking where supported, vision, structured output.

**Strengths:** Local GGUF execution, low-bit quantization, OpenAI-compatible APIs, broad hardware support, and multimodal operation are **VERIFIED** in the [README](https://github.com/ggml-org/llama.cpp/blob/master/README.md) and [multimodal guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md). Function calling is **VERIFIED but conditional** on compatible Jinja/chat templates per [function-calling documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md). JSON-schema generation is implemented in [server schema code](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-schema.cpp).

**Weaknesses:** Template/model compatibility is more operator-sensitive than a generic OpenAI surface suggests. It does not provide enterprise tenancy, admission control, or model lifecycle governance.

**Cloud dependencies:** Remote model URL/Hugging Face paths exist, but `--offline` forces cache/local operation in inspected [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

**Air-gap suitability:** **VERIFIED** when invoked with pinned local artifacts and `--offline` under network denial.

**Security observations:** Its [security guidance](https://github.com/ggml-org/llama.cpp/blob/master/SECURITY.md) recommends sandboxing untrusted models/inputs and warns against exposing RPC/server endpoints to untrusted networks. Multi-tenant isolation remains our responsibility.

**What we should learn from it:** Explicit local/offline switch, GGUF supply chain, hardware feature detection, and constrained-generation adapters.  
**What we should reuse:** Engine/server behind a least-privilege adapter.  
**What we should NOT copy:** Direct public model-server exposure or implicit trust in chat templates/model metadata.  
**Recommendation:** **WRAP** for CPU/edge/GGUF deployments.

### vLLM Semantic Router

**Repository:** [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)  
**License:** [Apache-2.0](https://github.com/vllm-project/semantic-router/blob/main/LICENSE).  
**Primary purpose:** Policy/signal-driven Mixture-of-Models routing through Envoy ExtProc.  
**Last meaningful activity:** Active on 2026-09-08; v0.3 released in June 2026.  
**Architecture summary:** Envoy data path plus router service, declarative model catalog/backends, signals/classifiers, decisions, projections, health/fallback, and several model-selection algorithms. Its [sample configuration](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) includes local backends, weighted pools, modality/context/capability metadata, privacy and safety signals, and optional remote providers.

**Relevant components:** semantic/capability routing, model catalog, safety/privacy signals, fallbacks, observability.

**Strengths:** Heterogeneous local pools, weighted backends, health/ejection, tool/modality/PII/jailbreak/complexity signals, and multiple selection algorithms are **VERIFIED**. Its [security document](https://github.com/vllm-project/semantic-router/blob/main/SECURITY.md) identifies adversarial classification, incorrect routing, external dependencies, and configuration bypass as threats.

**Weaknesses:** Young project, large issue/PR surface, and a critical inference-path proxy. Sample catalogs include remote provider paths and the normal install/playground experience is connected. Complete zero-egress artifact behavior was not verified.

**Cloud dependencies:** Optional OpenAI/Anthropic-compatible backends and downloadable models/catalog artifacts exist. Local backends are real, but air gap is configuration-dependent.

**Air-gap suitability:** Medium; **PARTIALLY VERIFIED** with local pools and pre-staged classifiers, but not as a sovereign trust boundary.

**Security observations:** An adversarial prompt can influence advisory classifiers. A semantic route must never override exact environment eligibility, permissions, or disabled-model state.

**What we should learn from it:** Signal taxonomy, route explanations, pool health, classifiers, workflows, and router-specific threat modeling.  
**What we should reuse:** Potentially an advisory scorer after the trusted eligibility filter, never the authoritative registry.  
**What we should NOT copy:** Remote-capable sample catalogs, installation downloads, or policy decisions delegated to probabilistic classifiers.  
**Recommendation:** **STUDY ONLY**, with a future isolated proof of concept.

### Haystack

**Repository:** [deepset-ai/haystack](https://github.com/deepset-ai/haystack)  
**License:** [Apache-2.0](https://github.com/deepset-ai/haystack/blob/main/LICENSE).  
**Primary purpose:** Component and pipeline framework for RAG, search, and agents.  
**Last meaningful activity:** Active on 2026-09-08; v3.1.1 released 2026-09-03.  
**Architecture summary:** Typed components joined into pipelines for conversion, preprocessing, embedding, sparse/dense retrieval, ranking, generation, routing, and agents. Integrations are modular rather than a single mandatory server.

**Relevant components:** RAG orchestration, BM25/vector retrieval, metadata filtering, reranking, agents, evaluation.

**Strengths:** Modular pipelines, local components, local models, in-memory BM25, ranking/filtering, and agent tools are **VERIFIED** in the [repository](https://github.com/deepset-ai/haystack). It is a better library boundary than adopting an all-in-one RAG workbench.

**Weaknesses:** The integration ecosystem contains numerous cloud providers; adopting the catalog wholesale would expand dependencies and configuration risk. It does not define our permission/provenance model.

**Cloud dependencies:** Optional. Anonymous telemetry is enabled unless `HAYSTACK_TELEMETRY_ENABLED=False`, verified in [telemetry source](https://github.com/deepset-ai/haystack/blob/main/haystack/telemetry/_telemetry.py).

**Air-gap suitability:** High for selected components after telemetry is compiled/configured off, integrations are allowlisted, and models are pre-staged.

**Security observations:** Pipeline composition must not bypass document ACL filters, model eligibility, or tool authorization. Deserializing arbitrary pipeline/component configuration should be administrator-only.

**What we should learn from it:** Small typed components, explicit pipeline wiring, document-store adapters, and reusable evaluation hooks.  
**What we should reuse:** Selected conversion/retrieval/ranking pipeline components behind our local-only contracts.  
**What we should NOT copy:** The full provider catalog, default telemetry, or authorization implicit in pipeline configuration.  
**Recommendation:** **WRAP selected components**.

### Qdrant

**Repository:** [qdrant/qdrant](https://github.com/qdrant/qdrant)  
**License:** [Apache-2.0](https://github.com/qdrant/qdrant/blob/master/LICENSE).  
**Primary purpose:** Vector database and retrieval engine.  
**Last meaningful activity:** Active on 2026-09-08; v1.19.1 released 2026-09-04.  
**Architecture summary:** Rust server supporting dense, sparse and multivector indexes, hybrid queries, filtering, payload indexes, quantization, replication, and local container deployment.

**Relevant components:** vector store, hybrid retrieval primitives, metadata filtering, multivectors, retrieval scaling.

**Strengths:** Dense/sparse/multivector search, hybrid query composition, filters, quantization, snapshots, API keys, read-only keys, and JWT-based fine-grained access are **VERIFIED**. It is a focused and mature storage component.

**Weaknesses:** It is not a knowledge permission, citation, embedding, or reranking layer. Cluster operation adds distributed-system cost that an initial single-node deployment may not need.

**Cloud dependencies:** No hosted database is mandatory. Built-in integrations should be disabled in a local-only profile.

**Air-gap suitability:** High with local images/packages and explicit hardening.

**Security observations:** Sovereign-safe defaults are **NOT SUPPORTED**: [configuration](https://github.com/qdrant/qdrant/blob/master/config/config.yaml) defaults telemetry to enabled and TLS to off; API keys/RBAC must be enabled. Bind it only to an internal network, disable telemetry, require TLS/auth where transport crosses trust boundaries, and enforce document ACLs before/within every query.

**What we should learn from it:** Hybrid query primitives, payload indexes, collection aliases/snapshots, and quantization.  
**What we should reuse:** Qdrant itself through a repository-neutral vector-store interface.  
**What we should NOT copy:** Default configuration or its internal types as business-facing knowledge contracts.  
**Recommendation:** **ADOPT with a hardened profile**.

### Docling

**Repository:** [docling-project/docling](https://github.com/docling-project/docling)  
**License:** [MIT](https://github.com/docling-project/docling/blob/main/LICENSE) for code; individual model licenses must be inventoried separately.  
**Primary purpose:** Local document conversion and structured document understanding.  
**Last meaningful activity:** Active on 2026-09-08; v2.126.0 released 2026-09-04.  
**Architecture summary:** Conversion pipelines for PDF and office/image inputs, with layout, reading order, tables, OCR, formulas, pictures/charts, and a structured document model exported to JSON/Markdown/HTML.

**Relevant components:** document intelligence, multimodal ingestion, tables, OCR coordination, canonical document representation, RAG preprocessing.

**Strengths:** Layout, reading order, table structure, OCR, formulas, images/charts, and loss-aware structured output are **VERIFIED** through [pipeline options](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py). Runtime air gap is unusually well supported: local `artifacts_path` is accepted and missing artifacts fail; remote services default disabled and raise `OperationNotAllowed` in [advanced options](https://github.com/docling-project/docling/blob/main/docs/usage/advanced_options.md). External plugins also default denied.

**Weaknesses:** Container builds normally download packages and model assets. Default page/file limits are effectively unbounded unless configured. No root security policy was found.

**Cloud dependencies:** None required at runtime after dependencies and model weights are embedded. Offline installation still needs our signed artifact pipeline.

**Air-gap suitability:** **VERIFIED at runtime** with pre-bundled artifacts; build-time air gap remains our responsibility.

**Security observations:** Parse hostile files only in a resource-limited worker. Enforce file/page/time/memory limits, MIME validation, decompression limits, no network, read-only inputs, and output quotas. Track each model’s license/hash.

**What we should learn from it:** Canonical document IR, pipeline specialization, explicit remote-service denial, and model artifact injection.  
**What we should reuse:** The converter behind an isolated document-processing contract.  
**What we should NOT copy:** Online Docker build/download steps or unbounded processing defaults.  
**Recommendation:** **ADOPT behind an isolated worker**.

### PaddleOCR

**Repository:** [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)  
**License:** [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).  
**Primary purpose:** Multilingual OCR and document-layout/table/formula/chart understanding.  
**Last meaningful activity:** v3.7.0 released 2026-06-11; repository activity observed through 2026-07-22.  
**Architecture summary:** Paddle-based detection, recognition, orientation, layout, table, formula, document and chart pipelines with Python/CLI/service interfaces.

**Relevant components:** scanned PDFs, OCR, handwriting/languages, layout, tables, formulas, chart-to-table, multimodal document extraction.

**Strengths:** OCR across 100+ languages, table cells/structure, layout, formulas, chart-to-table, JSON/Markdown output, local deployment, and explicit local model paths for isolated networks are **VERIFIED** in the [repository documentation](https://github.com/PaddlePaddle/PaddleOCR).

**Weaknesses:** Large framework/model dependency surface, hardware-specific packaging, and overlapping pipelines make version pinning and validation important. Engineering drawings/CAD semantics remain unproven.

**Cloud dependencies:** Local operation is supported, but ordinary utilities and examples automatically download model archives. Runtime zero egress is therefore **NOT SUPPORTED by default**.

**Air-gap suitability:** High only with pinned weights and dependencies mirrored/pre-bundled, all remote model-source paths disabled, and network denial.

**Security observations:** Treat images/PDFs and model packages as untrusted. Run as a non-root isolated worker with resource quotas; do not let a parser fetch weights on demand.

**What we should learn from it:** Specialized OCR/layout model routing and confidence/structure outputs.  
**What we should reuse:** A narrow OCR/layout adapter, called by the document pipeline when Docling’s default backend is insufficient.  
**What we should NOT copy:** Runtime download behavior or a second competing canonical document model.  
**Recommendation:** **WRAP** as a specialist backend.

### gVisor

**Repository:** [google/gvisor](https://github.com/google/gvisor)  
**License:** [Apache-2.0](https://github.com/google/gvisor/blob/master/LICENSE).  
**Primary purpose:** OCI-compatible sandbox runtime with a user-space application kernel.  
**Last meaningful activity:** Active in the 2026 research window with continuing releases and security maintenance.  
**Architecture summary:** `runsc` intercepts application syscalls and implements a large Linux surface in the Sentry, reducing direct access to the host kernel. It integrates with OCI/containerd/Kubernetes while retaining container-like startup and packaging; see the [architecture guide](https://github.com/google/gvisor/blob/master/website/docs/architecture_guide/README.md).

**Relevant components:** Python/shell execution, per-task workspaces, filesystem mediation, syscall isolation, container orchestration.

**Strengths:** A materially stronger boundary than ordinary namespaces/seccomp alone, mature OCI integration, active vulnerability handling, and usable container ergonomics. Isolation as a primitive is **VERIFIED** in its [security model](https://github.com/google/gvisor/blob/master/g3doc/architecture_guide/security.md); it is explicitly designed to reduce host-kernel attack surface.

**Weaknesses:** Incomplete syscall compatibility and workload-dependent I/O/network overhead. It is not a policy engine, approval service, secrets broker, or tenant scheduler.

**Cloud dependencies:** None inherent. Images and packages still need an offline registry/bundle.

**Air-gap suitability:** High when images are pre-staged. `--network=none` fully disables external networking while preserving loopback according to the [networking guide](https://github.com/google/gvisor/blob/master/g3doc/user_guide/networking.md), so the broker should make this mandatory unless a narrowly approved network profile applies.

**Security observations:** gVisor does not select `network=none` for us; outbound denial, UID mapping, capabilities, cgroups, quotas, and lifecycle teardown remain orchestration duties. Filesystem access is mediated through the Gofer and supports read-only/overlay modes in its [filesystem guide](https://github.com/google/gvisor/blob/master/g3doc/user_guide/filesystem.md); stricter profiles should disable performance-oriented direct filesystem paths where practical. The control plane must not expose the Docker socket to agents.

**What we should learn from it:** Explicit sandbox threat model, defense in depth, runtime-class separation, and compatibility testing.  
**What we should reuse:** `runsc` as the initial isolation backend behind our own sandbox broker.  
**What we should NOT copy:** Treating runtime selection alone as complete containment or exposing an engine socket/API to agent code.  
**Recommendation:** **ADOPT** as the first sandbox runtime; keep a pluggable boundary for future microVMs.

### Model Context Protocol

**Repository:** [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol); implementation evidence also checked in [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).  
**License:** [Mixed transition](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/LICENSE): newer specification/code is Apache-2.0, older unrelicensed contributions remain MIT, and non-specification documentation is CC-BY-4.0. Check each reused package.  
**Primary purpose:** Protocol and reference implementations for model-accessible tools/resources/prompts.  
**Last meaningful activity:** Specification commit observed 2026-09-08; release 2026-07-28. Reference servers also released in August 2026.  
**Architecture summary:** Provider-neutral JSON-RPC host/client/server contracts with capability negotiation and typed tools, resources, and prompts. Local stdio starts a server subprocess; streamable HTTP supports remote transport.

**Relevant components:** tool/plugin protocol, local connectors, capability discovery, filesystem tools.

**Strengths:** A clean process/protocol boundary, typed schemas, explicit server capabilities, and broad ecosystem value are **VERIFIED** in the [specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/draft/index.mdx). The filesystem reference server implements allowed roots and has [path-validation tests](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/__tests__/path-validation.test.ts), including traversal cases.

**Weaknesses:** Security depends on each server and host. Protocol metadata is not authorization, and reference implementations are examples rather than production assurance.

**Cloud dependencies:** Server-dependent; many reference/community integrations require external services or API keys. Local-only operation must use a strict allowlist. Streamable HTTP requires origin validation and recommends localhost/authentication; local stdio inherits credentials from its environment.

**Air-gap suitability:** Medium–High for audited local servers with pinned artifacts; not for arbitrary marketplace discovery or runtime installation.

**Security observations:** The specification explicitly leaves user consent/tool safety to the host; protocol negotiation is **NOT SUPPORTED as authorization**. Its [authorization security guidance](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/draft/basic/authorization/security-considerations.mdx) covers token theft, SSRF, confused-deputy and token-passthrough risks. Reference servers also include deliberately broad testing behavior; for example the “everything” server’s [environment tool](https://github.com/modelcontextprotocol/servers/blob/main/src/everything/tools/get-env.ts) can return process environment variables. Server identity, per-tool permissions, arguments, secrets, approvals, and audited results must be enforced by our host.

**What we should learn from it:** Process isolation, schema discovery, transports, capability negotiation, and ergonomic tool packaging.  
**What we should reuse:** Protocol/SDK pieces after license and dependency review.  
**What we should NOT copy:** Automatic server installation, broad ambient credentials, marketplace trust, or the assumption that a declared tool annotation grants authority.  
**Recommendation:** **WRAP protocol only** under our capability manifest and policy engine.

### OPA

**Repository:** [open-policy-agent/opa](https://github.com/open-policy-agent/opa)  
**License:** [Apache-2.0](https://github.com/open-policy-agent/opa/blob/main/LICENSE).  
**Primary purpose:** General-purpose policy decision point using Rego and structured input/data.  
**Last meaningful activity:** Active in 2026; v1.20.2 released 2026-09-03.  
**Architecture summary:** Applications submit structured authorization/policy input to an embedded or sidecar evaluator; policy/data can be loaded locally, tested, bundled, and decision-logged.

**Relevant components:** ABAC/policy, environment/tool/model eligibility, deployment admission, decision traces.

**Strengths:** Mature deny/allow policy evaluation, policy tests, local bundles, structured decisions, and wide deployment integration. It cleanly separates policy decision from application code. [Decision logs](https://github.com/open-policy-agent/opa/blob/main/docs/docs/management-decision-logs.md) include input, result, bundle/policy metadata and decision IDs, with masking/drop controls.

**Weaknesses:** Rego and policy operations add expertise and lifecycle burden. OPA does not enforce decisions: every trusted boundary still needs a project-owned policy-enforcement point.

**Cloud dependencies:** None required. Bundle discovery, status, and decision-log services can be remote in some configurations and must be local/disabled for air gap; see [OPA documentation source](https://github.com/open-policy-agent/opa/tree/main/docs/content).

**Air-gap suitability:** High with locally signed/versioned policies and local decision logs.

**Security observations:** OPA server authentication and authorization default to off in its [security guide](https://github.com/open-policy-agent/opa/blob/main/docs/docs/security.md), although the listener defaults to localhost. Fail closed on evaluator unavailability, unknown environment, invalid policy, undefined/indeterminate result, and restrict network-capable built-ins. Treat policy bundles as signed configuration; never allow prompts/model output to write policy or assert authorization facts.

**What we should learn from it:** Policy-as-code, decision input/output schemas, bundle versioning, tests, and separation of decision from enforcement.  
**What we should reuse:** OPA as an optional local PDP where rules justify it.  
**What we should NOT copy:** Remote control-plane assumptions or treating a PDP as enforcement/audit by itself.  
**Recommendation:** **ADOPT as PDP; BUILD our PEPs and domain policy model**.

## Architecture Patterns Worth Adopting

1. **Trusted eligibility before optimization.** Filter by active environment, enabled state, exact capability/modality, local artifact presence, permission, data classification, and hardware feasibility. Then rank by deterministic priority and an optional advisory quality/latency/load scorer. Record every exclusion and the final reason.
2. **Provider-neutral ports with narrow adapters.** Follow the useful separation seen in inference engines and goose, while keeping vLLM, llama.cpp, SGLang, Ollama, GGUF, and chat-template details inside provider adapters.
3. **Explicit state-machine agents.** Use typed graph state, idempotent nodes, checkpoint versioning, replay, bounded retries, cancellation, and interrupts. An interrupt requests approval; a trusted policy service grants it.
4. **Control plane separated from execution and model data planes.** The API, registry, policy, audit, and scheduler should never share a privilege boundary with agent code, document parsers, or inference servers.
5. **One canonical document IR.** Preserve page, bounding box, table cell, image, OCR confidence, source hash/version, and ACL provenance through chunking and citation. Docling is the best starting representation; specialist OCR should map into it.
6. **Hybrid retrieval as composition.** Keep BM25/sparse, dense, metadata/ACL filtering, reranking, and citation assembly as observable steps. Qdrant supplies retrieval primitives; it does not own authorization or provenance.
7. **Capability manifests for tools and agent packs.** Pin package hash/version/license, executable identity, allowed inputs, secrets, files, network destinations, resource budget, approval class, and output schema. MCP can be one transport beneath this manifest.
8. **External sandbox with ephemeral workspaces.** Use non-root gVisor-backed tasks, read-only base images, bounded writable mounts, no host sockets/devices, PID/CPU/memory/time/output quotas, default `network=none`, and verified teardown.
9. **Local policy decisions with enforcement at every boundary.** OPA can calculate decisions; the API, router, retriever, tool broker, artifact exporter, and sandbox launcher must enforce them independently and fail closed.
10. **Offline artifact promotion.** Build connected staging and disconnected runtime profiles: mirror packages/images/models, verify hashes/signatures/SBOMs/licenses, malware-scan, approve, then import through a controlled transfer process. Runtime downloads are forbidden.
11. **Append-only correlated events.** Give task, agent step, model route, retrieval result, tool call, approval, sandbox, and artifact a shared correlation lineage with actor, policy version, inputs/outputs hashes, and redaction rules.

## Architecture Patterns to Avoid

- Treating an `OFFLINE_MODE`, `HF_HUB_OFFLINE`, or telemetry environment variable as a network security boundary.
- Monolithic workbenches that mix UI, provider SDKs, RAG, arbitrary plugins, and execution authority in one process.
- Provider-specific request/configuration types in shared contracts or registries.
- A probabilistic classifier, LLM, prompt instruction, or tool annotation making authorization decisions.
- Directly exposing vLLM/llama.cpp/vector database/sandbox services to users or untrusted networks.
- Running agents on the host, mounting `/var/run/docker.sock`, privileged containers, shared tenant volumes, empty API tokens, or authentication disabled by default.
- Runtime installation of Python/npm packages, MCP servers, plugins, backends, models, tokenizers, or OCR weights.
- Deserializing untrusted Python objects in durable checkpoints or accepting unsigned workflow/policy definitions.
- Retrieval that applies ACL filters after vector search or loses source/page/chunk/version provenance before citation.
- Enabling telemetry and relying only on opt-out configuration; an outbound deny policy should make accidental calls impossible.
- Adopting all-in-one RAG/workbench stacks merely for feature breadth when focused components have smaller trusted and dependency surfaces.
- Maintaining vLLM, SGLang, and every alternate backend before measured workloads justify the operational cost.

## Build vs Reuse Matrix

| Sovereign Workbench Component | Best Existing Project(s) | Build/Reuse Decision | Reason |
|---|---|---|---|
| Frontend | Open WebUI, AnythingLLM, OpenHands | **BUILD OUR OWN; STUDY ONLY** | Required modularity, licensing, security boundaries, workspace/audit UX, and product identity are not met by an upstream base. |
| Inference | vLLM, llama.cpp; SGLang optional | **WRAP** | Mature serving, batching, quantization, multimodal and hardware support; adapters preserve replaceability. |
| Model provider abstraction | goose patterns; LiteLLM patterns | **BUILD OUR OWN** | Shared contracts must express our capabilities and sovereignty without cloud-provider types/dependencies. |
| Model registry | vLLM Semantic Router catalog; MLflow lifecycle registry | **BUILD OUR OWN** | Exact environment eligibility, disabled-state exclusion, deterministic priority, artifact trust, and provider-neutral capabilities are unique requirements. |
| Model router | vLLM Semantic Router, AnythingLLM, LiteLLM, RouteLLM | **BUILD OUR OWN; STUDY/optional advisory WRAP** | No candidate combines hard policy eligibility, hardware admission, deterministic fallback, explainability, and learned optimization safely. |
| Agent runtime | LangGraph | **WRAP behind our abstraction** | Mature graph/checkpoint/interrupt mechanics; our contracts retain provider, persistence, policy, and audit independence. |
| Coding agents | goose, aider, OpenHands SDK | **WRAP optional agent packs** | Reuse repository maps/edit loops/tool ecosystems only inside our sandbox and policy broker. |
| Tools/plugins | MCP SDK/reference patterns, goose; Extism for constrained plugins | **BUILD permission framework; WRAP protocol/runtime** | Transport/schema discovery and WASM confinement are reusable; capability grants, approvals, secrets, provenance, and enforcement are differentiating trust boundaries. |
| Sandbox | gVisor; Firecracker later | **ADOPT runtime; BUILD broker** | Isolation primitives are mature, but workspace/mount/network/resource/approval/audit orchestration is ours. |
| OCR | PaddleOCR, OCRmyPDF | **WRAP/ADOPT** | Mature multilingual OCR and searchable-PDF repair; isolate and pre-stage weights/binaries. |
| Document intelligence | Docling | **ADOPT behind worker** | Strong structured document IR and verified local artifacts/remote-service denial. |
| RAG | Haystack components; RAGFlow patterns | **BUILD thin sovereign composition; WRAP components** | ACLs, local-only model routing, lineage and deterministic citations must remain trusted and domain-owned. |
| Vector database | Qdrant | **ADOPT** | Mature hybrid primitives and filters; harden telemetry, auth, TLS, binding and backups. |
| Reranker | Local cross-encoder through vLLM/llama.cpp or dedicated worker | **WRAP; BUILD governance contract** | Serving can be reused; model eligibility, licenses, calibration, fallbacks and audit need our registry. |
| Artifacts | python-docx, PptxGenJS/python-pptx, XlsxWriter, WeasyPrint, LibreOffice conversion | **ADOPT libraries; BUILD artifact engine** | File-format primitives are mature; templates, deterministic validation, provenance, approvals and policy are product logic. |
| Authentication | Keycloak | **ADOPT** | Mature local OIDC/federation/session/MFA; do not rebuild identity. |
| Permissions | OPA; OpenFGA later | **BUILD domain model/PEPs; ADOPT OPA PDP** | Enforcement must live at trusted boundaries; OPA provides mature evaluation, while object relations can be deferred. |
| Secrets | OpenBao | **ADOPT as external service** | Reuse mature local secret/PKI lifecycle and audit devices; issue scoped, short-lived credentials through our broker. |
| Audit | OPA decision records plus local append-only storage patterns | **BUILD OUR OWN** | Cross-component lineage, redaction, integrity, retention and air-gapped export are core governance requirements. |
| Observability | OpenTelemetry; Langfuse only after hardening | **WRAP OTel; DEFER UI** | Use vendor-neutral local traces/metrics/logs; avoid default telemetry and hosted coupling. |
| Evaluation | Promptfoo, MLflow, router-specific evals | **WRAP in isolated lab; BUILD sovereign suites** | Reuse harness mechanics; maintain local golden sets, policy/security negatives, routing calibration and citation evaluation. |
| Deployment | OCI/containerd/Kubernetes, gVisor, Zarf, local registries | **BUILD supported offline profiles; ADOPT packaging primitives** | Upstreams assume connected builds. Zarf can capture disconnected payloads, but we must require signatures and prohibit unsandboxed package actions/runtime downloads. |

## Proposed Revised Architecture

No architecture files are changed by this report. The following recommendations refine the current plan.

### KEEP

- The web/API/orchestration/tool/provider layering and the rule that business logic never calls a model runtime directly.
- Provider-neutral model capabilities and exact active-environment eligibility.
- Lower numeric priority winning, lexicographic model-ID tie-breaking, and absolute exclusion of disabled models.
- Local-only inference/data, telemetry off, least privilege, no Docker socket, isolated execution, and zero-egress/air-gap as first-class requirements.
- Modular agents, tools, providers, knowledge connectors, document processors, and artifact generators.

### CHANGE

- Split the model router into a **trusted eligibility/admission stage** and an **advisory selection stage**. The first is deterministic and fail-closed; the second may use semantic, quality, latency, load, cache, or hardware scores only over the eligible set.
- Split orchestration into a trusted control plane and untrusted/less-trusted workers: inference, document processing, retrieval jobs, code execution, and artifact rendering.
- Represent human approval as a signed, scoped, expiring authorization decision bound to actor, task, tool, arguments/resource scope, policy version, and sandbox—not merely a paused graph node.
- Make RAG a provenance-preserving pipeline rather than a generic vector search call: authorization filter → sparse/dense retrieval → fusion → local rerank → citation assembly → answer verification.
- Treat artifact generation as a governed pipeline with schema/template versions, deterministic post-generation validation, malware/content checks, provenance, and approval before release.

### ADD

- A model-artifact catalog beside the runtime registry: content hash, source, license, format, tokenizer/template, supported runtime versions, quantization, remote-code requirement, approval state, and environment promotion status.
- A route-decision record containing excluded candidates and reasons, eligible set, selected model, policy/config versions, resource snapshot, fallback chain, and timing.
- A local capability broker for tools/MCP with manifests, per-principal grants, secrets injection, argument/resource constraints, approval classes, and audit hooks.
- A sandbox broker with pluggable gVisor/microVM backends, ephemeral volumes, explicit mount graph, default network denial, quotas, and destruction verification.
- A canonical document IR and provenance schema covering source hash/version, ACL labels, pages, bounding boxes, OCR confidence, tables/cells, images/diagrams, chunks, embeddings, retrieval scores, rerank scores, and citations.
- A persistent task/workspace state store distinct from agent-runtime checkpoints, with idempotency, leases, cancellation, replay policy, retention, and integrity controls.
- An append-only audit pipeline and local OpenTelemetry collector, both with field-level redaction and no external exporters in production profiles.
- A connected build/promotion zone and a separate disconnected runtime bill of materials, including mirrored packages/images/models, signatures/hashes, SBOMs, licenses, vulnerability status, and controlled import/export.

### REMOVE

- From future production profiles, any automatic runtime downloader, marketplace installer, cloud-provider plugin, external telemetry/exporter, update checker, or permissive fallback when environment/policy configuration is missing.
- From the trusted design, direct user access to inference/vector/sandbox services and any ability for agents to reach the host container runtime socket.
- From the router, any assumption that an OpenAI-compatible endpoint is semantically uniform across tools, multimodal inputs, structured output, tokenization, or safety behavior.

### DEFER

- Learned semantic/cost routing until deterministic routing, route logging, local evaluation data, and rollback are mature.
- SGLang as a third supported inference backend until repeatable local workloads show a material advantage over vLLM.
- Firecracker/microVM execution until the threat model or tenant isolation requirement justifies its infrastructure cost; preserve the sandbox interface now.
- OpenFGA until object-sharing graphs exceed clear application RBAC/ABAC rules.
- A large distributed vector cluster, distributed multi-agent scheduler, or full observability UI until single-node/on-prem operational evidence requires them.

## Top 5 Projects We Should Study Closely

1. **[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)** — not as our policy authority, but as the richest current vocabulary for signals, model pools, health, route explanation, multimodal/tool routing, and router threat modeling.
2. **[LangGraph](https://github.com/langchain-ai/langgraph)** — the best candidate to avoid rebuilding durable graph execution, checkpoint, replay, retry, and interrupt mechanics while retaining a sovereign wrapper.
3. **[Docling](https://github.com/docling-project/docling)** — the strongest fit for a local canonical document pipeline, with implementation evidence for local artifacts, denied remote services, and disabled external plugins.
4. **[vLLM](https://github.com/vllm-project/vllm)** — the likely primary GPU serving engine and the reference point for batching, structured output, tool parsing, multimodal, quantization, and distributed serving.
5. **[gVisor](https://github.com/google/gvisor)** — the most practical initial isolation primitive for Python/shell/tool workloads, provided we build the network, mount, quota, secrets, approval, and lifecycle broker around it.

Close runners-up are AnythingLLM for deterministic/sticky/fallback router behavior, goose for provider/tool/permission UX, llama.cpp for offline constrained hardware, Qdrant for hybrid storage, and OPA for policy evaluation.

## Components We Should Definitely Build Ourselves

- **Runtime model registry and artifact trust catalog**, because exact-environment eligibility, disabled models, deterministic ordering, provider-neutral capabilities, local availability, license/approval state, and fail-closed behavior are core invariants.
- **Two-stage model router**, including trusted admission, advisory scoring, health/fallback, deterministic tie-breaking, reason codes, and audit-grade route traces.
- **Provider-neutral model and agent contracts**, while wrapping engines/frameworks below them.
- **Tool permission/capability framework and approval service**, with enforcement outside prompts and models.
- **Sandbox broker**, though not the isolation kernel/runtime itself.
- **Audit and provenance model**, spanning user action, agent steps, routes, retrieval, tools, approvals, sandboxes, and artifacts.
- **Sovereign RAG composition and citation lineage**, while reusing stores, parsers, embedders, and rerankers.
- **Artifact engine governance**, templates, validation, provenance, approvals, and policy; not DOCX/PPTX/XLSX serialization.
- **Offline artifact promotion and deployment profiles**, including zero-egress verification and fail-closed configuration checks.
- **Workbench frontend/control experience**, informed by upstream UX but aligned with our modular architecture, license position, persistent workspaces, and security model.

## Components We Should Definitely Reuse

- **Inference:** vLLM and llama.cpp; benchmark SGLang later.
- **Agent mechanics:** LangGraph core if the proof of concept confirms acceptable dependency and serialization boundaries.
- **Vector retrieval:** Qdrant with hardened configuration.
- **Document/OCR:** Docling, PaddleOCR, and OCRmyPDF in isolated workers.
- **Sandbox isolation:** gVisor initially; Firecracker remains a pluggable future option.
- **Policy evaluation and identity:** OPA and Keycloak, while retaining project-owned enforcement and application authorization semantics.
- **Protocol/telemetry standards:** selected MCP SDK pieces and OpenTelemetry, both constrained to local-only configurations.
- **Artifact formats:** python-docx, PptxGenJS or python-pptx, XlsxWriter, WeasyPrint, and controlled LibreOffice conversion where needed.
- **Evaluation mechanics:** selected Promptfoo/MLflow/router-evaluation techniques inside egress-denied development infrastructure.

## Research Gaps

- No inspected router satisfied capability, exact-environment, hardware/load, deterministic priority, security policy, fallback, and audit requirements together.
- No general agent framework supplied enterprise-grade tool authorization, approval, sandbox containment, and append-only audit as one trustworthy local boundary.
- Engineering drawings, P&IDs, schematics, CAD semantics, and revision/change understanding were not satisfactorily solved by the general document projects.
- A governed, provider-neutral local reranker registry with calibrated fallbacks and clear offline packaging remains missing.
- Citation correctness evaluation—especially page/region/table-cell provenance and permission-safe citations—remains immature.
- Offline model/package/container supply-chain promotion, signatures, license manifests, and reproducible disconnected updates are mostly operator exercises rather than complete upstream solutions.
- High-assurance sandbox networking, data exfiltration controls, artifact scanning, and multi-tenant secrets injection still require substantial platform integration even with gVisor or Firecracker.
- Local multimodal safety evaluation for industrial images/documents and tool-call reliability across model templates needs an internal benchmark suite.
- Office artifact semantic quality and deterministic validation are not solved by file-format libraries; templates and domain constraints remain product work.

## Research Accounting and Limitations

- **Repositories discovered:** 30 in the formal landscape funnel, plus narrow dependency spot-checks for authentication, observability, evaluation, and artifact formats.
- **Repositories deeply inspected:** 20. Deep inspection means multiple repository-owned artifacts beyond the README; the remaining candidates received metadata/license/README/activity screening.
- **Repositories shortlisted:** 15.
- **Evidence limitations:** Contributor totals were unavailable through the connected GitHub interface. GitHub’s repository `open_issues_count` combines issues and pull requests, so it was used only qualitatively. Some large repositories are fast-moving; star, release, issue, and commit observations are a 2026-09-08 snapshot. Absence of a finding is not proof of absence, which is why uncertain claims are labeled PARTIALLY VERIFIED or CLAIMED ONLY.
- **License caution:** Apache-2.0, MIT, and BSD-family licenses are permissive subject to their notices/terms. Custom, mixed, MPL, AGPL, GPL, and commercial/source-available terms are flagged for legal review; this report does not make legal conclusions.
