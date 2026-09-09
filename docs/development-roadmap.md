# Development Roadmap

This roadmap is directional. Each phase should remain small, tested, documented, and usable without external network access at runtime.

[ADR 0001](adr/0001-foundational-architecture.md) is accepted and authoritative. Its current initial technology direction is:

- **vLLM — reuse through WRAP:** primary GPU inference runtime behind a dedicated provider adapter.
- **llama.cpp — WRAP:** lightweight and constrained inference behind a separate provider adapter.
- **LangGraph — WRAP initially, subject to evaluation:** orchestration mechanics remain behind a project-owned abstraction.
- **Docling — reuse where suitable through WRAP:** document conversion runs in a constrained worker behind the project-owned document-processing boundary.
- **Qdrant — REUSE:** initial locally operated vector store behind a vector-store interface.

These technologies require dependency, offline, security, and compatibility review before runtime adoption. The basic vLLM provider adapter and its OpenAI-compatible protocol integration are implemented without adding vLLM as a dependency. Installing the vLLM runtime and downloading or serving a real model remain future deployment work. The other listed technologies are not dependencies of the current implementation. gVisor, OPA, and Keycloak remain **EVALUATE LATER**, not committed dependencies.

## Phase 0: Foundation — complete

- Establish repository boundaries and contribution rules.
- Record architecture and security principles.
- Define a mock model registry without inference.
- Keep all application and infrastructure directories intentionally empty.

Exit criterion: the foundation is documented, syntactically valid, and contains no application dependencies or secrets.

## Phase 1: Backend foundation and provider adapters — current

- The initial mock path is complete:

  ```text
  Browser -> FastAPI -> Model Router -> Mock Provider -> Mock Models
  ```

- The provider adapter/protocol milestone is complete: `MockProvider`, `LocalOpenAICompatibleProvider`, and `VLLMProvider` implement the provider-neutral contract, and the two local HTTP adapters reuse a shared OpenAI-compatible transport.
- The local HTTP adapters currently implement basic text generation only. They do not provide tool calling, vision, embeddings, client streaming, structured output, or model management.
- Model registry loading and validation, exact environment eligibility, fail-closed environment handling, Stage 1 sovereign eligibility, and Stage 2 deterministic ordering are implemented.
- The health endpoint, deterministic mock inference, and negative tests for disabled, incapable, non-development, missing-environment, and unknown-environment models are implemented.
- No real inference runtime or model is included: vLLM is not installed, and no real model has been downloaded or served.

Current result: the backend supports the offline mock path plus basic protocol adapters for local OpenAI-compatible and vLLM endpoints. Provider-specific concepts do not enter business contracts, mock models remain `development`-only, and no runtime or model deployment is claimed.

## Phase 2: Contracts and threat model

- Refine provider-neutral contracts for real-runtime requests, streaming responses, capabilities, errors, and usage metadata through a separate decision.
- Threat-model identity, model access, tools, data handling, audit, and the `development`, `on-prem`, and `air-gapped` environments.
- Define acceptance criteria for later component evaluations without adopting deferred dependencies.
- Extend contract and configuration tests.

Exit criterion: future adapter and security boundaries have explicit contracts and threat-model requirements without silently resolving ADR 0001's deferred questions.

## Phase 3: Local inference and routing

- Deploy approved locally operated inference runtimes behind project-owned provider adapters; add further adapters only where needed.
- Preserve Stage 1 sovereign eligibility as authoritative, then perform Stage 2 advisory optimization only within its eligible candidate set.
- Evaluate and, if approved, install vLLM as the primary GPU runtime and download and serve an approved local model behind the already implemented `VLLMProvider`; separately evaluate llama.cpp for lightweight or constrained inference behind its own provider adapter.
- Evaluate and implement any later capability-specific support, including reasoning, coding, vision, embedding, and reranking models, through explicit provider-neutral contracts. None of those capability-specific integrations is part of the completed adapter milestone.
- Test provider failures, fallback policy, cancellation, and resource limits.

Exit criterion: business logic remains unchanged when local providers or model assignments change.

## Phase 4: Permissioned capabilities

- Define the agent and tool plugin contracts.
- Evaluate LangGraph against explicit acceptance criteria before using its mechanics behind the project-owned orchestrator abstraction.
- Implement explicit permission evaluation and auditable tool invocation.
- Add isolated code execution with deny-by-default filesystem and network policies.
- Add approved knowledge connector and artifact generator interfaces; evaluate Docling where suitable and Qdrant as the initial vector store behind project-owned interfaces.
- Test prompt injection boundaries, denied operations, and sandbox escape defenses.

Exit criterion: every privileged action is authorized outside the model, constrained, and auditable.

## Phase 5: Deployment and hardening

- Package reproducible `development`, `on-prem`, and `air-gapped` deployments.
- Document artifact mirroring, integrity verification, backup, recovery, upgrades, and key rotation.
- Add observability that protects confidential content.
- Perform load, resilience, security, and disaster-recovery testing.
- Produce dependency inventories and operational runbooks.

Exit criterion: deployment evidence and operating procedures support a production-readiness review.

## Deferred decisions

ADR 0001 intentionally leaves the following unresolved: the exact provider contract, LangGraph acceptance criteria, canonical document representation, Qdrant deployment topology, hardware admission model, OPA integration, final sandbox technology, audit event schema, artifact verification mechanism, and offline supply-chain promotion process. Resolve each through focused follow-up work without treating the initial technology direction as an implicit answer.

gVisor, OPA, and Keycloak remain **EVALUATE LATER**. They are not committed dependencies and are not part of the first implementation slice.
