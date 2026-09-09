# Development Roadmap

This roadmap is directional. Each phase should remain small, tested, documented, and usable without external network access at runtime.

[ADR 0001](adr/0001-foundational-architecture.md) is accepted and authoritative. Its current initial technology direction is:

- **vLLM — reuse through WRAP:** primary GPU inference runtime behind a dedicated provider adapter.
- **llama.cpp — WRAP:** lightweight and constrained inference behind a separate provider adapter.
- **LangGraph — WRAP initially, subject to evaluation:** orchestration mechanics remain behind a project-owned abstraction.
- **Docling — reuse where suitable through WRAP:** document conversion runs in a constrained worker behind the project-owned document-processing boundary.
- **Qdrant — REUSE:** initial locally operated vector store behind a vector-store interface.

These technologies require later dependency, offline, security, and compatibility review before adoption. None is a dependency or implementation target of the first implementation slice. gVisor, OPA, and Keycloak remain **EVALUATE LATER**, not committed dependencies.

## Phase 0: Foundation — complete

- Establish repository boundaries and contribution rules.
- Record architecture and security principles.
- Define a mock model registry without inference.
- Keep all application and infrastructure directories intentionally empty.

Exit criterion: the foundation is documented, syntactically valid, and contains no application dependencies or secrets.

## Phase 1: First implementation slice — current

- Implement only this path:

  ```text
  Browser -> FastAPI -> Model Router -> Mock Provider -> Mock Models
  ```

- Define the minimum provider-neutral contracts needed by the mock path without deciding the exact future provider contract.
- Load and validate the model registry; enforce exact environment eligibility and reject absent, unknown, or unauthorized environments.
- Implement Stage 1 sovereign eligibility before Stage 2 deterministic ordering by ascending numeric priority and lexicographic model ID.
- Add the health endpoint, deterministic mock inference, and negative tests for disabled, incapable, non-development, missing-environment, and unknown-environment models.

Exit criterion: the exact Browser → FastAPI → Model Router → Mock Provider → Mock Models slice works entirely offline, provider-specific concepts do not enter business contracts, and mock models remain `development`-only.

## Phase 2: Contracts and threat model

- Refine provider-neutral contracts for real-runtime requests, streaming responses, capabilities, errors, and usage metadata through a separate decision.
- Threat-model identity, model access, tools, data handling, audit, and the `development`, `on-prem`, and `air-gapped` environments.
- Define acceptance criteria for later component evaluations without adopting deferred dependencies.
- Extend contract and configuration tests.

Exit criterion: future adapter and security boundaries have explicit contracts and threat-model requirements without silently resolving ADR 0001's deferred questions.

## Phase 3: Local inference and routing

- Add approved adapters for locally operated inference runtimes.
- Preserve Stage 1 sovereign eligibility as authoritative, then perform Stage 2 advisory optimization only within its eligible candidate set.
- Evaluate and, if approved, add vLLM as the primary GPU runtime and llama.cpp for lightweight or constrained inference, each behind its own provider adapter.
- Support distinct reasoning, coding, vision, embedding, and reranking models.
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
