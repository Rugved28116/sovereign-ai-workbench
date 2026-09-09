# Architecture

## Status

This document describes both the intended architecture and the currently implemented backend boundaries. The repository contains a FastAPI service, deterministic routing, development-only mock inference, and basic text-generation protocol adapters for local OpenAI-compatible endpoints and vLLM. It contains no installed real inference runtime and no downloaded or served real model.

## Current provider implementation

- `MockProvider` implements deterministic, network-free development inference.
- `LocalOpenAICompatibleProvider` implements basic text generation against an explicitly configured local OpenAI-compatible endpoint.
- `VLLMProvider` implements the same basic protocol integration while preserving vLLM-specific provider identity and configuration.
- Both local HTTP adapters reuse a shared internal OpenAI-compatible transport for request/response handling and local-endpoint safeguards.
- The adapter milestone does not install a vLLM server, download a model, or serve a real model. Those runtime and model deployment steps remain future work.
- The implemented adapters do not provide tool calling, vision, embeddings, streaming to clients, structured output, or model management.

## Architectural drivers

The workbench is designed for confidential enterprise and industrial workloads in three deployment environments:

1. `development` on a trusted workstation.
2. `on-prem` within an organization's controlled infrastructure.
3. `air-gapped` with no external network dependency.

All three environments should use the same logical contracts. Deployment configuration and scale may differ, but business behavior must not depend on internet availability.

## Planned boundaries

```text
User
  |
Web application
  |
Application API / agent orchestration
  |--------------------|----------------------|
Permissioned tools     Knowledge connectors   Artifact generators
  |
Model router
  |-- Stage 1: sovereign eligibility
  |-- Stage 2: advisory optimization
  |
Internal provider contract
  |
Local provider adapters -> locally operated model runtimes
```

- **Web application:** presents workflows and results. It must not be the sole enforcement point for authorization.
- **Application API:** owns application use cases, policy enforcement, orchestration, and audit event production.
- **Model router:** first determines sovereign eligibility, then optimizes only within the eligible candidate set. It returns provider-neutral responses.
- **Provider adapters:** translate internal contracts to locally operated inference runtimes. Provider-specific details stop at this boundary.
- **Agents:** coordinate bounded workflows through declared contracts rather than direct infrastructure access.
- **Tools:** expose narrow operations with explicit permissions, validated inputs, timeouts, and auditable outcomes.
- **Knowledge connectors:** retrieve only from configured, authorized internal sources and retain source provenance.
- **Artifact generators:** produce files or structured outputs behind replaceable contracts.
- **Execution sandbox:** eventually runs untrusted or model-produced code with resource, filesystem, process, and network restrictions.

## Dependency direction

Business and orchestration logic may depend on shared contracts, but never on a model family or inference engine. Provider adapters may depend on internal provider contracts and, when later approved, local runtime clients. Infrastructure may assemble components but must not leak deployment-specific concerns into domain logic.

```text
apps/services -> packages/contracts <- provider adapters
       |                                  |
       +-------- domain behavior ---------+

No domain import may point from business logic to a concrete model provider.
```

Model providers and inference runtimes are replaceable infrastructure. vLLM, llama.cpp, and any later runtime must remain behind project-owned, provider-neutral contracts and dedicated provider adapters. Runtime-specific protocols and configuration must not become business-facing contracts.

## Accepted component direction

[ADR 0001](adr/0001-foundational-architecture.md) is the authoritative architectural decision and contains the detailed rationale, boundaries, implications, and deferred questions. At a high level:

- **BUILD:** the sovereign control plane, including the frontend, application API, provider contracts, model registry and router, governed retrieval, tool permissions, sandbox broker, artifact governance, audit/provenance, and offline provisioning.
- **WRAP:** vLLM as the primary GPU runtime, llama.cpp for lightweight or constrained inference, LangGraph initially for orchestration mechanics subject to evaluation, Docling where suitable for document conversion, and other selected processing or reranking components.
- **REUSE:** Qdrant as the initial locally operated vector store and mature local artifact serialization libraries, always behind project-owned interfaces where they meet application boundaries.
- **EVALUATE LATER:** additional inference runtimes, tool-protocol compatibility, gVisor, OPA, and Keycloak. These are not committed dependencies.

The implemented provider adapters establish the project-owned WRAP boundary for local OpenAI-compatible protocols, including a dedicated vLLM adapter. This code-level integration does not adopt or install the vLLM runtime itself, does not download or serve a model, and does not change ADR 0001's component decisions. The other runtime and component choices above remain future work unless separately documented as implemented.

## Model registry

`models/registry.yaml` is a deployment-neutral inventory. Entries declare identity, provider key, enablement, allowed environments, capabilities, context length, routing priority, and provider-independent metadata. The initial entries are mocks; they do not perform inference and are allowed only in the `development` environment.

Routing is capability based and has two ordered stages:

1. **Stage 1 — sovereign eligibility:** deterministically and fail-closed excludes any model that is disabled; lacks the exact active environment, required capabilities, permissions, approved local artifacts, or hardware compatibility; or violates an applicable security restriction. Missing or indeterminate facts cause exclusion.
2. **Stage 2 — advisory optimization:** ranks only the candidates admitted by Stage 1 using permitted quality, latency, resource, context, specialization, availability, or load signals. Stage 2 can never restore an excluded model, change environment eligibility, grant permission, or bypass a security control.

The canonical environments are `development`, `on-prem`, and `air-gapped`. A model is environment-eligible only when its `environments` list explicitly contains the exact active environment. Missing, unknown, or unauthorized environment configuration must fail closed.

The initial Stage 2 behavior orders candidates by ascending numeric `priority`: lower values win, so priority `10` is preferred over priority `20`. Equal priorities are resolved by ascending lexicographic model ID. This `(priority, id)` ordering is also the deterministic fallback if future advisory signals are absent, invalid, disabled, or tied.

Provider-specific configuration does not belong in the shared registry's `metadata` or other provider-independent fields. Runtime-specific settings, including mock response or fixture selection, must remain inside the corresponding provider adapter or its private configuration layer.

## Trust boundaries

Treat user input, retrieved documents, model output, uploaded files, tool results, and provider responses as untrusted. Validate data at process and privilege boundaries. Authentication, authorization, audit, secret loading, network egress, and sandboxing require explicit designs before implementation.

No component should silently contact an external endpoint. Network destinations, if ever supported, must be explicitly configured, allowlisted, permissioned, observable, and disabled in the `air-gapped` environment.

## Architecture change policy

Any change to component responsibilities, trust boundaries, dependency direction, data flow, deployment assumptions, or security controls must update this document. Material decisions with meaningful alternatives should receive a dated architecture decision record under `docs/adr/`.
