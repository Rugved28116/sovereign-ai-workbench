# Sovereign AI Workbench

Sovereign AI Workbench is an early-stage project for confidential industrial and enterprise AI work. It aims to provide a self-hosted, model-agnostic environment in which people can use agents, tools, organizational knowledge, and generated artifacts without sending sensitive data to cloud AI services.

## The problem

Many organizations cannot place engineering data, operational records, source code, intellectual property, or regulated information in externally hosted AI systems. They still need a practical way to run capable models, connect approved internal data, automate bounded workflows, and retain control over deployment and security.

This project is intended to make that possible in the `development`, `on-prem`, and `air-gapped` deployment environments.

## Goals

- Run without cloud AI APIs or SDKs at runtime.
- Support local models for reasoning, coding, vision, embeddings, and reranking.
- Keep business logic independent of model families and inference engines.
- Make agents, tools, model providers, knowledge connectors, and artifact generators modular and replaceable.
- Apply deny-by-default controls, explicit tool permissions, and isolated code execution.
- Keep the architecture and dependency footprint small, typed where practical, and testable.
- Treat offline operation as a first-class constraint rather than a fallback.

## Planned architecture

The workbench will be split into a small number of boundaries:

- `apps/web`: the future browser-based user experience.
- `services/api`: the initial application API, model router, and provider boundary.
- `services/model-router`: internal model selection and provider abstraction.
- `packages/contracts`: shared, strongly typed interfaces and schemas.
- `models`: declarative model inventory and deployment-neutral metadata.
- `infrastructure/docker`: future `on-prem` container assets.
- `tests`: cross-component and acceptance tests.
- `synthetic-company`: non-sensitive sample data and evaluation scenarios.

Business code calls capability-oriented internal contracts. Provider adapters are the only layer aware of a particular model server or model family. No real inference runtime is included at this stage.

See [Architecture](docs/architecture.md) for the planned boundaries.

## Privacy philosophy

Data sovereignty is the default. The system must not assume internet access, transmit information externally, or silently broaden a user's permissions. Secrets belong outside version control. Security-sensitive operations should fail closed, tool access should be explicit and auditable, and generated code should eventually execute only inside constrained sandboxes.

## Project status

This repository now contains the first backend slice: a FastAPI health and generation API, registry validation, sovereign eligibility filtering, deterministic routing, and development-only mock inference. It does **not** contain a frontend, real model inference, containers, databases, or production deployment assets.

## High-level roadmap

1. Define typed contracts and threat models.
2. Build a minimal API, web shell, and mock provider path.
3. Add local provider adapters and capability-based routing.
4. Introduce permissioned tools, isolated execution, and approved knowledge connectors.
5. Harden, test, package, and document `on-prem` and `air-gapped` operations.

The detailed sequence and exit criteria are in the [development roadmap](docs/development-roadmap.md).

## Repository policy

Architecture changes must be documented. Every implemented component should have proportionate automated tests. New dependencies require a clear need and must remain compatible with offline deployment. See [AGENTS.md](AGENTS.md) for contribution rules used by coding agents.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
