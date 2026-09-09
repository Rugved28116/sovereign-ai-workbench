# Project Mission

Sovereign AI Workbench is a self-hosted, model-agnostic, agentic AI platform for confidential enterprise and industrial work. The deployed product must support local open-weight models, air-gapped deployment, multimodal processing, agentic workflows, local tools, local knowledge retrieval, artifact generation, auditability, and configurable security policies.

These instructions apply to the entire repository unless a more specific `AGENTS.md` adds stricter rules.

# Core Architectural Rules

1. Business logic must never directly depend on a particular model, model family, provider, or inference runtime.
2. All model interaction must pass through internal model-provider abstractions.
3. Models must be replaceable without redesigning business logic.
4. The architecture must support multiple simultaneous models for different capabilities, including reasoning, coding, vision, embeddings, and reranking.
5. Agents, tools, model providers, knowledge connectors, document processors, and artifact generators must remain modular and replaceable.
6. Provider-specific configuration must stay inside provider adapters or provider-specific configuration layers.
7. Shared contracts and business-facing types must remain provider-neutral.
8. New components must use existing abstractions where they fit before introducing new abstractions.

# Sovereignty and Privacy Rules

- No cloud AI API may be introduced as a product runtime dependency.
- No OpenAI, Anthropic, Gemini, Cohere, or similar hosted-model SDK may be required by the product runtime.
- Development tools such as Codex may be used to build the project, but they are not part of the deployed product.
- Production functionality must not require internet connectivity.
- External telemetry must be disabled by default.
- User documents, prompts, embeddings, model outputs, tool results, and logs must remain local to operator-controlled infrastructure.
- Fully air-gapped operation is a first-class deployment requirement, not a fallback mode.

If an implementation would require external network access, identify the conflict explicitly. Do not silently add the dependency or network path.

# Deployment Environments

- `development`: may use mock providers and development tooling. Development conveniences must not become production requirements.
- `on-prem`: all AI inference and organizational data remain inside infrastructure controlled by the organization.
- `air-gapped`: the deployed system operates without any external network dependency.

Model and tool eligibility must respect the active environment. Missing, unknown, or unauthorized environment configuration must fail closed.

# Security Rules

- Use default-deny behavior for sensitive actions and access.
- Require explicit permissions for tools and enforce least-privilege access.
- Never place secret values, credentials, private keys, or production tokens in source control.
- Do not grant agents arbitrary host filesystem access.
- Do not give agents direct access to the Docker socket.
- Do not provide unrestricted shell execution.
- Code execution must occur in isolated sandboxes when that capability is implemented.
- Outbound network access from execution sandboxes must default to disabled.
- Destructive operations require explicit authorization.
- Future high-risk operations should support human approval gates.
- Enforce authorization at trusted boundaries; prompts and model output are not authorization.

Never silently weaken a security restriction to make a test, demonstration, or development workflow easier.

# Model Registry Rules

- Lower numeric priority wins; priority `10` is preferred over priority `20`.
- Equal priorities are resolved by ascending lexicographic model ID.
- A model must be explicitly eligible for the exact active environment.
- Disabled models are never routable.
- Provider-specific settings must not leak into the shared model registry.
- Model capabilities must remain declarative and provider-neutral.
- Mock models must remain development-only and must never be routed in `on-prem` or `air-gapped` environments.

# Coding Agent Workflow

Before changing code, coding agents must:

1. Inspect relevant files and repository instructions.
2. Understand the existing architecture and trust boundaries.
3. Identify the smallest change that satisfies the requirement.
4. Preserve and reuse existing abstractions.
5. Avoid unrelated edits.

After changing code, coding agents must:

1. Run the smallest relevant validation or tests, followed by broader relevant checks when justified.
2. Inspect the diff for correctness and scope.
3. Check for accidental secrets or confidential data.
4. Check for unintended external or cloud dependencies and network behavior.
5. Report every file changed.
6. Report tests and validation performed, including anything that could not run.
7. Report assumptions, unresolved risks, and security or compatibility implications.

# Scope Discipline

Coding agents must not:

- Rewrite unrelated components.
- Perform broad refactors without a requirement and clear justification.
- Introduce infrastructure merely because it may be useful later.
- Create premature microservices.
- Add caching, queues, databases, orchestration platforms, or distributed systems unless the current requirement needs them.

Prefer the simplest architecture that satisfies the current requirement while preserving future extensibility.

# Dependency Policy

Every new dependency must have a clear, documented reason. Prefer the standard library where practical, mature open-source local-first dependencies, and dependencies that can be installed and verified in offline environments. Avoid unnecessary SDK wrappers.

Do not introduce mandatory dependencies on AWS, Azure, GCP, Firebase, Supabase, hosted vector databases, hosted authentication, or hosted observability systems.

# Testing Rules

Require proportionate tests for implemented behavior. Prioritize coverage of:

- Model routing and environment restrictions.
- Permission enforcement and security policies.
- Sandbox boundaries and network isolation.
- Tool execution.
- Knowledge retrieval.
- Artifact generation.

Security tests must include negative cases proving that prohibited and unauthorized actions are blocked.

# Documentation Rules

Architecture changes must update the relevant documentation. Important design decisions must explain what changed, why it changed, security implications, air-gap implications, and compatibility implications. Do not allow implementation and architecture documentation to drift apart.

# Git and Change Hygiene

- Keep changes focused and diffs meaningful.
- Do not commit generated junk, secrets, or unrelated formatting changes.
- Do not commit large model binaries.
- Do not commit datasets unless they are deliberately approved synthetic or public fixtures.
- Preserve user changes and avoid destructive Git operations unless explicitly authorized.

# Hard Prohibitions

Coding agents must not introduce any of the following into the deployed product without a deliberate, documented architecture decision explicitly authorized by the user:

- Cloud AI APIs.
- External telemetry.
- Mandatory internet access.
- Direct model calls from business logic.
- Unrestricted host shell access.
- Agent access to `/var/run/docker.sock`.
- Production routing to mock models.
- Plaintext secrets.
- Security controls that fail open.
- Hidden outbound network calls.

# Conflict Handling

If a user request conflicts with sovereignty, security, air-gap support, or architectural boundaries:

1. Do not silently violate these rules.
2. Explain the specific conflict.
3. Propose the smallest compatible alternative.
4. Change an architectural rule only when the user explicitly decides to redesign that part of the system, and document the decision.

# Definition of Done

A task is complete only when:

- The requested behavior is implemented.
- Relevant tests or validation pass.
- Architecture rules remain satisfied.
- No accidental network or cloud dependency was added.
- No secrets were introduced.
- Documentation is updated when necessary.
- Changed files, validation, assumptions, and unresolved risks are reported.
