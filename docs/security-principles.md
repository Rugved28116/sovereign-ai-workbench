# Security Principles

Security is a system property and a prerequisite for useful sovereign AI. These principles guide later design and implementation.

## Data sovereignty

- Keep prompts, documents, model inputs and outputs, embeddings, logs, and artifacts inside the operator-controlled environment.
- Do not assume or silently initiate external network access.
- Support fully offline installation, operation, backup, and recovery.
- Collect the minimum data needed and define retention explicitly.

## Deny by default

- Deny model, tool, connector, filesystem, process, and network access unless explicitly granted.
- Apply the deterministic sovereign eligibility filter before any routing optimization. A model or tool is eligible only when its enabled status, exact active environment, required capabilities, permissions, approved local availability, and applicable security restrictions are established.
- Fail closed when identity, policy, configuration, environment, model eligibility, or tool eligibility is missing, unknown, unauthorized, or indeterminate.
- Use only the canonical deployment environments: `development`, `on-prem`, and `air-gapped`. Eligibility requires an exact environment match, and mock models are eligible only in `development`.
- Allow advisory optimization to rank only eligible candidates. It must never restore an excluded candidate, grant authority, or override an environment, permission, or security decision.
- Enforce authorization at trusted service boundaries, never only in clients or prompts.

## Explicit tool permissions

Each future tool must declare its operations, required permissions, accessible resources, network policy, input and output schemas, time and resource limits, and audit behavior. A model's request to use a tool is untrusted input, not authorization.

## Isolated execution

Model-produced or user-supplied code must eventually run in a disposable, constrained sandbox. The design must restrict filesystem mounts, processes, CPU, memory, execution time, system calls, and network access. The host environment and control plane must not be directly exposed.

Agents must never receive direct or indirect access to `/var/run/docker.sock`, other container runtime sockets, arbitrary host paths, or unrestricted host shell execution. Sandbox network egress defaults to disabled; `air-gapped` operation prohibits external access.

## Secrets and configuration

- Never store secrets in source control, examples, images, fixtures, logs, or generated artifacts.
- Load secrets through an operator-controlled mechanism appropriate to the deployment environment.
- Keep `.env.example` limited to documented names and safe placeholders.
- Rotate credentials and minimize their scope and lifetime.

## Supply chain

- Keep dependencies minimal, reviewed, pinned, and reproducible.
- Produce an offline bill of materials and verify mirrored artifacts before air-gapped installation.
- Avoid install-time downloads, remote assets, telemetry, and auto-update behavior.
- Track licenses and vulnerabilities without requiring production systems to reach the internet.

## Network isolation

- Deployed functionality must not require internet connectivity, and external telemetry is disabled by default.
- Execution sandboxes default to zero egress. Any future network-enabled profile requires an explicit destination allowlist, authorization, and auditability.
- `air-gapped` deployments must have no external network dependency or path. Application-level offline settings supplement, but do not replace, deployment-enforced network isolation.

## Input, output, and logging

- Validate and bound untrusted inputs at every trust boundary.
- Treat retrieved content and model output as potentially hostile, including prompt injection and malicious code.
- Encode outputs for their destination and prevent path traversal and unsafe deserialization.
- Record security-relevant metadata for audit while excluding confidential payloads and secrets by default.

## Verification

Security controls require automated tests for allowed and denied behavior. Threat models should be created before sensitive capabilities are enabled and revisited when boundaries, tools, connectors, models, or deployment assumptions change.
