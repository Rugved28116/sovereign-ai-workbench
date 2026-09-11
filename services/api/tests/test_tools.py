from dataclasses import FrozenInstanceError, replace
import socket

import pytest

from sovereign_api.config import DeploymentEnvironment as Env
from sovereign_api.tool_contracts import (
    ToolDescriptor, ToolPermission, ToolRegistry, ToolRequest, ToolResult,
    ToolResultStatus, ToolRiskLevel as Risk, ToolSideEffectLevel as Effect,
    ToolValidationError, UnknownToolError,
)
from sovereign_api.tool_policy import (
    DeterministicToolPolicyEvaluator, ToolPermissionDecision as Decision,
)


READ = ToolPermission("filesystem.read")
WRITE = ToolPermission("filesystem.write")
NETWORK = ToolPermission("network.access")


def descriptor(**changes):
    return replace(ToolDescriptor(
        tool_id="local-file-reader", display_name="Local reader",
        description="Test descriptor only", capabilities=("read",),
        risk_level=Risk.LOW, side_effect_level=Effect.READ,
        requires_network=False, supports_read=True, supports_write=False,
        required_permissions=(READ,),
    ), **changes)


def evaluate(tool=None, grants=frozenset(), environment=Env.DEVELOPMENT):
    return DeterministicToolPolicyEvaluator().evaluate(
        descriptor() if tool is None else tool,
        granted_permissions=grants, environment=environment,
    )


def request(arguments):
    return ToolRequest("request-1", "local-file-reader", "read", arguments,
                       "task-1", "stage-1")


def test_registry_and_descriptor_copy_and_freeze_collections():
    permissions = [READ]
    capabilities = ["read"]
    tool = descriptor(required_permissions=permissions, capabilities=capabilities)
    entries = [tool]
    registry = ToolRegistry(entries)
    permissions.clear()
    capabilities.clear()
    entries.clear()
    assert registry.get(tool.tool_id) == tool
    assert tool.required_permissions == (READ,)
    assert tool.capabilities == ("read",)
    with pytest.raises(FrozenInstanceError):
        tool.requires_network = True
    with pytest.raises(FrozenInstanceError):
        registry.descriptors = ()


def test_duplicate_registry_ids_rejected():
    with pytest.raises(ToolValidationError):
        ToolRegistry((descriptor(), descriptor()))


def test_unknown_tool_rejected():
    with pytest.raises(UnknownToolError):
        ToolRegistry((descriptor(),)).get("unknown")


@pytest.mark.parametrize("environment", list(Env))
def test_read_requires_explicit_permissions_in_every_environment(environment):
    assert evaluate(environment=environment) is Decision.DENY
    assert evaluate(grants=frozenset({READ}), environment=environment) is Decision.ALLOW


def test_missing_one_permission_denied():
    tool = descriptor(required_permissions=(READ, WRITE))
    assert evaluate(tool, frozenset({READ})) is Decision.DENY


@pytest.mark.parametrize("risk,effect", [
    (Risk.LOW, Effect.DESTRUCTIVE), (Risk.HIGH, Effect.WRITE),
    (Risk.CRITICAL, Effect.WRITE),
])
def test_permission_precedes_approval(risk, effect):
    tool = descriptor(tool_id="dangerous-delete", risk_level=risk,
                      side_effect_level=effect, supports_write=True,
                      required_permissions=(WRITE,))
    assert evaluate(tool) is Decision.DENY
    assert evaluate(tool, frozenset({WRITE})) is Decision.REQUIRE_APPROVAL


@pytest.mark.parametrize("environment", list(Env))
def test_network_requires_grant_and_cannot_override_air_gap(environment):
    # Network grant is enforced even if the descriptor omits it from declarations.
    tool = descriptor(tool_id="network-fetch", requires_network=True)
    assert evaluate(tool, frozenset({READ}), environment) is Decision.DENY
    expected = Decision.DENY if environment is Env.AIR_GAPPED else Decision.ALLOW
    assert evaluate(tool, frozenset({READ, NETWORK}), environment) is expected


def test_air_gap_deny_precedes_destructive_approval():
    tool = descriptor(requires_network=True, supports_write=True,
                      side_effect_level=Effect.DESTRUCTIVE)
    assert evaluate(tool, frozenset({READ, NETWORK}), Env.AIR_GAPPED) is Decision.DENY


@pytest.mark.parametrize("environment", [None, "development", "unknown"])
def test_unknown_or_untyped_environment_denied(environment):
    assert evaluate(grants=frozenset({READ}), environment=environment) is Decision.DENY


def test_empty_permission_declaration_is_not_public_access():
    assert evaluate(descriptor(required_permissions=())) is Decision.DENY


@pytest.mark.parametrize("changes", [
    {"risk_level": "unknown"}, {"side_effect_level": "unknown"},
    {"requires_network": 0}, {"supports_write": True},
    {"required_permissions": ("filesystem.read",)},
    {"capabilities": (object(),)},
])
def test_malformed_descriptors_fail_closed(changes):
    with pytest.raises(ToolValidationError):
        descriptor(**changes)


def test_evaluator_revalidates_descriptor_and_grants():
    tool = descriptor()
    object.__setattr__(tool, "risk_level", "unknown")
    assert evaluate(tool, frozenset({READ})) is Decision.DENY
    assert evaluate(object(), frozenset({READ})) is Decision.DENY
    assert evaluate(grants=frozenset({"filesystem.read"})) is Decision.DENY


def test_request_accepts_json_and_copies_every_nested_collection():
    original = {"values": ["text", 1, 1.5, True, None, {"nested": [2]}]}
    result = request(original)
    original["values"][5]["nested"].append(3)
    original["values"].clear()
    assert result.arguments["values"] == ("text", 1, 1.5, True, None,
                                           {"nested": (2,)})
    with pytest.raises(TypeError):
        result.arguments["extra"] = 1
    with pytest.raises(TypeError):
        result.arguments["values"][5]["nested"] = ()
    with pytest.raises(FrozenInstanceError):
        result.operation = "write"


@pytest.mark.parametrize("invalid", [
    object(), lambda: None, object, RuntimeError("private"), b"binary",
    {1, 2}, float("nan"), float("inf"), {1: "value"},
])
def test_nested_non_json_values_rejected(invalid):
    with pytest.raises(ToolValidationError):
        request({"outer": [{"inner": invalid}]})


def test_socket_object_rejected_without_opening_socket():
    # Allocate an uninitialized socket object: no OS socket or network access.
    value = socket.socket.__new__(socket.socket)
    with pytest.raises(ToolValidationError):
        request({"socket": value})


def test_cycles_and_excessive_depth_rejected():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ToolValidationError):
        request({"cycle": cyclic})
    nested = {"invalid": object()}
    for _ in range(20):
        nested = {"next": nested}
    with pytest.raises(ToolValidationError):
        request(nested)


def test_argument_node_limit():
    with pytest.raises(ToolValidationError):
        request({"items": [None] * 10_001})


@pytest.mark.parametrize("identifier", ["*", "filesystem.*", "", "network", 1])
def test_permission_identifiers_are_exact(identifier):
    with pytest.raises(ToolValidationError):
        ToolPermission(identifier)


def test_results_are_immutable_references_and_safe_fields_only():
    result = ToolResult("request-1", "tool-1", ToolResultStatus.SUCCEEDED,
                        output_reference="output-1")
    with pytest.raises(FrozenInstanceError):
        result.output_reference = "other"
    with pytest.raises(ToolValidationError):
        replace(result, safe_message=RuntimeError("private"))
    with pytest.raises(ToolValidationError):
        replace(result, status=ToolResultStatus.FAILED)
    failure = ToolResult("request-1", "tool-1", ToolResultStatus.FAILED,
                         safe_message="Tool failed", error_code="tool_failed")
    assert failure.output_reference is None


def test_identical_inputs_produce_identical_decisions():
    assert [evaluate(grants=frozenset({READ})) for _ in range(5)] == [Decision.ALLOW] * 5
