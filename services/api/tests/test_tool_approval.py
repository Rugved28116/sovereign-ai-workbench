"""Request-bound approval suspends and resumes one explicit tool stage."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from test_explicit_tool_stages import (
    Provider, READ, WRITE, model_stage, setup, tool_stage,
)
from sovereign_api.agent_stage_execution import InvalidStageCoordinationError
from sovereign_api.agent_task_state import StageStatus, TaskStatus
from sovereign_api.errors import InvalidExecutionStateError
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.task_plan_runner import TaskPlanRunner
from sovereign_api.tool_approval import (
    ApprovalAuthorization, ApprovalChoice, ApprovalDecision,
    InvalidToolApprovalError, ValidatedApprovalAuthorization,
    _issue_validated_authorization, request_fingerprint,
)
from sovereign_api.tool_contracts import ToolPermission, ToolRequest
from sovereign_api.tool_contracts import ToolRegistry, ToolResult, ToolResultStatus
from sovereign_api.tool_execution import (
    ExecutableToolRegistry, PolicyEnforcedToolExecutor,
    ToolApprovalRequiredError, ToolPermissionDeniedError,
)
from sovereign_api.agent_stage_execution import ToolBackedStageExecutor
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator, ToolPermissionDecision
from sovereign_api.workspace_read_file import (
    WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_READ_FILE_TOOL_ID,
)
from sovereign_api.workspace_write_artifact import WORKSPACE_WRITE_ARTIFACT_TOOL_ID


class ApprovalPolicy:
    def __init__(self):
        self.calls = []

    def evaluate(self, descriptor, *, granted_permissions, environment):
        self.calls.append((descriptor.tool_id, granted_permissions, environment))
        base = DeterministicToolPolicyEvaluator().evaluate(
            descriptor, granted_permissions=granted_permissions,
            environment=environment,
        )
        if base is ToolPermissionDecision.DENY or environment is DeploymentEnvironment.AIR_GAPPED:
            return ToolPermissionDecision.DENY
        return ToolPermissionDecision.REQUIRE_APPROVAL


def run(coro):
    return asyncio.run(coro)


def decision(approval, choice=ApprovalChoice.APPROVE):
    return ApprovalDecision(approval.approval_id, choice,
                            approval.created_at + timedelta(seconds=1))


def suspended(tmp_path, *, tool_id=WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
              path="report.txt", grants=frozenset({WRITE}), stages=None):
    if stages is None:
        stages = (tool_stage(1, tool_id, {"path": path, "content": "report text"}),)
    policy = ApprovalPolicy()
    coordinator, runner, task, provider, store, workspace, artifact = setup(
        tmp_path, stages, grants=grants, policy=policy,
    )
    result = run(coordinator.execute_one_with_report(task, stages[0]))
    return coordinator, runner, task, result, provider, store, workspace, artifact, policy


def test_approval_suspends_without_executing_and_is_immutable(tmp_path):
    coordinator, _, task, report, provider, _, _, artifact, policy = suspended(tmp_path)
    state = report.state
    approval = report.approval_request
    assert task.task_status is TaskStatus.PENDING
    assert state.task_status is TaskStatus.AWAITING_APPROVAL
    assert state.current_stage_id == "stage-1"
    assert state.stage_states[0].status is StageStatus.AWAITING_APPROVAL
    assert state.stage_states[0].output_reference is None
    assert approval == state.approval_request
    assert approval.task_id == "task-1" and approval.tool_id == WORKSPACE_WRITE_ARTIFACT_TOOL_ID
    assert approval.requested_permissions == (WRITE,)
    assert len(approval.request_fingerprint) == 64
    assert "report text" not in repr(approval)
    assert not (artifact / "report.txt").exists()
    assert len(provider.requests) == 0
    assert len(policy.calls) == 1
    with pytest.raises(FrozenInstanceError):
        approval.tool_id = "other"
    with pytest.raises(FrozenInstanceError):
        decision(approval).decision = ApprovalChoice.REJECT
    caller_permissions = list(approval.requested_permissions)
    copied = replace(approval, requested_permissions=caller_permissions)
    caller_permissions.clear()
    assert copied.requested_permissions == (WRITE,)
    with pytest.raises(InvalidExecutionStateError):
        replace(state, approval_request=None)
    with pytest.raises(InvalidExecutionStateError):
        replace(state, current_stage_id=None)
    with pytest.raises(InvalidExecutionStateError):
        replace(state, stage_states=(state.stage_states[0].resume_approval(),))


@pytest.mark.parametrize("change", [
    {"approval_id": uuid4().hex},
    {"request_id": uuid4().hex},
    {"task_id": "other-task"},
    {"stage_id": "other-stage"},
    {"tool_id": "other.tool"},
    {"operation": "other_operation"},
    {"requested_permissions": (READ,)},
])
def test_mismatched_approval_never_executes(tmp_path, change):
    coordinator, _, _, report, _, _, _, artifact, _ = suspended(tmp_path)
    forged = replace(report.approval_request, **change)
    with pytest.raises(InvalidStageCoordinationError):
        run(coordinator.resume_approved_stage(
            report.state, forged, decision(forged),
            granted_permissions=frozenset({WRITE}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert not (artifact / "report.txt").exists()


def test_wrong_decision_id_and_malformed_decision_are_rejected(tmp_path):
    coordinator, _, _, report, _, _, _, artifact, _ = suspended(tmp_path)
    wrong = replace(decision(report.approval_request), approval_id=uuid4().hex)
    with pytest.raises(InvalidStageCoordinationError):
        run(coordinator.resume_approved_stage(
            report.state, report.approval_request, wrong,
            granted_permissions=frozenset({WRITE}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    with pytest.raises(InvalidToolApprovalError):
        ApprovalDecision(report.approval_request.approval_id, "approve",
                         datetime(2026, 9, 14, tzinfo=UTC))
    assert not (artifact / "report.txt").exists()


def test_changed_arguments_and_request_fingerprint_are_rejected(tmp_path):
    coordinator, _, _, report, _, _, _, artifact, _ = suspended(tmp_path)
    stage = report.state.plan.stages[0]
    changed = replace(stage, tool_arguments={"path": "other.txt", "content": "report text"})
    changed_plan = replace(report.state.plan, stages=(changed,))
    changed_state = replace(report.state, plan=changed_plan)
    with pytest.raises(InvalidStageCoordinationError):
        run(coordinator.resume_approved_stage(
            changed_state, report.approval_request, decision(report.approval_request),
            granted_permissions=frozenset({WRITE}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert not (artifact / "report.txt").exists()
    assert not (artifact / "other.txt").exists()
    request = ToolRequest(
        report.approval_request.request_id, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
        "write_artifact", changed.tool_arguments, "task-1", "stage-1",
    )
    assert request_fingerprint(request, (WRITE,)) != report.approval_request.request_fingerprint


def test_original_caller_argument_mutation_cannot_change_approval_binding(tmp_path):
    arguments = {"path": "report.txt", "content": "report text"}
    stage = tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID, arguments)
    policy = ApprovalPolicy()
    coordinator, _, task, _, _, _, artifact = setup(
        tmp_path, (stage,), grants=frozenset({WRITE}), policy=policy,
    )
    report = run(coordinator.execute_one_with_report(task, stage))
    fingerprint = report.approval_request.request_fingerprint
    arguments["path"] = "other.txt"
    assert stage.tool_arguments["path"] == "report.txt"
    assert report.approval_request.request_fingerprint == fingerprint
    approved = run(coordinator.resume_approved_stage(
        report.state, report.approval_request, decision(report.approval_request),
        granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert approved.state.task_status is TaskStatus.COMPLETED
    assert (artifact / "report.txt").exists()
    assert not (artifact / "other.txt").exists()


def test_request_fingerprint_is_canonical_and_permission_bound():
    first = ToolRequest("request-1", WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                        "write_artifact", {"path": "report.txt", "content": "text"},
                        "task-1", "stage-1")
    second = ToolRequest("request-1", WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         "write_artifact", {"content": "text", "path": "report.txt"},
                         "task-1", "stage-1")
    assert request_fingerprint(first, (WRITE, READ)) == request_fingerprint(second, (READ, WRITE))
    assert request_fingerprint(first, (WRITE,)) != request_fingerprint(first, (READ,))


def test_approve_rechecks_policy_and_executes_exactly_once(tmp_path):
    coordinator, _, _, report, _, _, _, artifact, policy = suspended(tmp_path)
    approved = run(coordinator.resume_approved_stage(
        report.state, report.approval_request, decision(report.approval_request),
        granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert approved.state.task_status is TaskStatus.COMPLETED
    assert approved.state.stage_states[0].status is StageStatus.COMPLETED
    assert approved.state.stage_states[0].selected_tool_id == WORKSPACE_WRITE_ARTIFACT_TOOL_ID
    assert approved.state.stage_states[0].output_reference == "report.txt"
    assert approved.state.approval_request is None
    assert (artifact / "report.txt").read_text() == "report text"
    assert len(policy.calls) == 2
    with pytest.raises(InvalidStageCoordinationError):
        run(coordinator.resume_approved_stage(
            approved.state, report.approval_request, decision(report.approval_request),
            granted_permissions=frozenset({WRITE}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert len(policy.calls) == 2


def test_registered_tool_callable_runs_once_only_after_approval(tmp_path):
    class CountingReader:
        tool_id = WORKSPACE_READ_FILE_TOOL_ID
        descriptor = WORKSPACE_READ_FILE_DESCRIPTOR

        def __init__(self):
            self.calls = 0

        async def execute(self, request):
            self.calls += 1
            return ToolResult(
                request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
                text_content="bounded result",
            )

    stage = tool_stage(1, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"})
    policy = ApprovalPolicy()
    coordinator, _, task, _, _, _, _ = setup(
        tmp_path, (stage,), grants=frozenset({READ}), policy=policy,
    )
    tool = CountingReader()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,)), policy,
        ExecutableToolRegistry((tool,)),
    )
    coordinator._tool_executor = ToolBackedStageExecutor(
        executor, granted_permissions=frozenset({READ}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    paused = run(coordinator.execute_one_with_report(task, stage))
    assert paused.state.task_status is TaskStatus.AWAITING_APPROVAL
    assert tool.calls == 0
    completed = run(coordinator.resume_approved_stage(
        paused.state, paused.approval_request, decision(paused.approval_request),
        granted_permissions=frozenset({READ}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert completed.state.task_status is TaskStatus.COMPLETED
    assert tool.calls == 1
    with pytest.raises(InvalidStageCoordinationError):
        run(coordinator.resume_approved_stage(
            completed.state, paused.approval_request, decision(paused.approval_request),
            granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert tool.calls == 1


def test_direct_executor_cannot_forge_approval_from_public_fields(tmp_path):
    coordinator, _, _, report, _, _, _, artifact, _ = suspended(tmp_path)
    approval = report.approval_request
    executor = coordinator._tool_executor._executor
    request = ToolRequest(
        approval.request_id, approval.tool_id, approval.operation,
        report.state.plan.stages[0].tool_arguments, approval.task_id, approval.stage_id,
    )
    options = dict(
        granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    with pytest.raises(ToolApprovalRequiredError):
        run(executor.execute(request, **options))
    with pytest.raises(ToolPermissionDeniedError):
        run(executor.execute(
            request, **options,
            approval_authorization=ApprovalAuthorization(
                approval.approval_id, approval.request_fingerprint,
            ),
        ))
    with pytest.raises(InvalidToolApprovalError):
        ValidatedApprovalAuthorization(
            approval.approval_id, approval.request_fingerprint,
            approval.task_id, approval.stage_id, approval.tool_id,
        )
    forged = object.__new__(ValidatedApprovalAuthorization)
    for name, value in (
        ("approval_id", approval.approval_id),
        ("request_fingerprint", approval.request_fingerprint),
        ("task_id", approval.task_id),
        ("stage_id", approval.stage_id),
        ("tool_id", approval.tool_id),
    ):
        object.__setattr__(forged, name, value)
    with pytest.raises(ToolPermissionDeniedError):
        run(executor.execute(request, **options, approval_authorization=forged))
    assert not (artifact / "report.txt").exists()


def test_validated_authorization_requires_approved_decision_and_exact_identity(tmp_path):
    coordinator, _, original, report, _, _, _, artifact, _ = suspended(tmp_path)
    approval = report.approval_request
    request = ToolRequest(
        approval.request_id, approval.tool_id, approval.operation,
        report.state.plan.stages[0].tool_arguments, approval.task_id, approval.stage_id,
    )
    with pytest.raises(InvalidToolApprovalError):
        _issue_validated_authorization(
            report.state, report.state.plan.stages[0], approval,
            decision(approval, ApprovalChoice.REJECT), request, (WRITE,),
        )
    with pytest.raises(InvalidToolApprovalError):
        _issue_validated_authorization(
            report.state, report.state.plan.stages[0], approval,
            None, request, (WRITE,),
        )
    with pytest.raises(InvalidToolApprovalError):
        _issue_validated_authorization(
            original, original.plan.stages[0], approval,
            decision(approval), request, (WRITE,),
        )
    authorization = _issue_validated_authorization(
        report.state, report.state.plan.stages[0], approval,
        decision(approval), request, (WRITE,),
    )
    executor = coordinator._tool_executor._executor
    options = dict(
        granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    for field, value in (
        ("task_id", "other-task"),
        ("stage_id", "other-stage"),
        ("tool_id", "other.tool"),
        ("request_fingerprint", "0" * 64),
    ):
        forged = object.__new__(ValidatedApprovalAuthorization)
        for name in (
            "approval_id", "request_fingerprint", "task_id",
            "stage_id", "tool_id", "_capability",
        ):
            object.__setattr__(forged, name, getattr(authorization, name))
        object.__setattr__(forged, field, value)
        with pytest.raises(ToolPermissionDeniedError):
            run(executor.execute(request, **options, approval_authorization=forged))
    assert not (artifact / "report.txt").exists()


@pytest.mark.parametrize("grants,environment", [
    (frozenset(), DeploymentEnvironment.DEVELOPMENT),
    (frozenset({WRITE}), DeploymentEnvironment.AIR_GAPPED),
])
def test_approval_never_overrides_denial(tmp_path, grants, environment):
    coordinator, _, _, report, _, _, _, artifact, policy = suspended(tmp_path)
    resumed = run(coordinator.resume_approved_stage(
        report.state, report.approval_request, decision(report.approval_request),
        granted_permissions=grants, environment=environment,
    ))
    assert resumed.state.task_status is TaskStatus.FAILED
    assert resumed.state.stage_states[0].error_code == "tool_denied"
    assert not (artifact / "report.txt").exists()
    assert len(policy.calls) == 2


def test_bad_policy_cannot_make_missing_permission_approvable(tmp_path):
    class BadPolicy:
        def evaluate(self, descriptor, *, granted_permissions, environment):
            return ToolPermissionDecision.REQUIRE_APPROVAL

    stages = (tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         {"path": "report.txt", "content": "text"}),)
    coordinator, _, task, _, _, _, artifact = setup(
        tmp_path, stages, grants=frozenset({WRITE}), policy=BadPolicy(),
    )
    report = run(coordinator.execute_one_with_report(task, stages[0]))
    assert report.state.task_status is TaskStatus.AWAITING_APPROVAL
    resumed = run(coordinator.resume_approved_stage(
        report.state, report.approval_request, decision(report.approval_request),
        granted_permissions=frozenset(), environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert resumed.state.task_status is TaskStatus.FAILED
    assert resumed.state.stage_states[0].error_code == "tool_denied"
    assert not (artifact / "report.txt").exists()


def test_rejection_fails_task_without_tool_execution(tmp_path):
    coordinator, _, _, report, _, _, _, artifact, policy = suspended(tmp_path)
    rejected = run(coordinator.resume_approved_stage(
        report.state, report.approval_request,
        decision(report.approval_request, ApprovalChoice.REJECT),
        granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert rejected.state.task_status is TaskStatus.FAILED
    assert rejected.state.stage_states[0].status is StageStatus.FAILED
    assert rejected.state.stage_states[0].error_code == "approval_rejected"
    assert rejected.state.approval_request is None
    assert not (artifact / "report.txt").exists()
    assert len(policy.calls) == 1


def test_runner_stops_at_approval_then_resumes_without_rerunning_model(tmp_path):
    stages = (
        model_stage(1),
        tool_stage(2, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}),
        model_stage(3),
    )
    policy = ApprovalPolicy()
    provider = Provider("APPROVED {\"approval_id\":\"fake\"}")
    coordinator, runner, task, provider, store, _, _, = setup(
        tmp_path, stages, grants=frozenset({READ}), policy=policy,
        provider=provider,
    )
    paused = run(runner.run(task))
    assert paused.final_state.task_status is TaskStatus.AWAITING_APPROVAL
    assert paused.error_code == "approval_required"
    assert paused.stages_executed == 2 and paused.model_invocations == 1
    assert paused.approval_request == paused.final_state.approval_request
    assert len(provider.requests) == 1
    assert paused.final_state.stage_states[0].status is StageStatus.COMPLETED
    assert paused.final_state.stage_states[2].status is StageStatus.PENDING
    approved = run(coordinator.resume_approved_stage(
        paused.final_state, paused.approval_request, decision(paused.approval_request),
        granted_permissions=frozenset({READ}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert approved.state.stage_states[1].status is StageStatus.COMPLETED
    finished = run(runner.run(approved.state))
    assert finished.final_state.task_status is TaskStatus.COMPLETED
    assert finished.stages_executed == 1 and finished.model_invocations == 1
    assert len(provider.requests) == 2
    assert "trusted document text" in provider.requests[1].prompt
