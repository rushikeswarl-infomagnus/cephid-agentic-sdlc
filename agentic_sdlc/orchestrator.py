"""A small, provider-neutral Agentic AI SDLC orchestrator."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4


class ContractError(ValueError):
    """Raised when an agent response does not satisfy the handoff contract."""


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    APPROVED = "APPROVED"
    CHANGES_REQUIRED = "CHANGES_REQUIRED"


class Stage(str, Enum):
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    DEVELOPMENT = "development"
    TEST = "test"
    CODE_REVIEW = "code_review"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    RELEASE = "release"


class WorkflowStatus(str, Enum):
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class HandoffResult:
    """Structured result returned by every specialized agent."""

    status: ResultStatus
    summary: str
    artifacts: tuple[Any, ...] = ()
    issues: tuple[str, ...] = ()
    next_action: str = ""

    def __post_init__(self) -> None:
        try:
            status = ResultStatus(self.status)
        except ValueError as error:
            raise ContractError(f"Unknown result status: {self.status!r}") from error
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ContractError("summary must be a non-empty string")
        if isinstance(self.artifacts, (str, bytes)) or not isinstance(
            self.artifacts, Sequence
        ):
            raise ContractError("artifacts must be a sequence")
        if isinstance(self.issues, (str, bytes)) or not isinstance(
            self.issues, Sequence
        ):
            raise ContractError("issues must be a sequence")
        if not all(isinstance(issue, str) for issue in self.issues):
            raise ContractError("issues must contain only strings")
        if not isinstance(self.next_action, str) or not self.next_action.strip():
            raise ContractError("nextAction must be a non-empty string")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "issues", tuple(self.issues))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HandoffResult:
        required = {"status", "summary", "artifacts", "issues", "nextAction"}
        missing = required.difference(value)
        if missing:
            raise ContractError(
                f"Agent result is missing fields: {', '.join(sorted(missing))}"
            )
        return cls(
            status=value["status"],
            summary=value["summary"],
            artifacts=value["artifacts"],
            issues=value["issues"],
            next_action=value["nextAction"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
            "issues": list(self.issues),
            "nextAction": self.next_action,
        }


@dataclass(frozen=True)
class HandoffRecord:
    sequence: int
    from_stage: Stage
    to_stage: Stage | None
    result: HandoffResult
    reason: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "fromStage": self.from_stage.value,
            "toStage": self.to_stage.value if self.to_stage else None,
            "result": self.result.to_dict(),
            "reason": self.reason,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class AgentTask:
    """Context transferred to an agent during a handoff."""

    workflow_id: str
    requirement: str
    context: Mapping[str, Any]
    artifacts: tuple[Any, ...]
    issues: tuple[str, ...]
    history: tuple[HandoffRecord, ...]


@dataclass
class WorkflowState:
    workflow_id: str
    requirement: str
    context: dict[str, Any]
    current_stage: Stage = Stage.REQUIREMENTS
    status: WorkflowStatus = WorkflowStatus.RUNNING
    artifacts: list[Any] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    handoffs: list[HandoffRecord] = field(default_factory=list)
    attempts: dict[Stage, int] = field(default_factory=dict)
    failures: dict[Stage, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflowId": self.workflow_id,
            "requirement": self.requirement,
            "context": copy.deepcopy(self.context),
            "currentStage": self.current_stage.value,
            "status": self.status.value,
            "artifacts": copy.deepcopy(self.artifacts),
            "issues": list(self.issues),
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "attempts": {
                stage.value: attempts for stage, attempts in self.attempts.items()
            },
            "failures": {
                stage.value: failures for stage, failures in self.failures.items()
            },
        }


class WorkflowStore(Protocol):
    def get(self, workflow_id: str) -> WorkflowState: ...

    def save(self, state: WorkflowState) -> None: ...


class InMemoryWorkflowStore:
    """Thread-safe storage suitable for a single orchestrator process."""

    def __init__(self) -> None:
        self._states: dict[str, WorkflowState] = {}
        self._lock = RLock()

    def get(self, workflow_id: str) -> WorkflowState:
        with self._lock:
            try:
                return self._states[workflow_id]
            except KeyError as error:
                raise KeyError(f"Unknown workflow: {workflow_id}") from error

    def save(self, state: WorkflowState) -> None:
        with self._lock:
            self._states[state.workflow_id] = state


AgentExecutor = Callable[
    ["SpecializedAgent", AgentTask], HandoffResult | Mapping[str, Any]
]


class SpecializedAgent:
    stage: Stage
    name: str
    instructions: str

    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor

    def execute(self, task: AgentTask) -> HandoffResult:
        result = self._executor(self, task)
        if isinstance(result, HandoffResult):
            return result
        if not isinstance(result, Mapping):
            raise ContractError("Agent executor must return a result object or mapping")
        return HandoffResult.from_dict(result)


class RequirementsAgent(SpecializedAgent):
    stage = Stage.REQUIREMENTS
    name = "Requirements Agent"
    instructions = (
        "Analyze the requirement, create user stories and acceptance criteria, and "
        "identify missing information. Return BLOCKED only when clarification is "
        "required to continue."
    )


class ArchitectAgent(SpecializedAgent):
    stage = Stage.ARCHITECTURE
    name = "Architect Agent"
    instructions = (
        "Analyze the existing codebase, design the technical solution, and identify "
        "the impacted components and files."
    )


class DeveloperAgent(SpecializedAgent):
    stage = Stage.DEVELOPMENT
    name = "Developer Agent"
    instructions = (
        "Implement the approved solution using existing project standards. Modify "
        "only files needed for the requirement and supplied feedback."
    )


class TestAgent(SpecializedAgent):
    stage = Stage.TEST
    name = "Test Agent"
    instructions = (
        "Create or update focused tests, run the relevant test suite, and analyze "
        "failures. Return FAILED when implementation changes are needed."
    )


class CodeReviewAgent(SpecializedAgent):
    stage = Stage.CODE_REVIEW
    name = "Code Review Agent"
    instructions = (
        "Review quality, correctness, maintainability, and best practices. Return "
        "APPROVED or CHANGES_REQUIRED with actionable issues."
    )


class SecurityAgent(SpecializedAgent):
    stage = Stage.SECURITY
    name = "Security Agent"
    instructions = (
        "Check vulnerabilities, secrets, authorization, input validation, and "
        "dependencies. Represent PASS as SUCCESS and FAIL as FAILED."
    )


class DocumentationAgent(SpecializedAgent):
    stage = Stage.DOCUMENTATION
    name = "Documentation Agent"
    instructions = "Update relevant documentation and release notes."


class ReleaseAgent(SpecializedAgent):
    stage = Stage.RELEASE
    name = "Release Agent"
    instructions = "Prepare the final delivery summary and release or PR information."


def create_agents(executor: AgentExecutor) -> tuple[SpecializedAgent, ...]:
    """Create all required agents using a shared model/tool executor."""

    return (
        RequirementsAgent(executor),
        ArchitectAgent(executor),
        DeveloperAgent(executor),
        TestAgent(executor),
        CodeReviewAgent(executor),
        SecurityAgent(executor),
        DocumentationAgent(executor),
        ReleaseAgent(executor),
    )


class Orchestrator:
    """Runs the SDLC state machine until delivery or genuine human intervention."""

    _successors = {
        Stage.REQUIREMENTS: Stage.ARCHITECTURE,
        Stage.ARCHITECTURE: Stage.DEVELOPMENT,
        Stage.DEVELOPMENT: Stage.TEST,
        Stage.TEST: Stage.CODE_REVIEW,
        Stage.CODE_REVIEW: Stage.SECURITY,
        Stage.SECURITY: Stage.DOCUMENTATION,
        Stage.DOCUMENTATION: Stage.RELEASE,
    }
    _allowed_statuses = {
        Stage.REQUIREMENTS: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.ARCHITECTURE: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.DEVELOPMENT: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.TEST: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.CODE_REVIEW: {
            ResultStatus.APPROVED,
            ResultStatus.CHANGES_REQUIRED,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.SECURITY: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.DOCUMENTATION: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
        Stage.RELEASE: {
            ResultStatus.SUCCESS,
            ResultStatus.FAILED,
            ResultStatus.BLOCKED,
        },
    }

    def __init__(
        self,
        agents: Iterable[SpecializedAgent],
        *,
        store: WorkflowStore | None = None,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._agents: dict[Stage, SpecializedAgent] = {}
        for agent in agents:
            if agent.stage in self._agents:
                raise ValueError(f"Duplicate agent for stage: {agent.stage.value}")
            self._agents[agent.stage] = agent
        missing = set(Stage).difference(self._agents)
        if missing:
            names = ", ".join(sorted(stage.value for stage in missing))
            raise ValueError(f"Missing agents for stages: {names}")
        self._store = store or InMemoryWorkflowStore()
        self._max_retries = max_retries
        self._lock = RLock()

    def start(
        self, requirement: str, *, context: Mapping[str, Any] | None = None
    ) -> WorkflowState:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("requirement must be a non-empty string")
        state = WorkflowState(
            workflow_id=str(uuid4()),
            requirement=requirement,
            context=copy.deepcopy(dict(context or {})),
        )
        self._store.save(state)
        return self.run(state.workflow_id)

    def run(self, workflow_id: str) -> WorkflowState:
        with self._lock:
            state = self._store.get(workflow_id)
            if state.status is not WorkflowStatus.RUNNING:
                return copy.deepcopy(state)

            while state.status is WorkflowStatus.RUNNING:
                stage = state.current_stage
                result = self._invoke(self._agents[stage], self._task_for(state))
                state.attempts[stage] = state.attempts.get(stage, 0) + 1
                state.artifacts.extend(copy.deepcopy(result.artifacts))
                state.issues.extend(result.issues)

                if result.status not in self._allowed_statuses[stage]:
                    result = HandoffResult(
                        status=ResultStatus.BLOCKED,
                        summary=f"{self._agents[stage].name} returned an invalid status",
                        artifacts=result.artifacts,
                        issues=(
                            f"{result.status.value} is not valid for {stage.value}",
                        ),
                        next_action="Correct the agent response contract and resume",
                    )
                    state.issues.extend(result.issues)

                is_failure = result.status in {
                    ResultStatus.FAILED,
                    ResultStatus.CHANGES_REQUIRED,
                }
                if is_failure:
                    state.failures[stage] = state.failures.get(stage, 0) + 1
                elif result.status is not ResultStatus.BLOCKED:
                    state.failures[stage] = 0

                if is_failure and state.failures[stage] > self._max_retries:
                    next_stage = None
                    reason = f"Retry limit exceeded for {stage.value}"
                    state.issues.append(reason)
                    state.status = WorkflowStatus.FAILED
                else:
                    next_stage, reason = self._next_stage(stage, result.status)
                    if result.status is ResultStatus.BLOCKED:
                        state.status = WorkflowStatus.BLOCKED
                    elif stage is Stage.RELEASE and result.status is ResultStatus.SUCCESS:
                        state.status = WorkflowStatus.COMPLETED

                state.handoffs.append(
                    HandoffRecord(
                        sequence=len(state.handoffs) + 1,
                        from_stage=stage,
                        to_stage=next_stage,
                        result=result,
                        reason=reason,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                if next_stage is not None:
                    state.current_stage = next_stage
                self._store.save(state)

            return copy.deepcopy(state)

    def provide_input(
        self, workflow_id: str, context: Mapping[str, Any]
    ) -> WorkflowState:
        """Resume a blocked workflow after clarification or approval is supplied."""

        with self._lock:
            state = self._store.get(workflow_id)
            if state.status is not WorkflowStatus.BLOCKED:
                raise ValueError("Only a blocked workflow can be resumed")
            state.context.update(copy.deepcopy(dict(context)))
            state.failures[state.current_stage] = 0
            state.status = WorkflowStatus.RUNNING
            self._store.save(state)
            return self.run(workflow_id)

    def get_state(self, workflow_id: str) -> WorkflowState:
        return copy.deepcopy(self._store.get(workflow_id))

    @staticmethod
    def _task_for(state: WorkflowState) -> AgentTask:
        return AgentTask(
            workflow_id=state.workflow_id,
            requirement=state.requirement,
            context=copy.deepcopy(state.context),
            artifacts=tuple(copy.deepcopy(state.artifacts)),
            issues=tuple(state.issues),
            history=tuple(copy.deepcopy(state.handoffs)),
        )

    @staticmethod
    def _invoke(agent: SpecializedAgent, task: AgentTask) -> HandoffResult:
        try:
            return agent.execute(task)
        except Exception as error:
            return HandoffResult(
                status=ResultStatus.FAILED,
                summary=f"{agent.name} execution failed",
                issues=(f"{type(error).__name__}: {error}",),
                next_action="Retry the agent with the preserved workflow context",
            )

    @classmethod
    def _next_stage(
        cls, stage: Stage, status: ResultStatus
    ) -> tuple[Stage | None, str]:
        if status is ResultStatus.BLOCKED:
            return None, "Human clarification or approval required"
        if status is ResultStatus.FAILED:
            if stage in {Stage.TEST, Stage.SECURITY}:
                return Stage.DEVELOPMENT, f"{stage.value} failed; implementation required"
            return stage, f"{stage.value} failed; retrying the responsible agent"
        if status is ResultStatus.CHANGES_REQUIRED:
            return Stage.DEVELOPMENT, "Code review requested implementation changes"
        if stage is Stage.RELEASE:
            return None, "Delivery completed"
        return cls._successors[stage], f"{stage.value} completed"
