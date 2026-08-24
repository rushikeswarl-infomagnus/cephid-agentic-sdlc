import unittest
from collections import Counter, deque

from agentic_sdlc import (
    ContractError,
    HandoffResult,
    Orchestrator,
    ResultStatus,
    Stage,
    WorkflowStatus,
    create_agents,
)


def result(status, summary=None, *, artifacts=(), issues=()):
    return HandoffResult(
        status=status,
        summary=summary or status.value,
        artifacts=artifacts,
        issues=issues,
        next_action="Continue automatically",
    )


class ScriptedExecutor:
    def __init__(self, responses):
        self.responses = {
            stage: deque(stage_responses)
            for stage, stage_responses in responses.items()
        }
        self.calls = []
        self.tasks = []

    def __call__(self, agent, task):
        self.calls.append(agent.stage)
        self.tasks.append(task)
        try:
            return self.responses[agent.stage].popleft()
        except (KeyError, IndexError) as error:
            raise AssertionError(f"No response for {agent.stage.value}") from error


def successful_responses():
    return {
        Stage.REQUIREMENTS: [result(ResultStatus.SUCCESS)],
        Stage.ARCHITECTURE: [result(ResultStatus.SUCCESS)],
        Stage.DEVELOPMENT: [result(ResultStatus.SUCCESS)],
        Stage.TEST: [result(ResultStatus.SUCCESS)],
        Stage.CODE_REVIEW: [result(ResultStatus.APPROVED)],
        Stage.SECURITY: [result(ResultStatus.SUCCESS)],
        Stage.DOCUMENTATION: [result(ResultStatus.SUCCESS)],
        Stage.RELEASE: [result(ResultStatus.SUCCESS)],
    }


class OrchestratorTests(unittest.TestCase):
    def test_runs_all_agents_and_preserves_handoff_context(self):
        responses = successful_responses()
        responses[Stage.REQUIREMENTS] = [
            result(ResultStatus.SUCCESS, artifacts=("stories.md",))
        ]
        executor = ScriptedExecutor(responses)

        state = Orchestrator(create_agents(executor)).start("Add audit logging")

        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(executor.calls, list(Stage))
        self.assertEqual(len(state.handoffs), 8)
        self.assertEqual(state.handoffs[-1].to_stage, None)
        self.assertEqual(state.artifacts, ["stories.md"])
        self.assertEqual(executor.tasks[1].artifacts, ("stories.md",))
        self.assertEqual(len(executor.tasks[-1].history), 7)

    def test_failed_quality_gates_return_to_development(self):
        responses = successful_responses()
        responses[Stage.DEVELOPMENT] = [
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
        ]
        responses[Stage.TEST] = [
            result(ResultStatus.FAILED, issues=("Tests failed",)),
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
        ]
        responses[Stage.CODE_REVIEW] = [
            result(ResultStatus.CHANGES_REQUIRED, issues=("Simplify API",)),
            result(ResultStatus.APPROVED),
            result(ResultStatus.APPROVED),
        ]
        responses[Stage.SECURITY] = [
            result(ResultStatus.FAILED, issues=("Validate input",)),
            result(ResultStatus.SUCCESS),
        ]
        executor = ScriptedExecutor(responses)

        state = Orchestrator(create_agents(executor)).start("Create an API")

        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        counts = Counter(executor.calls)
        self.assertEqual(counts[Stage.REQUIREMENTS], 1)
        self.assertEqual(counts[Stage.ARCHITECTURE], 1)
        self.assertEqual(counts[Stage.DEVELOPMENT], 4)
        self.assertEqual(counts[Stage.TEST], 4)
        self.assertEqual(counts[Stage.CODE_REVIEW], 3)
        self.assertEqual(counts[Stage.SECURITY], 2)
        routes = [
            (record.from_stage, record.to_stage) for record in state.handoffs
        ]
        self.assertIn((Stage.TEST, Stage.DEVELOPMENT), routes)
        self.assertIn((Stage.CODE_REVIEW, Stage.DEVELOPMENT), routes)
        self.assertIn((Stage.SECURITY, Stage.DEVELOPMENT), routes)

    def test_blocked_workflow_resumes_with_human_input(self):
        responses = successful_responses()
        responses[Stage.REQUIREMENTS] = [
            result(ResultStatus.BLOCKED, issues=("Target users are unknown",)),
            result(ResultStatus.SUCCESS),
        ]
        executor = ScriptedExecutor(responses)
        orchestrator = Orchestrator(create_agents(executor))

        blocked = orchestrator.start("Create a dashboard")
        self.assertEqual(blocked.status, WorkflowStatus.BLOCKED)
        self.assertEqual(blocked.current_stage, Stage.REQUIREMENTS)

        completed = orchestrator.provide_input(
            blocked.workflow_id, {"targetUsers": "release managers"}
        )

        self.assertEqual(completed.status, WorkflowStatus.COMPLETED)
        self.assertEqual(executor.tasks[1].context["targetUsers"], "release managers")

    def test_human_input_resets_the_blocked_stage_retry_budget(self):
        responses = successful_responses()
        responses[Stage.REQUIREMENTS] = [
            result(ResultStatus.FAILED),
            result(ResultStatus.BLOCKED),
            result(ResultStatus.FAILED),
            result(ResultStatus.SUCCESS),
        ]
        executor = ScriptedExecutor(responses)
        orchestrator = Orchestrator(create_agents(executor), max_retries=1)

        blocked = orchestrator.start("Create a dashboard")
        completed = orchestrator.provide_input(
            blocked.workflow_id, {"audience": "operators"}
        )

        self.assertEqual(blocked.status, WorkflowStatus.BLOCKED)
        self.assertEqual(completed.status, WorkflowStatus.COMPLETED)

    def test_retry_limit_fails_repeated_failure(self):
        responses = successful_responses()
        responses[Stage.DEVELOPMENT] = [
            result(ResultStatus.SUCCESS),
            result(ResultStatus.SUCCESS),
        ]
        responses[Stage.TEST] = [
            result(ResultStatus.FAILED),
            result(ResultStatus.FAILED),
        ]
        executor = ScriptedExecutor(responses)

        state = Orchestrator(create_agents(executor), max_retries=1).start(
            "Create a service"
        )

        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.current_stage, Stage.TEST)
        self.assertIn("Retry limit exceeded for test", state.issues)
        self.assertEqual(state.handoffs[-1].to_stage, None)

    def test_mapping_contract_is_validated_and_serialized(self):
        contract = {
            "status": "SUCCESS",
            "summary": "Requirements complete",
            "artifacts": ["stories.md"],
            "issues": [],
            "nextAction": "Start architecture",
        }

        self.assertEqual(HandoffResult.from_dict(contract).to_dict(), contract)
        with self.assertRaises(ContractError):
            HandoffResult.from_dict({"status": "SUCCESS"})

    def test_requires_exactly_one_agent_for_every_stage(self):
        agents = create_agents(lambda agent, task: result(ResultStatus.SUCCESS))

        with self.assertRaisesRegex(ValueError, "Missing agents"):
            Orchestrator(agents[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate agent"):
            Orchestrator((*agents, agents[0]))


if __name__ == "__main__":
    unittest.main()
