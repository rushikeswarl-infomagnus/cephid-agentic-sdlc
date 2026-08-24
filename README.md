# Agentic AI SDLC Orchestrator

A dependency-free Python orchestration core for automatically handing work between
requirements, architecture, development, test, code review, security,
documentation, and release agents.

## Behavior

- Every agent receives the original requirement, shared context, accumulated
  artifacts and issues, and the complete handoff history.
- Successful stages advance automatically.
- Failed tests, rejected reviews, and failed security checks return to
  development. The changed implementation then passes through the downstream
  quality gates again without repeating requirements or architecture.
- Agent failures are retried up to a configurable limit, then the workflow ends
  with `FAILED`.
- `BLOCKED` pauses a workflow for genuine human clarification or approval;
  `provide_input` resumes it from the same stage.
- Every transition, retry, pause, and final delivery is recorded in
  `WorkflowState.handoffs`.

Security agents represent `PASS` with `SUCCESS` and `FAIL` with `FAILED` so every
response uses the common handoff contract:

```json
{
  "status": "SUCCESS|FAILED|BLOCKED|APPROVED|CHANGES_REQUIRED",
  "summary": "...",
  "artifacts": [],
  "issues": [],
  "nextAction": "..."
}
```

## Integrate an agent runtime

Provide one executor that invokes your model and tools. The specialized agent
supplies its role instructions; `AgentTask` supplies the preserved workflow
context.

```python
from agentic_sdlc import (
    HandoffResult,
    Orchestrator,
    ResultStatus,
    Stage,
    create_agents,
)


def execute(agent, task):
    # Replace this example with an LLM/tool call using:
    # agent.name, agent.instructions, and task.
    status = (
        ResultStatus.APPROVED
        if agent.stage is Stage.CODE_REVIEW
        else ResultStatus.SUCCESS
    )
    return HandoffResult(
        status=status,
        summary=f"{agent.name} completed",
        artifacts=(),
        issues=(),
        next_action="Continue automatically",
    )


orchestrator = Orchestrator(create_agents(execute), max_retries=2)
state = orchestrator.start(
    "Add an authenticated reporting endpoint",
    context={"repository": "example/service"},
)
print(state.to_dict())
```

An executor may return either `HandoffResult` or a mapping matching the JSON
contract. Use a custom `WorkflowStore` implementation when state must survive
process restarts.

## Test

Requires Python 3.10 or newer:

```bash
python -m unittest discover -s tests -v
```
