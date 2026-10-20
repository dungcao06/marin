# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural probes that grade agent protocol behavior separately from answer correctness.

Each probe pairs a minimal task with a grader over a :class:`ProbeEvidence` bundle: the request
settings, the raw response, the parsed tool calls, the executor events, the URL accesses observed by
the task's endpoint, and the files read from the task environment after the agent stopped. Graders
never read the model's account of what it did. A narrated shell session, tool-call markup in
ordinary text, or "I wrote the file" in chat produces no event or file, so it fails.

Every result reports three axes independently:

* ``protocol_compliance``: the check's required behavior happened (a tool ran, a URL was fetched,
  no reasoning was emitted).
* ``submission_completeness``: the answer was delivered where the task asked for it.
* ``answer_correctness``: the delivered answer matches the expected one.

An axis is ``None`` when the check does not assess it or a prerequisite failed. A prerequisite
failure (a serving or extraction error, an unsupported setting, or evidence that was never captured)
sets ``not_evaluable`` instead of a model-violation status.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from finestore.eval import EvalSample, EvaluationStore, Grading, Message, SampleKind
from pydantic import BaseModel, ConfigDict

EVIDENCE_SCHEMA_VERSION = 1
PROBE_TASK_PREFIX = "protocol_probe"
GRADING_METHOD = "protocol_probe:v1"

ANSWER_PATH = "/workspace/answer.txt"
# Where the tool-execution task environment writes the per-trial nonce.
NONCE_PATH = "/run/probe/nonce"
SHELL_TOOL = "bash"
FETCH_TOOLS = ("fetch", "bash")


class ProbeAxis(StrEnum):
    """An independently reported result dimension; see the module docstring."""

    PROTOCOL_COMPLIANCE = "protocol_compliance"
    SUBMISSION_COMPLETENESS = "submission_completeness"
    ANSWER_CORRECTNESS = "answer_correctness"


class ProbeCheck(StrEnum):
    REQUIRED_FILE = "required_file"
    TOOL_EXECUTION = "tool_execution"
    THINKING_DISABLED = "thinking_disabled"
    URL_LOADING = "url_loading"


class ServingOutcome(StrEnum):
    """Whether the serving stack returned a gradeable response."""

    OK = "ok"
    SERVER_ERROR = "server_error"
    EXTRACTION_ERROR = "extraction_error"
    UNSUPPORTED_SETTING = "unsupported_setting"


class ExecutionStatus(StrEnum):
    """The executor's lifecycle outcome for one dispatched tool call."""

    COMPLETED = "completed"
    EXECUTOR_ERROR = "executor_error"
    TIMED_OUT = "timed_out"


class SubmissionStatus(StrEnum):
    SUBMITTED = "submitted"
    EMPTY = "empty"
    MISSING = "missing"


class ToolExecutionStatus(StrEnum):
    EXECUTED = "executed"
    NO_INVOCATION = "no_invocation"
    INVALID_INVOCATION = "invalid_invocation"
    NOT_DISPATCHED = "not_dispatched"
    EXECUTOR_ERROR = "executor_error"
    TIMED_OUT = "timed_out"


class ThinkingStatus(StrEnum):
    RESPECTED = "respected"
    REASONING_FIELD = "reasoning_field"
    REASONING_DELIMITER = "reasoning_delimiter"


class RetrievalStatus(StrEnum):
    RETRIEVED = "retrieved"
    NOT_ATTEMPTED = "not_attempted"
    RETRIEVAL_FAILED = "retrieval_failed"


class RequestSettings(BaseModel):
    """The request knobs the probe pinned, recorded as sent."""

    model_config = ConfigDict(frozen=True)

    model: str
    chat_template: str
    enable_thinking: bool | None = None
    tools: tuple[str, ...] = ()


class ParsedToolCall(BaseModel):
    """A tool call the parser extracted from the generated text.

    ``arguments_valid`` is false when the parser found a call but could not decode its arguments.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, object] = {}
    arguments_valid: bool = True


class ExecutionEvent(BaseModel):
    """The executor's record of running one parsed call."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    status: ExecutionStatus
    exit_code: int | None = None
    output: str = ""


class UrlAccess(BaseModel):
    """One request seen by the task's URL endpoint or recorded by the retrieval executor."""

    model_config = ConfigDict(frozen=True)

    url: str
    succeeded: bool
    error: str | None = None


class CheckedFile(BaseModel):
    """A file read from the task environment after the agent stopped."""

    model_config = ConfigDict(frozen=True)

    path: str
    exists: bool
    content: str = ""


class ProbeEvidence(BaseModel):
    """Everything a probe grader may read for one trial.

    ``raw_response_captured``, ``execution_captured``, and ``url_access_captured`` say whether the
    harness recorded pre-parser text, executor lifecycle, and endpoint access at all. An empty field
    without capture means unknown, not absent. ``reasoning_content`` is the parsed message's
    reasoning field; ``final_message`` is the parsed final assistant text.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = EVIDENCE_SCHEMA_VERSION
    request: RequestSettings
    serving: ServingOutcome = ServingOutcome.OK
    serving_error: str | None = None
    raw_response_captured: bool = False
    raw_response: str = ""
    final_message: str = ""
    reasoning_content: str | None = None
    parsed_tool_calls: tuple[ParsedToolCall, ...] = ()
    execution_captured: bool = False
    execution_events: tuple[ExecutionEvent, ...] = ()
    url_access_captured: bool = False
    url_accesses: tuple[UrlAccess, ...] = ()
    files: tuple[CheckedFile, ...] = ()


@dataclass(frozen=True)
class RequiredFileProbe:
    """Write a known answer to ``path``; only the file in the task environment counts."""

    instruction: str
    expected_answer: str
    path: str = ANSWER_PATH
    check: ProbeCheck = ProbeCheck.REQUIRED_FILE


@dataclass(frozen=True)
class ToolExecutionProbe:
    """Obtain a per-trial nonce that only an executed tool call can reveal."""

    instruction: str
    nonce: str
    tools: tuple[str, ...] = (SHELL_TOOL,)
    check: ProbeCheck = ProbeCheck.TOOL_EXECUTION


@dataclass(frozen=True)
class ThinkingDisabledProbe:
    """Answer with ``enable_thinking=false`` under a pinned template.

    ``reasoning_delimiters`` are the template's reasoning-channel markers; any of them in the raw
    response is a violation. The check inspects emitted output only.
    """

    instruction: str
    expected_answer: str
    chat_template: str
    reasoning_delimiters: tuple[str, ...]
    check: ProbeCheck = ProbeCheck.THINKING_DISABLED


@dataclass(frozen=True)
class UrlLoadingProbe:
    """Report a per-trial nonce served only at ``url``, with a retrieval tool available."""

    instruction: str
    url: str
    nonce: str
    tools: tuple[str, ...] = FETCH_TOOLS
    check: ProbeCheck = ProbeCheck.URL_LOADING


Probe = RequiredFileProbe | ToolExecutionProbe | ThinkingDisabledProbe | UrlLoadingProbe


class ProbeResult(BaseModel):
    """One graded trial. ``status`` is the check-specific label; ``not_evaluable`` names a failed
    prerequisite, in which case ``status`` is ``None``."""

    model_config = ConfigDict(frozen=True)

    check: ProbeCheck
    status: SubmissionStatus | ToolExecutionStatus | ThinkingStatus | RetrievalStatus | None
    not_evaluable: str | None = None
    protocol_compliance: bool | None = None
    submission_completeness: bool | None = None
    answer_correctness: bool | None = None

    def axes(self) -> dict[ProbeAxis, bool]:
        """The assessed axes only; an unassessed axis is absent, never false."""
        values = {
            ProbeAxis.PROTOCOL_COMPLIANCE: self.protocol_compliance,
            ProbeAxis.SUBMISSION_COMPLETENESS: self.submission_completeness,
            ProbeAxis.ANSWER_CORRECTNESS: self.answer_correctness,
        }
        return {name: value for name, value in values.items() if value is not None}


def probe_tasks(*, nonce: str, url: str, chat_template: str, reasoning_delimiters: tuple[str, ...]) -> tuple[Probe, ...]:
    """The four minimal probe tasks for one trial.

    ``nonce`` must be fresh per trial so a correct answer cannot come from memory; ``url`` must serve
    exactly ``nonce`` from a task-local endpoint.
    """
    return (
        RequiredFileProbe(
            instruction=f"Write the word `marin` to {ANSWER_PATH}. Do not answer in chat; only the file is graded.",
            expected_answer="marin",
        ),
        ToolExecutionProbe(
            instruction=(
                f"Use the {SHELL_TOOL} tool to run `cat {NONCE_PATH}` and reply with only its output. "
                "The value cannot be guessed."
            ),
            nonce=nonce,
        ),
        ThinkingDisabledProbe(
            instruction="What is 17 + 25? Reply with only the number.",
            expected_answer="42",
            chat_template=chat_template,
            reasoning_delimiters=reasoning_delimiters,
        ),
        UrlLoadingProbe(
            instruction=f"Retrieve {url} and reply with only the token it contains. The token cannot be guessed.",
            url=url,
            nonce=nonce,
        ),
    )


def _not_evaluable(check: ProbeCheck, reason: str) -> ProbeResult:
    return ProbeResult(check=check, status=None, not_evaluable=reason)


def _serving_failure(check: ProbeCheck, evidence: ProbeEvidence) -> ProbeResult | None:
    if evidence.serving == ServingOutcome.OK:
        return None
    detail = f": {evidence.serving_error}" if evidence.serving_error else ""
    return _not_evaluable(check, f"{evidence.serving}{detail}")


def _missing_tool(probe: ToolExecutionProbe | UrlLoadingProbe, evidence: ProbeEvidence) -> ProbeResult | None:
    if set(probe.tools) & set(evidence.request.tools):
        return None
    return _not_evaluable(probe.check, f"request exposed none of the required tools {probe.tools}")


def _matches(text: str, expected: str) -> bool:
    return text.strip() == expected


def _grade_required_file(probe: RequiredFileProbe, evidence: ProbeEvidence) -> ProbeResult:
    checked = [file for file in evidence.files if file.path == probe.path]
    if not checked:
        return _not_evaluable(probe.check, f"{probe.path} was not checked in the task environment")
    file = checked[0]
    if not file.exists:
        status = SubmissionStatus.MISSING
    elif not file.content.strip():
        status = SubmissionStatus.EMPTY
    else:
        status = SubmissionStatus.SUBMITTED
    submitted = status == SubmissionStatus.SUBMITTED
    return ProbeResult(
        check=probe.check,
        status=status,
        submission_completeness=submitted,
        answer_correctness=submitted and _matches(file.content, probe.expected_answer),
    )


def _tool_status(probe: ToolExecutionProbe, evidence: ProbeEvidence) -> ToolExecutionStatus:
    calls = [call for call in evidence.parsed_tool_calls if call.name in probe.tools]
    if not evidence.parsed_tool_calls:
        return ToolExecutionStatus.NO_INVOCATION
    valid = [call for call in calls if call.arguments_valid]
    if not valid:
        return ToolExecutionStatus.INVALID_INVOCATION
    events = {event.call_id: event for event in evidence.execution_events}
    statuses = [events[call.call_id].status for call in valid if call.call_id in events]
    if ExecutionStatus.COMPLETED in statuses:
        return ToolExecutionStatus.EXECUTED
    if ExecutionStatus.TIMED_OUT in statuses:
        return ToolExecutionStatus.TIMED_OUT
    if ExecutionStatus.EXECUTOR_ERROR in statuses:
        return ToolExecutionStatus.EXECUTOR_ERROR
    return ToolExecutionStatus.NOT_DISPATCHED


def _grade_tool_execution(probe: ToolExecutionProbe, evidence: ProbeEvidence) -> ProbeResult:
    if not evidence.execution_captured:
        return _not_evaluable(probe.check, "executor lifecycle was not captured")
    if (missing := _missing_tool(probe, evidence)) is not None:
        return missing
    status = _tool_status(probe, evidence)
    submitted = bool(evidence.final_message.strip())
    return ProbeResult(
        check=probe.check,
        status=status,
        protocol_compliance=status == ToolExecutionStatus.EXECUTED,
        submission_completeness=submitted,
        answer_correctness=submitted and _matches(evidence.final_message, probe.nonce),
    )


def _grade_thinking_disabled(probe: ThinkingDisabledProbe, evidence: ProbeEvidence) -> ProbeResult:
    if evidence.request.enable_thinking is not False:
        return _not_evaluable(probe.check, "request did not set enable_thinking=false")
    if evidence.request.chat_template != probe.chat_template:
        return _not_evaluable(
            probe.check, f"request used template {evidence.request.chat_template!r}, not {probe.chat_template!r}"
        )
    if not evidence.raw_response_captured:
        return _not_evaluable(probe.check, "raw response was not captured")
    if evidence.reasoning_content:
        status = ThinkingStatus.REASONING_FIELD
    elif any(delimiter in evidence.raw_response for delimiter in probe.reasoning_delimiters):
        status = ThinkingStatus.REASONING_DELIMITER
    else:
        status = ThinkingStatus.RESPECTED
    submitted = bool(evidence.final_message.strip())
    return ProbeResult(
        check=probe.check,
        status=status,
        protocol_compliance=status == ThinkingStatus.RESPECTED,
        submission_completeness=submitted,
        answer_correctness=submitted and _matches(evidence.final_message, probe.expected_answer),
    )


def _grade_url_loading(probe: UrlLoadingProbe, evidence: ProbeEvidence) -> ProbeResult:
    if not evidence.url_access_captured:
        return _not_evaluable(probe.check, "URL access was not captured")
    if (missing := _missing_tool(probe, evidence)) is not None:
        return missing
    accesses = [access for access in evidence.url_accesses if access.url == probe.url]
    if not accesses:
        status = RetrievalStatus.NOT_ATTEMPTED
    elif any(access.succeeded for access in accesses):
        status = RetrievalStatus.RETRIEVED
    else:
        status = RetrievalStatus.RETRIEVAL_FAILED
    submitted = bool(evidence.final_message.strip())
    return ProbeResult(
        check=probe.check,
        status=status,
        # A failed fetch is still an attempt; the failure is recorded in ``status``, not charged to the model.
        protocol_compliance=status != RetrievalStatus.NOT_ATTEMPTED,
        submission_completeness=submitted,
        answer_correctness=submitted and _matches(evidence.final_message, probe.nonce),
    )


def grade_probe(probe: Probe, evidence: ProbeEvidence) -> ProbeResult:
    """Grade one trial's evidence against its probe."""
    failure = _serving_failure(probe.check, evidence)
    if failure is not None:
        return failure
    match probe:
        case RequiredFileProbe():
            return _grade_required_file(probe, evidence)
        case ToolExecutionProbe():
            return _grade_tool_execution(probe, evidence)
        case ThinkingDisabledProbe():
            return _grade_thinking_disabled(probe, evidence)
        case UrlLoadingProbe():
            return _grade_url_loading(probe, evidence)


@dataclass(frozen=True)
class AxisRate:
    """The pass rate of one axis over the trials that assessed it."""

    passed: int
    assessed: int

    @property
    def rate(self) -> float | None:
        return self.passed / self.assessed if self.assessed else None


def probe_rates(results: Iterable[ProbeResult]) -> dict[ProbeCheck, dict[ProbeAxis, AxisRate]]:
    """Per-check, per-axis pass counts with their denominators. ``not_evaluable`` trials are excluded
    from every denominator."""
    counts: dict[ProbeCheck, dict[ProbeAxis, list[int]]] = {}
    for result in results:
        per_check = counts.setdefault(result.check, {})
        for axis, value in result.axes().items():
            tally = per_check.setdefault(axis, [0, 0])
            tally[0] += int(value)
            tally[1] += 1
    return {
        check: {axis: AxisRate(passed=passed, assessed=assessed) for axis, (passed, assessed) in axes.items()}
        for check, axes in counts.items()
    }


def probe_sample(
    probe: Probe, evidence: ProbeEvidence, result: ProbeResult, *, doc_id: str, evidence_uri: str
) -> EvalSample:
    """The normalized sample for one graded trial.

    ``correct`` is answer correctness alone, so a structural pass never raises answer accuracy. The
    grading detail carries the full result and the evidence bundle's URI.
    """
    detail = {**result.model_dump(mode="json"), "evidence_uri": evidence_uri}
    return EvalSample(
        task=f"{PROBE_TASK_PREFIX}/{probe.check}",
        doc_id=doc_id,
        kind=SampleKind.AGENTIC,
        prompt_messages=[Message(role="user", content=probe.instruction)],
        output=evidence.final_message or None,
        grading=Grading(
            method=GRADING_METHOD,
            metric=ProbeAxis.PROTOCOL_COMPLIANCE,
            score=None if result.protocol_compliance is None else float(result.protocol_compliance),
            passed=result.protocol_compliance,
            detail=json.dumps(detail, sort_keys=True),
        ),
        metrics={axis: float(value) for axis, value in result.axes().items()},
        correct=result.answer_correctness,
    )


def archive_probe(
    store: EvaluationStore, probe: Probe, evidence: ProbeEvidence, *, doc_id: str, trial_id: str
) -> ProbeResult:
    """Grade one trial, store its evidence bundle, and append the linked sample."""
    result = grade_probe(probe, evidence)
    evidence_uri = store.add_artifact(
        f"{trial_id}/{probe.check}/{doc_id}/probe_evidence.json",
        evidence.model_dump_json(indent=2).encode(),
        metadata={"check": probe.check, "doc_id": doc_id, "trial_id": trial_id},
    )
    store.add_sample(probe_sample(probe, evidence, result, doc_id=doc_id, evidence_uri=evidence_uri), trial_id=trial_id)
    return result
