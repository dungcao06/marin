# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from finestore.eval import EvaluationStore, sample_from_archive_row
from finestore.reader import ReadView
from marin.evaluation.protocol_probes import (
    ANSWER_PATH,
    NONCE_PATH,
    AxisRate,
    CheckedFile,
    ExecutionEvent,
    ExecutionStatus,
    ParsedToolCall,
    ProbeCheck,
    ProbeEvidence,
    RequestSettings,
    RequiredFileProbe,
    RetrievalStatus,
    ServingOutcome,
    SubmissionStatus,
    ThinkingDisabledProbe,
    ThinkingStatus,
    ToolExecutionProbe,
    ToolExecutionStatus,
    UrlAccess,
    UrlLoadingProbe,
    archive_probe,
    grade_probe,
    probe_rates,
    probe_tasks,
)

NONCE = "q7f3k9"
URL = "http://probe.local/token"
TEMPLATE = "qwen3"
DELIMITERS = ("<think>", "</think>")

FILE_PROBE, TOOL_PROBE, THINKING_PROBE, URL_PROBE = probe_tasks(
    nonce=NONCE, url=URL, chat_template=TEMPLATE, reasoning_delimiters=DELIMITERS
)
assert isinstance(FILE_PROBE, RequiredFileProbe)
assert isinstance(TOOL_PROBE, ToolExecutionProbe)
assert isinstance(THINKING_PROBE, ThinkingDisabledProbe)
assert isinstance(URL_PROBE, UrlLoadingProbe)

REQUEST = RequestSettings(model="m", chat_template=TEMPLATE, enable_thinking=False, tools=("bash",))
NARRATED_SHELL = f"I'll run the command.\n$ cat {NONCE_PATH}\n{NONCE}x\n\nThe nonce is {NONCE}x."
BASH_CALL = ParsedToolCall(call_id="c1", name="bash", arguments={"command": f"cat {NONCE_PATH}"})


def evidence(**fields) -> ProbeEvidence:
    return ProbeEvidence(request=REQUEST, **fields)


def answer_file(content: str | None) -> CheckedFile:
    if content is None:
        return CheckedFile(path=ANSWER_PATH, exists=False)
    return CheckedFile(path=ANSWER_PATH, exists=True, content=content)


# (probe, evidence, status, not_evaluable reason fragment, protocol, submission, answer)
FIXTURES = {
    # Required-file submission.
    "file_written": (
        FILE_PROBE,
        evidence(files=(answer_file("marin\n"),)),
        SubmissionStatus.SUBMITTED,
        None,
        None,
        True,
        True,
    ),
    "file_written_wrong_answer": (
        FILE_PROBE,
        evidence(files=(answer_file("grug"),)),
        SubmissionStatus.SUBMITTED,
        None,
        None,
        True,
        False,
    ),
    "answer_only_in_chat": (
        FILE_PROBE,
        evidence(final_message=f"I wrote marin to {ANSWER_PATH}.", files=(answer_file(None),)),
        SubmissionStatus.MISSING,
        None,
        None,
        False,
        False,
    ),
    "empty_file": (FILE_PROBE, evidence(files=(answer_file("  \n"),)), SubmissionStatus.EMPTY, None, None, False, False),
    "file_never_checked": (FILE_PROBE, evidence(), None, ANSWER_PATH, None, None, None),
    # Actual tool execution.
    "tool_executed": (
        TOOL_PROBE,
        evidence(
            final_message=NONCE,
            parsed_tool_calls=(BASH_CALL,),
            execution_captured=True,
            execution_events=(
                ExecutionEvent(call_id="c1", status=ExecutionStatus.COMPLETED, exit_code=0, output=NONCE),
            ),
        ),
        ToolExecutionStatus.EXECUTED,
        None,
        True,
        True,
        True,
    ),
    "narrated_shell_session": (
        TOOL_PROBE,
        evidence(raw_response=NARRATED_SHELL, final_message=NARRATED_SHELL, execution_captured=True),
        ToolExecutionStatus.NO_INVOCATION,
        None,
        False,
        True,
        False,
    ),
    "tool_markup_and_invented_response_in_text": (
        TOOL_PROBE,
        evidence(
            raw_response=f'<tool_call>{{"name": "bash"}}</tool_call>\n<tool_response>{NONCE}</tool_response>\n{NONCE}',
            final_message=NONCE,
            execution_captured=True,
        ),
        ToolExecutionStatus.NO_INVOCATION,
        None,
        False,
        True,
        True,
    ),
    "malformed_call": (
        TOOL_PROBE,
        evidence(
            parsed_tool_calls=(ParsedToolCall(call_id="c1", name="bash", arguments_valid=False),),
            execution_captured=True,
        ),
        ToolExecutionStatus.INVALID_INVOCATION,
        None,
        False,
        False,
        False,
    ),
    "unknown_tool": (
        TOOL_PROBE,
        evidence(parsed_tool_calls=(ParsedToolCall(call_id="c1", name="python"),), execution_captured=True),
        ToolExecutionStatus.INVALID_INVOCATION,
        None,
        False,
        False,
        False,
    ),
    "call_never_dispatched": (
        TOOL_PROBE,
        evidence(parsed_tool_calls=(BASH_CALL,), execution_captured=True),
        ToolExecutionStatus.NOT_DISPATCHED,
        None,
        False,
        False,
        False,
    ),
    "executor_error": (
        TOOL_PROBE,
        evidence(
            parsed_tool_calls=(BASH_CALL,),
            execution_captured=True,
            execution_events=(ExecutionEvent(call_id="c1", status=ExecutionStatus.EXECUTOR_ERROR),),
        ),
        ToolExecutionStatus.EXECUTOR_ERROR,
        None,
        False,
        False,
        False,
    ),
    "executor_timeout": (
        TOOL_PROBE,
        evidence(
            parsed_tool_calls=(BASH_CALL,),
            execution_captured=True,
            execution_events=(ExecutionEvent(call_id="c1", status=ExecutionStatus.TIMED_OUT),),
        ),
        ToolExecutionStatus.TIMED_OUT,
        None,
        False,
        False,
        False,
    ),
    "execution_not_captured": (
        TOOL_PROBE,
        evidence(final_message=NONCE, parsed_tool_calls=(BASH_CALL,)),
        None,
        "executor lifecycle",
        None,
        None,
        None,
    ),
    # Thinking disabled.
    "no_reasoning_emitted": (
        THINKING_PROBE,
        evidence(raw_response="42", final_message="42"),
        ThinkingStatus.RESPECTED,
        None,
        True,
        True,
        True,
    ),
    "reasoning_field_leak": (
        THINKING_PROBE,
        evidence(raw_response="42", final_message="42", reasoning_content="17 plus 25 is 42"),
        ThinkingStatus.REASONING_FIELD,
        None,
        False,
        True,
        True,
    ),
    "reasoning_delimiter_leak": (
        THINKING_PROBE,
        evidence(raw_response="<think>17 plus 25</think>42", final_message="42"),
        ThinkingStatus.REASONING_DELIMITER,
        None,
        False,
        True,
        True,
    ),
    "server_error": (
        THINKING_PROBE,
        evidence(serving=ServingOutcome.SERVER_ERROR, serving_error="HTTP 500"),
        None,
        "server_error: HTTP 500",
        None,
        None,
        None,
    ),
    "extraction_error": (
        THINKING_PROBE,
        evidence(serving=ServingOutcome.EXTRACTION_ERROR, raw_response="<think>unterminated"),
        None,
        "extraction_error",
        None,
        None,
        None,
    ),
    "thinking_not_disabled_in_request": (
        THINKING_PROBE,
        ProbeEvidence(request=RequestSettings(model="m", chat_template=TEMPLATE), raw_response="<think>x</think>42"),
        None,
        "enable_thinking=false",
        None,
        None,
        None,
    ),
    # URL loading.
    "url_retrieved": (
        URL_PROBE,
        evidence(final_message=NONCE, url_access_captured=True, url_accesses=(UrlAccess(url=URL, succeeded=True),)),
        RetrievalStatus.RETRIEVED,
        None,
        True,
        True,
        True,
    ),
    "url_answer_without_retrieval": (
        URL_PROBE,
        evidence(final_message="The token is abc123.", url_access_captured=True),
        RetrievalStatus.NOT_ATTEMPTED,
        None,
        False,
        True,
        False,
    ),
    "url_retrieval_failed": (
        URL_PROBE,
        evidence(
            final_message="I could not reach the URL.",
            url_access_captured=True,
            url_accesses=(UrlAccess(url=URL, succeeded=False, error="connection refused"),),
        ),
        RetrievalStatus.RETRIEVAL_FAILED,
        None,
        True,
        True,
        False,
    ),
    "url_access_not_captured": (URL_PROBE, evidence(final_message=NONCE), None, "URL access", None, None, None),
}


@pytest.mark.parametrize(
    "probe, trial, status, reason, protocol, submission, answer", FIXTURES.values(), ids=FIXTURES.keys()
)
def test_grade_probe_labels_each_fixture(probe, trial, status, reason, protocol, submission, answer):
    result = grade_probe(probe, trial)

    assert result.check == probe.check
    assert result.status == status
    if reason is None:
        assert result.not_evaluable is None
    else:
        assert reason in result.not_evaluable
    assert (result.protocol_compliance, result.submission_completeness, result.answer_correctness) == (
        protocol,
        submission,
        answer,
    )


def test_probe_rates_exclude_unassessed_axes_and_not_evaluable_trials():
    results = [grade_probe(probe, trial) for probe, trial, *_ in FIXTURES.values()]

    rates = probe_rates(results)

    # Tool fixtures: 8 evaluable, one executed. Two answer correctly, one of them only by narration
    # through invented tool output, so a correct answer never implies protocol compliance.
    assert rates[ProbeCheck.TOOL_EXECUTION]["protocol_compliance"] == AxisRate(passed=1, assessed=8)
    assert rates[ProbeCheck.TOOL_EXECUTION]["answer_correctness"] == AxisRate(passed=2, assessed=8)
    # Required-file fixtures never assess protocol compliance; the unchecked file is excluded.
    assert "protocol_compliance" not in rates[ProbeCheck.REQUIRED_FILE]
    assert rates[ProbeCheck.REQUIRED_FILE]["submission_completeness"] == AxisRate(passed=2, assessed=4)
    assert rates[ProbeCheck.THINKING_DISABLED]["protocol_compliance"] == AxisRate(passed=1, assessed=3)
    assert rates[ProbeCheck.URL_LOADING]["protocol_compliance"] == AxisRate(passed=2, assessed=3)


def test_archive_probe_keeps_evidence_and_separates_answer_from_protocol(tmp_path):
    root = str(tmp_path / "results")
    probe, trial, *_ = FIXTURES["tool_markup_and_invented_response_in_text"]

    with EvaluationStore.open(root, writer_id="probe") as store:
        archive_probe(store, probe, trial, doc_id="tool-0", trial_id="trial-0")

    reader = ReadView(root)
    [row] = reader.scan("samples").to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(row)
    assert sample.task == "protocol_probe/tool_execution"
    assert sample.correct is True
    assert sample.grading.passed is False
    assert sample.metrics == {"protocol_compliance": 0.0, "submission_completeness": 1.0, "answer_correctness": 1.0}

    detail = json.loads(sample.grading.detail)
    assert detail["status"] == ToolExecutionStatus.NO_INVOCATION
    stored = reader.read_blob("trial-0/probe_evidence.json")
    assert ProbeEvidence.model_validate_json(stored) == trial
    assert detail["evidence_uri"].endswith("trial-0/probe_evidence.json")
