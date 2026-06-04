from __future__ import annotations

import dspy

FAILURE_CLASSES = [
    "prompt_drift",
    "tool_misuse",
    "context_overflow",
    "goal_misalignment",
    "hallucination_loop",
    "none",
]

MANIFESTATIONS = ["tool_misuse", "hallucination_loop", "none"]


class DiagnoseFailure(dspy.Signature):
    """Classify the root-cause failure of a multi-agent LLM system run.

    Failure classes:
    - prompt_drift        : agent behaviour deviates from original intent over turns
    - tool_misuse         : wrong tool called, wrong args, or tool returned Error messages
    - context_overflow    : agent repeats identical tool calls; hits turn budget limit
    - goal_misalignment   : agent completed task but missed explicit requirements (partial
                            completion, requirement omission, or wrong objective)
    - hallucination_loop  : agent asserts facts across turns without tool grounding
    - none                : no significant failure detected

    CRITICAL DISAMBIGUATION — tool_misuse vs goal_misalignment:
    - tool_misuse REQUIRES observable tool errors: "Error:" in trace, "unknown tool",
      "ValidationError", or tool_error_loop observer flags. If tools ran without errors
      but requirements were not satisfied → goal_misalignment, NOT tool_misuse.
    - requirement_satisfaction < 0.75 with no tool errors = goal_misalignment.
    - requirement_satisfaction >= 0.75 with no observer flags = none.

    Manifestation (secondary behavioural pattern):
    - tool_misuse | hallucination_loop | none
    """

    task_description: str = dspy.InputField(
        desc="Original task the agent was given"
    )
    observer_flags: str = dspy.InputField(
        desc="JSON list of anomaly flags from Observer (prompt_drift, tool_error_loop, "
             "tool_repetition, turn_budget_warning). Empty list means no behavioral anomaly."
    )
    critic_scores: str = dspy.InputField(
        desc="JSON of Critic v2 scores including requirement_satisfaction (0.0-1.0)"
    )
    requirement_summary: str = dspy.InputField(
        desc="Requirements checklist from Critic v2. Format: 'REQ: <text> [OK/MISS]' per line. "
             "MISS means the agent did not satisfy that requirement."
    )
    trace_summary: str = dspy.InputField(
        desc="Summarised executor tool-call trace. Look for 'Error:' messages to identify "
             "tool_misuse. Look for repeated identical calls for context_overflow."
    )

    failure_class: str = dspy.OutputField(
        desc=f"Primary failure class — exactly one of: {', '.join(FAILURE_CLASSES)}"
    )
    failure_manifestation: str = dspy.OutputField(
        desc=f"Secondary behavioural pattern — exactly one of: {', '.join(MANIFESTATIONS)}"
    )
    confidence: str = dspy.OutputField(
        desc="Confidence score as a decimal string, e.g. '0.85'"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning. Must cite specific evidence from trace, flags, "
             "or requirement_summary that justifies the classification."
    )


class DiagnosticProgram(dspy.Module):
    def __init__(self) -> None:
        self.cot = dspy.ChainOfThought(DiagnoseFailure)

    def forward(
        self,
        task_description: str,
        observer_flags: str,
        critic_scores: str,
        requirement_summary: str,
        trace_summary: str,
    ) -> dspy.Prediction:
        return self.cot(
            task_description=task_description,
            observer_flags=observer_flags,
            critic_scores=critic_scores,
            requirement_summary=requirement_summary,
            trace_summary=trace_summary,
        )


def build_lm(api_key: str, model: str = "groq/llama-3.3-70b-versatile") -> dspy.LM:
    return dspy.LM(model, api_key=api_key, temperature=0)
