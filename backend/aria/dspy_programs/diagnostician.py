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
    - tool_misuse         : wrong tool called, wrong args, or wrong sequence
    - context_overflow    : agent contradicts/repeats earlier state; constraint violations
    - goal_misalignment   : agent optimises a proxy metric instead of the real objective
    - hallucination_loop  : agent repeats confident false information across turns
    - none                : no significant failure detected

    Manifestation (secondary behavioural pattern, can co-occur with any root cause):
    - tool_misuse | hallucination_loop | none
    """

    task_description: str = dspy.InputField(desc="Original task the agent was given")
    observer_flags: str = dspy.InputField(desc="JSON list of anomaly flags from Observer")
    critic_scores: str = dspy.InputField(desc="JSON of correctness/completeness/efficiency/safety scores")
    trace_summary: str = dspy.InputField(desc="Summarised executor tool-call trace")

    failure_class: str = dspy.OutputField(
        desc=f"Primary failure class — exactly one of: {', '.join(FAILURE_CLASSES)}"
    )
    failure_manifestation: str = dspy.OutputField(
        desc=f"Secondary behavioural pattern — exactly one of: {', '.join(MANIFESTATIONS)}"
    )
    confidence: str = dspy.OutputField(desc="Confidence score as a decimal string, e.g. '0.85'")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning for this diagnosis")


class DiagnosticProgram(dspy.Module):
    def __init__(self) -> None:
        self.cot = dspy.ChainOfThought(DiagnoseFailure)

    def forward(
        self,
        task_description: str,
        observer_flags: str,
        critic_scores: str,
        trace_summary: str,
    ) -> dspy.Prediction:
        return self.cot(
            task_description=task_description,
            observer_flags=observer_flags,
            critic_scores=critic_scores,
            trace_summary=trace_summary,
        )


def build_lm(api_key: str, model: str = "groq/llama-3.3-70b-versatile") -> dspy.LM:
    return dspy.LM(model, api_key=api_key, temperature=0)
