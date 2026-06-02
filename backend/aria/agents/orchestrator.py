from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from aria.config import get_settings
from aria.memory.embeddings import get_engine
from aria.state.schema import ARIAState
from aria.utils.display import console, print_agent_output, print_phase

_SYSTEM_PROMPT = """\
You are the Orchestrator of ARIA — an autonomous multi-agent system for failure detection and self-correction.

Your responsibilities:
1. Receive a task description from the user.
2. Classify the task into exactly one category: code_generation | web_research | data_analysis | reasoning | tool_chaining
3. Decompose the task into clear, executable subtasks (Phase 1: produce exactly ONE subtask).
4. Specify which tools the Executor should use: calculator, web_search, write_file, read_file.

IMPORTANT: Respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.

Output schema:
{
  "task_class": "<one of the five classes>",
  "subtasks": [
    {
      "id": "subtask_1",
      "description": "<precise, self-contained instruction the Executor can act on>",
      "expected_tools": ["<tool1>", "<tool2>"],
      "success_criteria": "<how we know this subtask is complete>"
    }
  ],
  "reasoning": "<one sentence explaining the decomposition choice>"
}
"""


def _get_llm() -> Any:
    settings = get_settings()
    if settings.orchestrator_provider == "groq":
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
        )
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0,
    )


def _extract_json(raw: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    return json.loads(cleaned)


def orchestrator_node(state: ARIAState) -> dict:
    settings = get_settings()
    print_phase("decompose", f"task='{state['task_description'][:60]}…'")

    llm = _get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {state['task_description']}"),
    ]

    response = llm.invoke(messages)
    raw_content = response.content

    # Track API call counts
    groq_delta = 1 if settings.orchestrator_provider == "groq" else 0
    ollama_delta = 1 if settings.orchestrator_provider == "ollama" else 0

    try:
        parsed = _extract_json(raw_content)
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Orchestrator JSON parse error: {exc}[/red]")
        console.print(f"[dim]Raw response:\n{raw_content}[/dim]")
        # Fallback: create a single passthrough subtask
        parsed = {
            "task_class": "general",
            "subtasks": [
                {
                    "id": "subtask_1",
                    "description": state["task_description"],
                    "expected_tools": [],
                    "success_criteria": "Task completed without error.",
                }
            ],
            "reasoning": "Fallback decomposition due to parse error.",
        }

    subtasks = parsed.get("subtasks", [])
    task_class = parsed.get("task_class", "general")

    print_agent_output(
        "Orchestrator",
        f"Class: {task_class}\nSubtasks: {len(subtasks)}\n"
        f"Reasoning: {parsed.get('reasoning', '')}",
        color="cyan",
    )

    # Embed the task description for drift detection downstream
    engine = get_engine(get_settings().embedding_model)
    task_embedding = engine.embed(state["task_description"])

    return {
        "task_class": task_class,
        "subtasks": subtasks,
        "current_subtask_index": 0,
        "active_subtask": subtasks[0] if subtasks else None,
        "task_description_embedding": task_embedding,
        "current_phase": "execute",
        "api_calls_groq": state["api_calls_groq"] + groq_delta,
        "api_calls_ollama": state["api_calls_ollama"] + ollama_delta,
    }
