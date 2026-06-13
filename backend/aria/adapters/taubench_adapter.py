"""tau-bench adapter — drive an ARIA-configured LLM through a tau-bench episode.

tau-bench (sierra-research/tau-bench) emulates a multi-turn conversation between
a simulated user and a tool-using agent in a retail/airline domain. It is the
richest natural source of `goal_misalignment` (multi-turn, real policy to
follow) and surfaces long-horizon loops.

This adapter is intentionally thin: it runs one episode and returns the trace
in ARIA's ExecutorTraceEntry shape plus the env's ground-truth reward. The
ARIA diagnosis (Observer/Critic/Diagnostician) is applied by the runner
(scripts/taubench_run.py) on the returned trace, exactly like every other
benchmark — so tau-bench traces are diagnosed by the same pipeline.

The vendored tau-bench lives in third_party/tau-bench; we add it to sys.path
lazily so importing ARIA never hard-depends on it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from aria.config import get_settings
from aria.memory.embeddings import get_engine

# backend/aria/adapters/taubench_adapter.py -> parents[2] == backend/
_TAUBENCH_PATH = Path(__file__).resolve().parents[2] / "third_party" / "tau-bench"

_RESPOND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Send a natural-language message to the user.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
}

_SYSTEM_TEMPLATE = """\
You are a customer-service agent. Follow the domain policy exactly and use the
provided tools to help the user. Call exactly one tool per turn. To talk to the
user, call the `respond` tool. Do not invent information; look it up with tools.

Domain policy:
{wiki}
"""


def _ensure_taubench_on_path() -> None:
    import sys
    p = str(_TAUBENCH_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_llm(provider: str, model: str) -> Any:
    s = get_settings()
    if provider == "groq":
        return ChatGroq(api_key=s.groq_api_key, model=model or s.groq_model, temperature=0)
    return ChatOllama(base_url=s.ollama_base_url, model=model or s.ollama_model, temperature=0)


def run_episode(
    env_name: str,
    task_index: int,
    *,
    agent_provider: str = "groq",
    agent_model: str = "llama-3.1-8b-instant",
    user_model: str = "llama-3.3-70b-versatile",
    user_provider: str = "groq",
    task_split: str = "test",
    max_turns: int = 15,
) -> dict:
    """Run one tau-bench episode and return an ARIA-shaped trace + reward.

    Returns a dict with: trace, executor_output, reward (0/1 ground truth),
    done, turn_count, goal_embedding_history, task_instruction, run_error.
    """
    _ensure_taubench_on_path()
    from tau_bench.envs import get_env
    from tau_bench.types import Action

    settings = get_settings()
    # tau-bench's user simulator calls Groq via litellm, which reads the key
    # from os.environ (not ARIA's pydantic settings). Bridge it across.
    if settings.groq_api_key and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
    engine = get_engine(settings.embedding_model)

    env = get_env(
        env_name,
        user_strategy="llm",
        user_model=user_model,
        user_provider=user_provider,
        task_split=task_split,
        task_index=task_index,
    )
    reset = env.reset(task_index=task_index)
    user_msg = reset.observation
    task_instruction = getattr(env.task, "instruction", "")

    llm = _build_llm(agent_provider, agent_model)
    llm_with_tools = llm.bind_tools(env.tools_info + [_RESPOND_SCHEMA])

    messages: list = [
        SystemMessage(content=_SYSTEM_TEMPLATE.format(wiki=env.wiki)),
        HumanMessage(content=user_msg),
    ]

    trace: list[dict] = []
    goal_embeddings: list[list[float]] = []
    reward, done, run_error = 0.0, False, None
    final_output = "Episode ended without resolution."
    turn = 0

    while turn < max_turns and not done:
        t0 = time.monotonic()
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower() or "rate_limit" in msg.lower():
                raise  # let the runner back off / resume
            # malformed tool call from a weak model — capture as a trace turn
            latency = int((time.monotonic() - t0) * 1000)
            trace.append({
                "turn": turn, "tool_name": "__tool_use_failed__", "tool_args": {},
                "tool_result": msg[:300], "llm_output": "", "latency_ms": latency, "token_count": 0,
            })
            goal_embeddings.append(engine.embed("malformed tool call"))
            final_output = "Agent produced a malformed tool call the provider rejected."
            turn += 1
            break

        latency = int((time.monotonic() - t0) * 1000)
        content = response.content if isinstance(response.content, str) else str(response.content)
        messages.append(response)

        # Determine the single action this turn (tau-bench is one action/turn).
        if response.tool_calls:
            tc = response.tool_calls[0]
            action_name, action_args = tc["name"], dict(tc.get("args") or {})
            tool_call_id = tc["id"]
        else:
            # No tool call → treat free text as a message to the user.
            action_name, action_args, tool_call_id = "respond", {"content": content}, None

        try:
            resp = env.step(Action(name=action_name, kwargs=action_args))
            observation, reward, done = resp.observation, resp.reward, resp.done
        except Exception as exc:
            observation, reward, done = f"Error: {exc}", 0.0, False

        if tool_call_id is not None:
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
        else:
            messages.append(HumanMessage(content=str(observation)))

        trace.append({
            "turn": turn,
            "tool_name": action_name,
            "tool_args": action_args,
            "tool_result": str(observation)[:600],
            "llm_output": content,
            "latency_ms": latency,
            "token_count": (getattr(response, "usage_metadata", {}) or {}).get("total_tokens", 0),
        })
        goal_embeddings.append(engine.embed((content or str(observation))[:512]))

        if action_name == "respond":
            final_output = content or str(observation)
        turn += 1

    return {
        "trace": trace,
        "executor_output": final_output,
        "reward": float(reward),
        "done": bool(done),
        "turn_count": turn,
        "goal_embedding_history": goal_embeddings,
        "task_instruction": task_instruction,
        "run_error": run_error,
    }
