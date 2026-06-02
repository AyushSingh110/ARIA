from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from aria.config import get_settings
from aria.memory.embeddings import get_engine
from aria.state.schema import ARIAState, ExecutorTraceEntry
from aria.tools import EXECUTOR_TOOLS
from aria.utils.display import console, print_agent_output, print_phase

_SYSTEM_PROMPT = """\
You are the Executor agent in ARIA. Your job is to complete the given subtask accurately using the tools available to you.

Available tools:
- calculator  : Evaluate mathematical expressions (e.g. "sqrt(144) + 2**8")
- web_search  : Search the web for information
- write_file  : Write content to a file in the workspace
- read_file   : Read content from a file in the workspace

Guidelines:
- Think step by step before calling a tool.
- Call only one tool at a time.
- When you have the final answer, state it clearly starting with "FINAL ANSWER:" on its own line.
- Do NOT repeat a tool call you have already made with identical arguments.
- Stay focused on the subtask — do not scope-creep.
"""

# Map tool name → callable for fast dispatch
_TOOL_MAP: dict[str, Any] = {t.name: t for t in EXECUTOR_TOOLS}


def _get_llm() -> Any:
    settings = get_settings()
    if settings.executor_provider == "groq":
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


def _args_hash(args: dict) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True).encode()).hexdigest()


def executor_node(state: ARIAState) -> dict:
    settings = get_settings()
    subtask = state["active_subtask"]
    if subtask is None:
        console.print("[red]Executor: no active subtask — skipping.[/red]")
        return {"current_phase": "observe", "executor_output": "No active subtask."}

    print_phase("execute", f"subtask='{subtask['description'][:60]}…'")

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(EXECUTOR_TOOLS)

    messages: list = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Subtask: {subtask['description']}"),
    ]

    trace: list[ExecutorTraceEntry] = []
    goal_embeddings: list[list[float]] = []
    seen_calls: set[str] = set()        # (tool_name, args_hash)
    engine = get_engine(settings.embedding_model)
    turn = 0
    final_output: str = "Executor reached max turns without a final answer."
    total_tokens = 0

    while turn < settings.executor_max_turns:
        t0 = time.monotonic()
        response = llm_with_tools.invoke(messages)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Token counting (best-effort — not all providers expose this)
        usage = getattr(response, "usage_metadata", None) or {}
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        total_tokens += tokens

        groq_delta = 1 if settings.executor_provider == "groq" else 0
        ollama_delta = 1 if settings.executor_provider == "ollama" else 0

        content_str = response.content if isinstance(response.content, str) else str(response.content)

        # ── No tool calls → final answer ─────────────────────────────────────
        if not response.tool_calls:
            final_output = content_str
            trace.append(
                ExecutorTraceEntry(
                    turn=turn,
                    tool_name="__llm__",
                    tool_args={},
                    tool_result="",
                    llm_output=content_str,
                    latency_ms=latency_ms,
                    token_count=tokens,
                )
            )
            # Embed the final output for drift scoring
            emb = engine.embed(content_str[:512])
            goal_embeddings.append(emb)
            break

        # ── Execute each tool call ────────────────────────────────────────────
        messages.append(response)
        tool_results_for_trace: list[str] = []

        for tc in response.tool_calls:
            tool_name: str = tc["name"]
            tool_args: dict = tc["args"]
            call_key = f"{tool_name}:{_args_hash(tool_args)}"

            t1 = time.monotonic()
            tool_fn = _TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result_str = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result_str = str(tool_fn.invoke(tool_args))
                except Exception as exc:
                    result_str = f"Error: {exc}"
            tool_latency = int((time.monotonic() - t1) * 1000)

            messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"])
            )
            tool_results_for_trace.append(result_str)

            seen_calls.add(call_key)

            console.print(
                f"  [dim]turn {turn} | tool={tool_name} | "
                f"args={json.dumps(tool_args)[:80]} | "
                f"result={result_str[:80]}[/dim]"
            )

        # Record one trace entry per turn (aggregate if multiple tool calls)
        primary_tc = response.tool_calls[0]
        emb = engine.embed(
            f"{primary_tc['name']} {json.dumps(primary_tc['args'])[:200]}"
        )
        goal_embeddings.append(emb)

        trace.append(
            ExecutorTraceEntry(
                turn=turn,
                tool_name=primary_tc["name"],
                tool_args=primary_tc["args"],
                tool_result=" | ".join(tool_results_for_trace),
                llm_output=content_str,
                latency_ms=latency_ms,
                token_count=tokens,
            )
        )
        turn += 1

    print_agent_output(
        "Executor",
        f"Turns used: {turn}\nOutput: {final_output[:300]}",
        color="green",
    )

    return {
        "executor_output": final_output,
        "executor_trace": trace,
        "executor_turn_count": turn,
        "goal_embedding_history": goal_embeddings,
        "current_phase": "observe",
        "total_tokens_used": state["total_tokens_used"] + total_tokens,
        "api_calls_groq": state["api_calls_groq"] + (turn + 1) * groq_delta,
        "api_calls_ollama": state["api_calls_ollama"] + (turn + 1) * ollama_delta,
    }
