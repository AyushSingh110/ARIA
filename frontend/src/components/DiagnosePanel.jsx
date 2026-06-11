import { useState } from "react";
import { diagnoseTrace, submitFeedback } from "../api";
import { classColor, classLabel } from "../constants";

const EXAMPLE_TOOLS = JSON.stringify(
  [
    {
      tool_name: "web_search",
      tool_args: { query: "compound interest formula" },
      tool_result: "1. Compound Interest — A = P(1 + r/n)^(nt) ...",
      turn: 0,
    },
  ],
  null,
  2
);

export default function DiagnosePanel({ onDiagnosed }) {
  const [task, setTask] = useState("");
  const [toolsJson, setToolsJson] = useState(EXAMPLE_TOOLS);
  const [finalOutput, setFinalOutput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [feedbackMsg, setFeedbackMsg] = useState(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    setFeedbackMsg(null);
    try {
      let toolCalls = [];
      if (toolsJson.trim()) {
        toolCalls = JSON.parse(toolsJson);
        if (!Array.isArray(toolCalls))
          throw new Error("Tool calls must be a JSON array.");
      }
      const res = await diagnoseTrace({
        task_description: task,
        tool_calls: toolCalls,
        final_output: finalOutput,
      });
      setResult(res);
      onDiagnosed?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const sendFeedback = async (correct) => {
    if (!result) return;
    try {
      let humanLabel = null;
      if (!correct) {
        humanLabel = window.prompt(
          "Correct failure class? (prompt_drift, tool_misuse, context_overflow, goal_misalignment, hallucination_loop, none)"
        );
        if (!humanLabel) return;
      }
      const res = await submitFeedback({
        task_id: result.task_id,
        aria_correct: correct,
        human_label: humanLabel,
        notes: null,
      });
      setFeedbackMsg(res.message);
      onDiagnosed?.();
    } catch (e) {
      setFeedbackMsg(`Feedback error: ${e.message}`);
    }
  };

  const cls = result?.failure_class || "none";

  return (
    <div className="card panel-card">
      <h3>Submit an agent trace for diagnosis</h3>

      <label>Task description</label>
      <textarea
        rows={2}
        placeholder="What was the agent asked to do?"
        value={task}
        onChange={(e) => setTask(e.target.value)}
      />

      <label>Tool calls (JSON array)</label>
      <textarea
        rows={7}
        className="mono"
        value={toolsJson}
        onChange={(e) => setToolsJson(e.target.value)}
        spellCheck={false}
      />

      <label>Final agent output</label>
      <textarea
        rows={3}
        placeholder="The agent's final answer or output..."
        value={finalOutput}
        onChange={(e) => setFinalOutput(e.target.value)}
      />

      <button className="btn" onClick={run} disabled={busy || !task.trim()}>
        {busy ? "Diagnosing…" : "Diagnose"}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="result-box">
          <div className="result-head">
            <span
              className="class-badge"
              style={{ background: classColor(cls), fontSize: 13 }}
            >
              {classLabel(cls)}
            </span>
            <span className="mono" style={{ color: "#9aa4b2" }}>
              confidence {result.confidence?.toFixed(2)} · id {result.task_id}
            </span>
          </div>
          <div className="result-body">
            <div className="kv">
              <div className="k">Requirement satisfaction</div>
              <div className="v">
                {Math.round((result.requirement_satisfaction || 0) * 100)}%
                <div className="meter">
                  <div
                    style={{
                      width: `${(result.requirement_satisfaction || 0) * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            {result.requirements?.length > 0 && (
              <div className="kv">
                <div className="k">Requirements checklist</div>
                {result.requirements.map((req, i) => {
                  const ok = result.requirements_satisfied?.[i];
                  return (
                    <div className="req-item" key={i}>
                      <span className={ok ? "req-ok" : "req-miss"}>
                        {ok ? "✓" : "✗"}
                      </span>
                      <span>{req}</span>
                    </div>
                  );
                })}
              </div>
            )}

            {result.evidence?.length > 0 && (
              <div className="kv">
                <div className="k">Evidence</div>
                {result.evidence.map((e, i) => (
                  <div className="v" key={i}>
                    · {e}
                  </div>
                ))}
              </div>
            )}

            <div className="kv">
              <div className="k">Reasoning</div>
              <div className="v">{result.reasoning}</div>
            </div>

            <div className="kv">
              <div className="k">Suggested action</div>
              <div className="v">{result.suggested_action}</div>
            </div>

            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn secondary" onClick={() => sendFeedback(true)}>
                ✓ Diagnosis correct
              </button>
              <button className="btn secondary" onClick={() => sendFeedback(false)}>
                ✗ Wrong — correct it
              </button>
            </div>
            {feedbackMsg && (
              <div className="v" style={{ color: "#5e9468" }}>
                {feedbackMsg}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
