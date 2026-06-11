// Failure class display config — light research palette
export const CLASS_COLORS = {
  none: "#c8e0cc",
  goal_misalignment: "#cdc0e8",
  tool_misuse: "#f2c6c2",
  context_overflow: "#f5e3b3",
  prompt_drift: "#b8cde8",
  hallucination_loop: "#f2d7b6",
  gap: "#d9dee5",
};

export const CLASS_LABELS = {
  none: "Clean run",
  goal_misalignment: "Goal misalignment",
  tool_misuse: "Tool misuse",
  context_overflow: "Context overflow",
  prompt_drift: "Prompt drift",
  hallucination_loop: "Hallucination loop",
  gap: "Outside taxonomy",
};

export const classColor = (cls) => CLASS_COLORS[cls] || "#d9dee5";
export const classLabel = (cls) => CLASS_LABELS[cls] || cls || "Clean run";
