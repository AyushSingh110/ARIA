import { classLabel } from "../constants";

export default function StatCards({ stats }) {
  const fmt = (v, suffix = "") =>
    v === null || v === undefined ? "—" : `${v}${suffix}`;

  const cards = [
    {
      label: "Total runs analyzed",
      value: fmt(stats?.total_runs),
      hint: "RealBench + API diagnoses",
    },
    {
      label: "Avg requirement satisfaction",
      value:
        stats?.avg_requirement_satisfaction != null
          ? `${Math.round(stats.avg_requirement_satisfaction * 100)}%`
          : "—",
      hint: "Critic v2 — requirements met per run",
    },
    {
      label: "Pass rate",
      value:
        stats?.pass_rate != null
          ? `${Math.round(stats.pass_rate * 100)}%`
          : "—",
      hint: "req_sat ≥ 0.75 and correctness ≥ 2",
    },
    {
      label: "Most common failure",
      value: stats?.most_common_failure
        ? classLabel(stats.most_common_failure)
        : "—",
      hint: "Excluding clean runs",
    },
    {
      label: "Human-labeled runs",
      value: fmt(stats?.labeled_runs),
      hint:
        stats?.human_agreement_rate != null
          ? `Agreement: ${Math.round(stats.human_agreement_rate * 100)}%`
          : "Submit feedback to track agreement",
    },
  ];

  return (
    <div className="stat-grid">
      {cards.map((c) => (
        <div className="card stat-card" key={c.label}>
          <div className="label">{c.label}</div>
          <div className="value">{c.value}</div>
          <div className="hint">{c.hint}</div>
        </div>
      ))}
    </div>
  );
}
