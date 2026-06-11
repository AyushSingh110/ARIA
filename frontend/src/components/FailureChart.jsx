import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";
import { classColor, classLabel } from "../constants";

export default function FailureChart({ stats }) {
  const dist = stats?.class_distribution || {};
  const data = Object.entries(dist)
    .map(([cls, count]) => ({
      name: classLabel(cls),
      cls,
      value: count,
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="card chart-card">
      <h3>Failure class distribution</h3>
      {data.length === 0 ? (
        <div className="empty-note">No runs recorded yet.</div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={62}
              outerRadius={100}
              paddingAngle={2}
              strokeWidth={1.5}
              stroke="#ffffff"
            >
              {data.map((d) => (
                <Cell key={d.cls} fill={classColor(d.cls)} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value, name) => [`${value} runs`, name]}
              contentStyle={{
                borderRadius: 10,
                border: "1px solid #e6e9ee",
                fontSize: 13,
              }}
            />
            <Legend
              iconType="circle"
              iconSize={9}
              formatter={(v) => (
                <span style={{ color: "#6b7686", fontSize: 12.5 }}>{v}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
