import { classColor, classLabel } from "../constants";

export default function RecentFailures({ stats }) {
  const rows = stats?.recent_failures || [];

  return (
    <div className="card table-card">
      <h3>Recent detected failures</h3>
      {rows.length === 0 ? (
        <div className="empty-note">No failures detected yet.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Class</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {rows
              .slice()
              .reverse()
              .map((r, i) => (
                <tr key={`${r.task_id}-${i}`}>
                  <td>
                    <span className="mono" style={{ color: "#9aa4b2" }}>
                      {r.task_id}
                    </span>{" "}
                    {r.task}
                  </td>
                  <td>
                    <span
                      className="class-badge"
                      style={{ background: classColor(r.failure_class) }}
                    >
                      {classLabel(r.failure_class)}
                    </span>
                  </td>
                  <td className="mono">
                    {r.confidence != null ? r.confidence.toFixed(2) : "—"}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
