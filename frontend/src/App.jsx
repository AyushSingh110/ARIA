import { useEffect, useState } from "react";
import { fetchDashboard, fetchHealth } from "./api";
import StatCards from "./components/StatCards.jsx";
import FailureChart from "./components/FailureChart.jsx";
import RecentFailures from "./components/RecentFailures.jsx";
import DiagnosePanel from "./components/DiagnosePanel.jsx";

export default function App() {
  const [stats, setStats] = useState(null);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState(null);

  const refresh = async () => {
    try {
      await fetchHealth();
      setOnline(true);
      const data = await fetchDashboard();
      setStats(data);
      setError(null);
    } catch (e) {
      setOnline(false);
      setError(
        "ARIA API is not reachable. Start it with: uvicorn api.main:app --port 8000"
      );
    }
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>ARIA</h1>
          <span className="sub">Agent Failure Diagnostics</span>
        </div>
        <div className="status-pill">
          <span className={`status-dot ${online ? "online" : "offline"}`} />
          {online ? "Runtime API connected" : "API offline"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="section-title">Overview</div>
      <StatCards stats={stats} />

      <div className="section-title">Failure Analysis</div>
      <div className="two-col">
        <FailureChart stats={stats} />
        <RecentFailures stats={stats} />
      </div>

      <div className="section-title">Diagnose a Trace</div>
      <DiagnosePanel onDiagnosed={refresh} />

      <div className="footer-note">
        ARIA — Autonomous Reflective Intelligence Architecture · Observer →
        Critic v2 → Diagnostician pipeline
      </div>
    </div>
  );
}
