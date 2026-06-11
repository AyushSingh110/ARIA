const BASE = "/api";

export async function fetchDashboard() {
  const res = await fetch(`${BASE}/dashboard`);
  if (!res.ok) throw new Error(`Dashboard fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("API offline");
  return res.json();
}

export async function diagnoseTrace(payload) {
  const res = await fetch(`${BASE}/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Diagnose failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function submitFeedback(payload) {
  const res = await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Feedback failed (${res.status}): ${detail}`);
  }
  return res.json();
}
