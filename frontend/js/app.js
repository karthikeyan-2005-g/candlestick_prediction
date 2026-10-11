"use strict";

const predictionForm = document.querySelector("#prediction-form");
const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const predictButton = document.querySelector("#predict-button");
const formMessage = document.querySelector("#form-message");
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
let previewUrl = null;

function setMessage(element, message, type = "") {
  element.textContent = message;
  element.className = `form-message ${type}`.trim();
}

function makeBadge(value) {
  const badge = document.createElement("span");
  const tone = String(value || "").toLowerCase();
  const color = tone === "up" || tone === "down" ? "dataset-class" : tone;
  badge.className = `badge ${["bullish", "bearish", "uncertain"].includes(color) ? color : ""}`;
  badge.textContent = value || "—";
  return badge;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : date.toLocaleString();
}

function makeRecentTable(records) {
  const host = document.querySelector("#recent-history");
  host.replaceChildren();
  if (!records.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No predictions have been recorded yet.";
    host.append(empty);
    return;
  }
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Time", "Symbol", "Timeframe", "Prediction", "Score", "Actual"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.append(th);
  });
  head.append(headRow);
  const body = document.createElement("tbody");
  records.forEach((record) => {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    time.textContent = formatDate(record.created_at);
    row.append(time);
    [record.symbol || "—", record.timeframe || "—"].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      row.append(td);
    });
    const prediction = document.createElement("td");
    prediction.append(makeBadge(record.prediction));
    row.append(prediction);
    const score = document.createElement("td");
    score.textContent = `${(record.confidence * 100).toFixed(1)}%`;
    row.append(score);
    const actual = document.createElement("td");
    actual.textContent = record.actual_direction || "Pending";
    row.append(actual);
    row.tabIndex = 0;
    row.setAttribute("role", "link");
    row.addEventListener("click", () => { window.location.href = `/prediction.html?id=${encodeURIComponent(record.id)}`; });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter") window.location.href = `/prediction.html?id=${encodeURIComponent(record.id)}`;
    });
    body.append(row);
  });
  table.append(head, body);
  host.append(table);
}

async function refreshStatus() {
  const labels = { "api-status": "API", "db-status": "Database", "model-status": "Model" };
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    Object.entries(labels).forEach(([id, label]) => {
      const element = document.getElementById(id);
      const value = id === "api-status" ? (response.ok ? "ok" : "bad") : payload[id === "db-status" ? "database" : "model"];
      element.textContent = `${label}: ${value || "unavailable"}`;
      element.className = `status-pill ${value === "ok" || value === "available" ? "status-ok" : "status-bad"}`;
    });
  } catch {
    document.querySelector("#api-status").textContent = "API: unavailable";
    document.querySelector("#api-status").className = "status-pill status-bad";
  }
}

async function refreshRecent() {
  const host = document.querySelector("#recent-history");
  try {
    const response = await fetch("/api/history?per_page=5");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Could not load history.");
    makeRecentTable(payload.data);
  } catch (error) {
    host.textContent = error.message;
    host.className = "table-wrap form-message error";
  }
}

function showPrediction(result) {
  document.querySelector("#empty-result").classList.add("hidden");
  document.querySelector("#prediction-result").classList.remove("hidden");
  const direction = document.querySelector("#result-direction");
  direction.className = "direction-card dataset-class";
  document.querySelector("#result-label").textContent = result.prediction;
  document.querySelector("#result-confidence").textContent = `${(result.confidence * 100).toFixed(1)}%`;
  document.querySelector("#confidence-bar").style.width = `${Math.max(0, Math.min(100, result.confidence * 100))}%`;
  document.querySelector("#result-symbol").textContent = result.symbol || "—";
  document.querySelector("#result-timeframe").textContent = result.timeframe || "—";
  document.querySelector("#result-version").textContent = result.model_version;
  document.querySelector("#result-time").textContent = formatDate(result.created_at);
  document.querySelector("#detail-link").href = `/prediction.html?id=${encodeURIComponent(result.prediction_id)}`;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  predictButton.disabled = !file;
  document.querySelector("#button-label").textContent = file ? "Analyze chart" : "Select an image to continue";
  setMessage(formMessage, "");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.classList.add("hidden");
  if (!file) {
    document.querySelector("#file-name").textContent = "No image selected";
    return;
  }
  document.querySelector("#file-name").textContent = file.name;
  if (file.size > MAX_UPLOAD_BYTES) {
    predictButton.disabled = true;
    setMessage(formMessage, "The selected image exceeds the 5 MB upload limit.", "error");
    return;
  }
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.classList.remove("hidden");
});

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file || file.size > MAX_UPLOAD_BYTES) return;
  const data = new FormData();
  data.append("image", file);
  data.append("symbol", document.querySelector("#symbol-input").value);
  data.append("timeframe", document.querySelector("#timeframe-input").value);
  predictButton.disabled = true;
  document.querySelector("#button-label").textContent = "Analyzing…";
  document.querySelector("#button-spinner").classList.remove("hidden");
  setMessage(formMessage, "Validating image and running model inference…");
  try {
    const response = await fetch("/api/predict", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Prediction request failed.");
    showPrediction(payload);
    try {
      sessionStorage.setItem("lastPredictionImage", imagePreview.src);
      sessionStorage.setItem("lastPredictionId", String(payload.prediction_id));
    } catch { /* Storage may be disabled. */ }
    setMessage(formMessage, payload.message, "success");
    await refreshRecent();
    await refreshStatus();
  } catch (error) {
    setMessage(formMessage, error.message, "error");
  } finally {
    predictButton.disabled = !imageInput.files[0];
    document.querySelector("#button-label").textContent = "Analyze chart";
    document.querySelector("#button-spinner").classList.add("hidden");
  }
});

refreshStatus();
refreshRecent();
