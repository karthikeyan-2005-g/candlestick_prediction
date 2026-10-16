"use strict";

const API_BASE = window.location.origin.includes(":5000") ? "" : "http://127.0.0.1:5000";

function getPageUrl(path) {
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  if (window.location.pathname.includes("/frontend/")) {
    return `/frontend/${cleanPath}`;
  }
  return `/${cleanPath}`;
}

const tableHost = document.querySelector("#history-table");
const message = document.querySelector("#history-message");
const previousButton = document.querySelector("#previous-page");
const nextButton = document.querySelector("#next-page");
const pageLabel = document.querySelector("#page-label");
let page = 1;
let pages = 1;
let symbolFilter = "";

function setMessage(text, error = false) {
  message.textContent = text;
  message.className = `form-message ${error ? "error" : ""}`.trim();
}

function cell(row, value) {
  const td = document.createElement("td");
  td.textContent = value ?? "—";
  row.append(td);
  return td;
}

function renderRows(records) {
  tableHost.replaceChildren();
  if (!records.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No predictions match this filter.";
    tableHost.append(empty);
    return;
  }
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Prediction time", "Symbol", "Timeframe", "Prediction", "Score", "Actual outcome", "Result", ""].forEach((name) => {
    const th = document.createElement("th");
    th.textContent = name;
    headerRow.append(th);
  });
  head.append(headerRow);
  const body = document.createElement("tbody");
  records.forEach((record) => {
    const row = document.createElement("tr");
    const date = record.created_at ? new Date(record.created_at).toLocaleString() : "—";
    cell(row, date);
    cell(row, record.symbol || "—");
    cell(row, record.timeframe || "—");
    const prediction = document.createElement("td");
    const badge = document.createElement("span");
    const tone = ["up", "down"].includes(record.prediction.toLowerCase())
      ? "dataset-class"
      : record.prediction.toLowerCase();
    badge.className = `badge ${tone}`;
    badge.textContent = record.prediction;
    prediction.append(badge);
    row.append(prediction);
    cell(row, `${(record.confidence * 100).toFixed(1)}%`);
    cell(row, record.actual_direction || "Pending");
    const result = document.createElement("td");
    if (record.actual_direction) {
      if (["Down", "Up"].includes(record.prediction)) {
        result.textContent = "Not comparable (label meaning unverified)";
      } else {
        const correct = record.prediction === record.actual_direction;
        result.className = correct ? "match" : "miss";
        result.textContent = correct ? "Match" : "Mismatch";
      }
    } else {
      result.textContent = "—";
    }
    row.append(result);
    const action = document.createElement("td");
    const link = document.createElement("a");
    link.className = "text-link";
    link.href = getPageUrl(`prediction.html?id=${encodeURIComponent(record.id)}`);
    link.textContent = "Details";
    action.append(link);
    row.append(action);
    body.append(row);
  });
  table.append(head, body);
  tableHost.append(table);
}

async function loadHistory() {
  setMessage("Loading prediction history…");
  const params = new URLSearchParams({ page: String(page), per_page: "20" });
  if (symbolFilter) params.set("symbol", symbolFilter);
  try {
    const response = await fetch(`${API_BASE}/api/history?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Could not load history.");
    renderRows(payload.data);
    pages = Math.max(1, payload.pagination.pages);
    pageLabel.textContent = `Page ${page} of ${pages} · ${payload.pagination.total} records`;
    previousButton.disabled = page <= 1;
    nextButton.disabled = page >= pages;
    setMessage("");
  } catch (error) {
    tableHost.replaceChildren();
    setMessage(error.message, true);
  }
}

document.querySelector("#filter-form").addEventListener("submit", (event) => {
  event.preventDefault();
  symbolFilter = document.querySelector("#symbol-filter").value.trim().toUpperCase();
  page = 1;
  loadHistory();
});
document.querySelector("#clear-filter").addEventListener("click", () => {
  document.querySelector("#symbol-filter").value = "";
  symbolFilter = "";
  page = 1;
  loadHistory();
});
previousButton.addEventListener("click", () => { if (page > 1) { page -= 1; loadHistory(); } });
nextButton.addEventListener("click", () => { if (page < pages) { page += 1; loadHistory(); } });

if (window.location.pathname.includes("/frontend/")) {
  document.querySelectorAll("nav a, a.brand, footer a").forEach((link) => {
    const href = link.getAttribute("href");
    if (href === "/") {
      link.href = "/frontend/index.html";
    } else if (href && !href.startsWith("http") && !href.startsWith("/frontend/")) {
      link.href = `/frontend/${href.replace(/^\//, "")}`;
    }
  });
}

loadHistory();
