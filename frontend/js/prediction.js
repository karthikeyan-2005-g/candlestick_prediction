"use strict";

const predictionId = new URLSearchParams(window.location.search).get("id");
const message = document.querySelector("#detail-message");
const detailSection = document.querySelector("#prediction-detail");
const outcomePanel = document.querySelector("#outcome-panel");
let predictionRecord = null;

function setMessage(text, error = false) {
  message.textContent = text;
  message.className = `form-message ${error ? "error" : ""}`.trim();
}

function displayDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : date.toLocaleString();
}

function render(record) {
  predictionRecord = record;
  document.querySelector("#detail-label").textContent = record.prediction;
  document.querySelector("#detail-direction").className = "direction-card dataset-class";
  document.querySelector("#detail-confidence").textContent = `${(record.confidence * 100).toFixed(1)}%`;
  document.querySelector("#detail-symbol").textContent = record.symbol || "—";
  document.querySelector("#detail-timeframe").textContent = record.timeframe || "—";
  document.querySelector("#detail-version").textContent = record.model_version;
  document.querySelector("#detail-created").textContent = displayDate(record.created_at);
  document.querySelector("#detail-hash").textContent = record.image_sha256;
  document.querySelector("#detail-actual").textContent = record.actual_direction || "Not recorded";
  document.querySelector("#detail-observed").textContent = displayDate(record.observed_at);
  detailSection.classList.remove("hidden");
  if (!record.actual_direction) outcomePanel.classList.remove("hidden");
  const previousImage = sessionStorage.getItem("lastPredictionImage");
  if (previousImage && record.id === Number(sessionStorage.getItem("lastPredictionId"))) {
    const image = document.querySelector("#detail-image");
    image.src = previousImage;
    image.classList.remove("hidden");
    document.querySelector("#image-unavailable").classList.add("hidden");
  }
}

async function loadPrediction() {
  if (!predictionId || !/^\d+$/.test(predictionId)) {
    setMessage("A valid prediction ID is required.", true);
    return;
  }
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(predictionId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Could not load prediction.");
    render(payload.data);
    setMessage("");
  } catch (error) {
    setMessage(error.message, true);
  }
}

document.querySelector("#outcome-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!predictionRecord) return;
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(predictionId)}/outcome`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_direction: document.querySelector("#actual-direction").value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Could not save outcome.");
    render(payload.data);
    setMessage("Actual outcome saved.", false);
    outcomePanel.classList.add("hidden");
  } catch (error) {
    setMessage(error.message, true);
    button.disabled = false;
  }
});

loadPrediction();
