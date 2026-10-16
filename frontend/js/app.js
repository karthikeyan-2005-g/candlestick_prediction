"use strict";

/* =========================================================
   AI CANDLESTICK PREDICTION SYSTEM
   Frontend JavaScript
   ========================================================= */

const predictionForm = document.querySelector("#prediction-form");
const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const predictButton = document.querySelector("#predict-button");
const formMessage = document.querySelector("#form-message");

const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const API_BASE = window.location.origin.includes(":5000") ? "" : "http://127.0.0.1:5000";

function getPageUrl(path) {
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  if (window.location.pathname.includes("/frontend/")) {
    return `/frontend/${cleanPath}`;
  }
  return `/${cleanPath}`;
}

let previewUrl = null;

/* =========================================================
   1. MESSAGE HANDLER
   ========================================================= */

function setMessage(element, message, type = "") {
  if (!element) return;

  element.textContent = message;
  element.className = `form-message ${type}`.trim();
}

/* =========================================================
   2. SAFE JSON RESPONSE HANDLER
   Fixes: Unexpected end of JSON input
   ========================================================= */

async function readJsonResponse(response) {
    const responseText = await response.text();

    let payload = null;

    if (responseText.trim() !== "") {
        try {
            payload = JSON.parse(responseText);
        } catch (error) {
            throw new Error(
                `Invalid JSON response (HTTP ${response.status}). ` +
                responseText.slice(0, 200)
            );
        }
    }

    if (!response.ok) {
        const message =
            payload?.message ||
            payload?.error ||
            `HTTP ${response.status}: ${response.statusText}`;

        throw new Error(message);
    }

    if (payload === null) {
        throw new Error(
            `Server returned an empty response (HTTP ${response.status}).`
        );
    }

    return payload;
}

/* =========================================================
   3. API ERROR HANDLER
   ========================================================= */

function getApiError(payload, fallbackMessage) {
  return (
    payload?.error?.message ||
    payload?.message ||
    fallbackMessage
  );
}

/* =========================================================
   4. BADGE CREATION
   ========================================================= */

function makeBadge(value) {
  const badge = document.createElement("span");

  const text = String(value || "—");
  const tone = text.toLowerCase();

  badge.className = "badge";

  if (tone === "up" || tone === "down") {
    badge.classList.add("dataset-class");
  } else if (
    ["bullish", "bearish", "uncertain"].includes(tone)
  ) {
    badge.classList.add(tone);
  }

  badge.textContent = text;

  return badge;
}

/* =========================================================
   5. DATE FORMATTER
   ========================================================= */

function formatDate(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.valueOf())) {
    return "—";
  }

  return date.toLocaleString();
}

/* =========================================================
   6. RECENT PREDICTIONS TABLE
   ========================================================= */

function makeRecentTable(records) {
  const host = document.querySelector("#recent-history");

  if (!host) return;

  host.replaceChildren();

  if (!Array.isArray(records) || records.length === 0) {
    const empty = document.createElement("p");

    empty.className = "muted";
    empty.textContent =
      "No predictions have been recorded yet.";

    host.append(empty);
    return;
  }

  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");

  [
    "Time",
    "Symbol",
    "Timeframe",
    "Prediction",
    "Score",
    "Actual"
  ].forEach((label) => {
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

    const symbol = document.createElement("td");

    symbol.textContent = record.symbol || "—";

    row.append(symbol);

    const timeframe = document.createElement("td");

    timeframe.textContent = record.timeframe || "—";

    row.append(timeframe);

    const prediction = document.createElement("td");

    prediction.append(makeBadge(record.prediction));

    row.append(prediction);

    const score = document.createElement("td");

    const confidence = Number(record.confidence);

    score.textContent = Number.isFinite(confidence)
      ? `${(confidence * 100).toFixed(1)}%`
      : "—";

    row.append(score);

    const actual = document.createElement("td");

    actual.textContent =
      record.actual_direction || "Pending";

    row.append(actual);

    row.tabIndex = 0;
    row.setAttribute("role", "link");
    row.setAttribute(
      "aria-label",
      `View prediction details for ${record.symbol || "unknown symbol"}`
    );

    const openDetails = () => {
      if (record.id === undefined || record.id === null) {
        return;
      }

      window.location.href =
        getPageUrl(`prediction.html?id=${encodeURIComponent(record.id)}`);
    };

    row.addEventListener("click", openDetails);

    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        openDetails();
      }
    });

    body.append(row);
  });

  table.append(head, body);

  host.append(table);
}

/* =========================================================
   7. REFRESH API, DATABASE AND MODEL STATUS
   ========================================================= */

async function refreshStatus() {
  const apiElement = document.querySelector("#api-status");
  const dbElement = document.querySelector("#db-status");
  const modelElement = document.querySelector("#model-status");

  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      method: "GET",
      headers: {
        Accept: "application/json"
      },
      cache: "no-store"
    });

    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(
        getApiError(payload, `Health API failed: HTTP ${response.status}`)
      );
    }

    if (apiElement) {
      apiElement.textContent = "API: ok";
      apiElement.className = "status-pill status-ok";
    }

    if (dbElement) {
      const database = String(
        payload.database || "unavailable"
      ).toLowerCase();

      dbElement.textContent = `Database: ${database}`;

      dbElement.className =
        `status-pill ${
          ["ok", "available", "connected"].includes(database)
            ? "status-ok"
            : "status-bad"
        }`;
    }

    if (modelElement) {
      const model = String(
        payload.model || "unavailable"
      ).toLowerCase();

      modelElement.textContent = `Model: ${model}`;

      modelElement.className =
        `status-pill ${
          ["ok", "available", "loaded"].includes(model)
            ? "status-ok"
            : "status-bad"
        }`;
    }

  } catch (error) {
    console.error("Health API error:", error);

    if (apiElement) {
      apiElement.textContent = "API: unavailable";
      apiElement.className = "status-pill status-bad";
    }

    if (dbElement) {
      dbElement.textContent = "Database: unavailable";
      dbElement.className = "status-pill status-bad";
    }

    if (modelElement) {
      modelElement.textContent = "Model: unavailable";
      modelElement.className = "status-pill status-bad";
    }
  }
}

/* =========================================================
   8. REFRESH RECENT PREDICTION HISTORY
   ========================================================= */

async function refreshRecent() {
  const host = document.querySelector("#recent-history");

  if (!host) return;

  try {
    const response = await fetch(
      `${API_BASE}/api/history?per_page=5`,
      {
        method: "GET",
        headers: {
          Accept: "application/json"
        },
        cache: "no-store"
      }
    );

    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(
        getApiError(
          payload,
          `Could not load history (HTTP ${response.status}).`
        )
      );
    }

    if (!Array.isArray(payload.data)) {
      console.error(
        "Unexpected history API response:",
        payload
      );

      throw new Error(
        "History API response is missing the data array. Check history_routes.py."
      );
    }

    host.className = "table-wrap";

    makeRecentTable(payload.data);

  } catch (error) {
    console.error("History API error:", error);

    host.replaceChildren();

    const message = document.createElement("p");

    message.className = "form-message error";
    message.textContent = error.message;

    host.append(message);
  }
}

/* =========================================================
   9. DISPLAY PREDICTION RESULT
   ========================================================= */

function showPrediction(result) {
  const emptyResult = document.querySelector("#empty-result");
  const predictionResult = document.querySelector("#prediction-result");

  if (!emptyResult || !predictionResult) {
    throw new Error(
      "Prediction result elements were not found in prediction.html."
    );
  }

  const confidence = Number(result.confidence);

  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error(
      "The server returned an invalid confidence score."
    );
  }

  emptyResult.classList.add("hidden");
  predictionResult.classList.remove("hidden");

  const direction = document.querySelector("#result-direction");

  if (direction) {
    direction.className = "direction-card dataset-class";
  }

  const resultLabel = document.querySelector("#result-label");

  if (resultLabel) {
    resultLabel.textContent = result.prediction || "Unknown";
  }

  const resultConfidence =
    document.querySelector("#result-confidence");

  if (resultConfidence) {
    resultConfidence.textContent =
      `${(confidence * 100).toFixed(1)}%`;
  }

  const confidenceBar =
    document.querySelector("#confidence-bar");

  if (confidenceBar) {
    confidenceBar.style.width =
      `${Math.max(0, Math.min(100, confidence * 100))}%`;
  }

  const resultSymbol = document.querySelector("#result-symbol");

  if (resultSymbol) {
    resultSymbol.textContent = result.symbol || "—";
  }

  const resultTimeframe =
    document.querySelector("#result-timeframe");

  if (resultTimeframe) {
    resultTimeframe.textContent = result.timeframe || "—";
  }

  const resultVersion =
    document.querySelector("#result-version");

  if (resultVersion) {
    resultVersion.textContent = result.model_version || "—";
  }

  const resultTime = document.querySelector("#result-time");

  if (resultTime) {
    resultTime.textContent = formatDate(result.created_at);
  }

  const detailLink = document.querySelector("#detail-link");

  if (detailLink && result.prediction_id != null) {
    detailLink.href =
      getPageUrl(`prediction.html?id=${encodeURIComponent(result.prediction_id)}`);
  }
}

/* =========================================================
   10. IMAGE SELECTION AND PREVIEW
   ========================================================= */

if (imageInput && predictButton && imagePreview) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files?.[0];

    predictButton.disabled = !file;

    const buttonLabel = document.querySelector("#button-label");

    if (buttonLabel) {
      buttonLabel.textContent = file
        ? "Analyze chart"
        : "Select an image to continue";
    }

    setMessage(formMessage, "");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }

    imagePreview.classList.add("hidden");

    if (!file) {
      const fileName = document.querySelector("#file-name");

      if (fileName) {
        fileName.textContent = "No image selected";
      }

      return;
    }

    const fileName = document.querySelector("#file-name");

    if (fileName) {
      fileName.textContent = file.name;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      predictButton.disabled = true;

      setMessage(
        formMessage,
        "The selected image exceeds the 5 MB upload limit.",
        "error"
      );

      return;
    }

    if (file.size === 0) {
      predictButton.disabled = true;

      setMessage(
        formMessage,
        "The selected image is empty. Choose another image.",
        "error"
      );

      return;
    }

    if (file.type && !file.type.startsWith("image/")) {
      predictButton.disabled = true;

      setMessage(
        formMessage,
        "Please select a valid image file.",
        "error"
      );

      return;
    }

    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;

    imagePreview.classList.remove("hidden");
  });
}

/* =========================================================
   11. SUBMIT IMAGE FOR PREDICTION
   ========================================================= */

if (predictionForm && imageInput && predictButton) {
  predictionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files?.[0];

    if (!file) {
      setMessage(
        formMessage,
        "Please select a candlestick chart image first.",
        "error"
      );

      return;
    }

    if (file.size === 0 || file.size > MAX_UPLOAD_BYTES) {
      setMessage(
        formMessage,
        "Choose a non-empty image smaller than 5 MB.",
        "error"
      );

      return;
    }

    const data = new FormData();

    data.append("image", file);

    const symbolInput = document.querySelector("#symbol-input");
    const timeframeInput = document.querySelector("#timeframe-input");

    data.append(
      "symbol",
      symbolInput ? symbolInput.value : ""
    );

    data.append(
      "timeframe",
      timeframeInput ? timeframeInput.value : ""
    );

    predictButton.disabled = true;

    const buttonLabel = document.querySelector("#button-label");
    const buttonSpinner = document.querySelector("#button-spinner");

    if (buttonLabel) {
      buttonLabel.textContent = "Analyzing…";
    }

    if (buttonSpinner) {
      buttonSpinner.classList.remove("hidden");
    }

    setMessage(
      formMessage,
      "Validating image and running model inference…"
    );

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: {
          Accept: "application/json"
        },
        body: data
      });

      const payload = await readJsonResponse(response);

      if (!response.ok) {
        throw new Error(
          getApiError(
            payload,
            `Prediction request failed (HTTP ${response.status}).`
          )
        );
      }

      if (payload.success === false) {
        throw new Error(
          getApiError(payload, "Prediction failed.")
        );
      }

      if (
        !payload.prediction ||
        typeof payload.confidence !== "number"
      ) {
        console.error(
          "Unexpected prediction API response:",
          payload
        );

        throw new Error(
          "Prediction response is missing required fields. Check prediction_routes.py."
        );
      }

      showPrediction(payload);

      try {
        if (imagePreview.src) {
          sessionStorage.setItem(
            "lastPredictionImage",
            imagePreview.src
          );
        }

        if (payload.prediction_id != null) {
          sessionStorage.setItem(
            "lastPredictionId",
            String(payload.prediction_id)
          );
        }
      } catch (storageError) {
        console.warn(
          "Could not save prediction to session storage:",
          storageError
        );
      }

      setMessage(
        formMessage,
        payload.message ||
          "Prediction completed successfully.",
        "success"
      );

      await refreshRecent();
      await refreshStatus();

    } catch (error) {
      console.error("Prediction request error:", error);

      setMessage(
        formMessage,
        error.message ||
          "An unexpected error occurred while processing the image.",
        "error"
      );

    } finally {
      predictButton.disabled = !imageInput.files?.[0];

      if (buttonLabel) {
        buttonLabel.textContent = "Analyze chart";
      }

      if (buttonSpinner) {
        buttonSpinner.classList.add("hidden");
      }
    }
  });
}

/* =========================================================
   12. INITIALIZE DASHBOARD
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {
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
  refreshStatus();
  refreshRecent();
});

/* =========================================================
   END OF APP.JS
   ========================================================= */