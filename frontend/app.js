const queryInput = document.querySelector("#query");
const classifyButton = document.querySelector("#classify");
const resultSection = document.querySelector("#result");
const errorBox = document.querySelector("#error");

errorBox.style.whiteSpace = "pre-wrap";

/*
 * Live Server runs on 5500/5501.
 * FastAPI runs on 8000.
 */
const API_BASE = (() => {
  if (window.API_BASE) {
    return String(window.API_BASE).replace(/\/$/, "");
  }

  const { protocol, hostname, port } = window.location;

  if (port === "5500" || port === "5501") {
    return `${protocol}//${hostname}:8000`;
  }

  return window.location.origin;
})();

console.info("[classifier] Page origin:", window.location.origin);
console.info("[classifier] API base:", API_BASE);

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
  });
});

function render(result) {
  document.querySelector("#intent").textContent =
    result.intent || "Unknown";

  document.querySelector("#confidence").textContent =
    typeof result.confidence === "number"
      ? `${(result.confidence * 100).toFixed(1)}%`
      : "N/A";

  document.querySelector("#source").textContent =
    result.route_source || "Unknown";

  document.querySelector("#reason").textContent =
    result.reason || "No reason returned.";

  const clarification = document.querySelector("#clarification");

  clarification.hidden = !result.clarification_question;
  clarification.textContent = result.clarification_question || "";

  const executionPath = Array.isArray(result.execution_path)
    ? result.execution_path
    : [];

  document.querySelector("#path").innerHTML = executionPath
    .map((node) => `<span>${node}</span>`)
    .join("<span>→</span>");

  const matches = Array.isArray(result.top_matches)
    ? result.top_matches
    : [];

  document.querySelector("#matches").innerHTML = matches.length
    ? matches
        .map(
          (match) => `
            <div class="match">
              <strong>${match.intent}</strong>
              <span>${(match.score * 100).toFixed(1)}%</span>
              <span>${match.utterance}</span>
            </div>
          `
        )
        .join("")
    : "<p>No semantic matches were available.</p>";

  const logs = Array.isArray(result.logs) ? result.logs : [];

  document.querySelector("#logs").textContent =
    logs.length > 0 ? logs.join("\n") : "No logs returned.";

  resultSection.hidden = false;
}

async function classifyQuery(event) {
  // Prevent accidental form submission.
  event?.preventDefault();

  const query = queryInput.value.trim();

  if (!query) {
    errorBox.textContent = "Please enter a query.";
    errorBox.hidden = false;
    return;
  }

  classifyButton.disabled = true;
  classifyButton.textContent = "Classifying…";
  errorBox.hidden = true;
  errorBox.textContent = "";

  const endpoint = `${API_BASE}/api/classify`;

  console.info("[classifier] Sending request:", {
    method: "POST",
    endpoint,
    body: { query }
  });

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    });

    const responseText = await response.text();

    console.info("[classifier] HTTP response:", {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
      body: responseText
    });

    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText}\n` +
        `URL: ${response.url}\n` +
        `Response: ${responseText || "<empty response>"}`
      );
    }

    let result;

    try {
      result = JSON.parse(responseText);
    } catch (parseError) {
      throw new Error(
        `Backend returned invalid JSON.\n` +
        `URL: ${response.url}\n` +
        `Response: ${responseText}`
      );
    }

    console.info("[classifier] Classification result:", result);

    render(result);
  } catch (error) {
    console.error("[classifier] Request failed:", error);

    errorBox.textContent =
      `Request failed:\n${error.message || String(error)}\n\n` +
      `Expected backend URL: ${endpoint}`;

    errorBox.hidden = false;
  } finally {
    classifyButton.disabled = false;
    classifyButton.textContent = "Classify query";
  }
}

classifyButton.addEventListener("click", classifyQuery);

// Optional: Ctrl + Enter submits the query.
queryInput.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    classifyQuery(event);
  }
});