/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - Scan History page
   Filters by crop, by status, and clears history.
   ============================================================ */

let historyAll = [];
let filterCrop = "all";
let filterStatus = "all";

async function initHistory() {
  if (!requireAuth()) return;
  initNav("history");
  historyAll = HistoryStore.all();

  // populate crop filter
  const cropSelect = document.getElementById("filter-crop");
  const crops = [...new Set(historyAll.map(r => r.crop))].sort();
  cropSelect.innerHTML = `<option value="all">${Lang.key("all_crops")}</option>` +
    crops.map(c => `<option value="${c}">${c}</option>`).join("");

  document.getElementById("filter-status").innerHTML = `
    <option value="all">${Lang.key("status_all")}</option>
    <option value="Healthy">${Lang.key("healthy")}</option>
    <option value="Diseased">${Lang.key("diseased")}</option>
    <option value="Analysis Not Available">${Lang.key("analysis_not_available")}</option>
    <option value="Unknown">${Lang.key("unknown")}</option>`;

  cropSelect.addEventListener("change", e => { filterCrop = e.target.value; renderHistory(); });
  document.getElementById("filter-status").addEventListener("change", e => {
    filterStatus = e.target.value; renderHistory();
  });

  document.getElementById("btn-clear").addEventListener("click", () => {
    if (confirm(Lang.key("clear_confirm"))) {
      HistoryStore.clear();
      historyAll = [];
      renderHistory();
    }
  });

  renderHistory();
}

function renderHistory() {
  const listEl = document.getElementById("history-list");
  const container = document.getElementById("history-page");

  let filtered = historyAll.filter(r => {
    const okCrop = filterCrop === "all" || r.crop === filterCrop;
    const okStatus = filterStatus === "all" || r.status === filterStatus;
    return okCrop && okStatus;
  });

  if (!filtered.length) {
    listEl.innerHTML = `<div class="card center"><p class="muted">${Lang.key("history_empty")}</p></div>`;
    return;
  }

  listEl.innerHTML = filtered.map(r => {
    const st = r.status || "Unknown";
    const pillClass = st.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const statusKey = st === "Healthy" ? "healthy"
      : st === "Diseased" ? "diseased"
      : st === "Not a Plant" ? "not_a_plant"
      : st === "Low Plant Confidence" ? "low_confidence"
      : st === "Analysis Not Available" ? "analysis_not_available"
      : st === "Unsupported for Disease Detection" ? "unsupported_crop"
      : st === "Unknown Plant" ? "unknown_plant"
      : "unknown";
    const label = Lang.key(statusKey);
    const disease = "";
    const sev = r.severity
      ? `<span class="pill ${r.severity.toLowerCase()}">${Lang.key("severity_label")}: ${Lang.key("severity_" + r.severity.toLowerCase())}</span>` +
        (r.affected_area_pct != null
          ? ` <span class="muted">${Lang.key("affected_leaf_area")}: ${Number(r.affected_area_pct).toFixed(1)}%</span>` : "")
      : "";
    const spray = r.spray_decision
      ? `<div class="muted" style="font-size:.78rem;">🚿 ${Lang.key("spray_decision")}: ${r.spray_decision.replace(/_/g, " ")}</div>`
      : "";
    const thumb = r.thumb ? `<img class="thumb" src="${r.thumb}" alt="">`
      : `<div class="thumb" style="display:flex;align-items:center;justify-content:center;">🌿</div>`;
    const confVal = (r.plant_validation_confidence != null)
      ? r.plant_validation_confidence : r.confidence;
    return `
      <div class="card" style="margin:10px 16px;padding:12px;">
        <div class="history-item">
          ${thumb}
          <div class="meta">
            <div class="crop">${r.crop}</div>
            <div class="time">${r.timeLabel}</div>
            ${disease}
            ${sev}
            ${spray}
            <span class="pill ${pillClass}">${label}</span>
          </div>
          <div class="conf">${Number(confVal || 0).toFixed(1)}%</div>
        </div>
      </div>`;
  }).join("") ;
}