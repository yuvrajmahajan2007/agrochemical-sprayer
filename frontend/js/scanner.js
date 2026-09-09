/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - Scanner page
   - rear camera capture (capture="environment")
   - gallery upload
   - preview + Analyze Plant button
   - REAL AI prediction via Flask API
   - result rendering (Plant Health Analysis: status / affected
     area % / severity / spray decision support / real overlay)
   - saves scan with thumbnail to history
   ============================================================ */

let currentBlob = null;

async function initScanner() {
  if (!requireAuth()) return;
  initNav("scan");

  const takeBtn = document.getElementById("btn-take");
  const uploadBtn = document.getElementById("btn-upload");
  const fileInput = document.getElementById("camera-input");
  const galleryInput = document.getElementById("gallery-input");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview-img");
  const analyzeBtn = document.getElementById("btn-analyze");
  const retakeBtn = document.getElementById("btn-retake");

  takeBtn.addEventListener("click", () => {
    if (fileInput) fileInput.click();
  });

  uploadBtn.addEventListener("click", () => {
    if (galleryInput) galleryInput.click();
  });

  const onFile = (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    currentBlob = f;
    const url = URL.createObjectURL(f);
    previewImg.src = url;
    previewWrap.style.display = "block";
    analyzeBtn.classList.remove("btn-disabled");
  };

  if (fileInput) fileInput.addEventListener("change", onFile);
  if (galleryInput) galleryInput.addEventListener("change", onFile);

  retakeBtn.addEventListener("click", () => {
    currentBlob = null;
    previewWrap.style.display = "none";
    previewImg.src = "";
    if (fileInput) fileInput.value = "";
    if (galleryInput) galleryInput.value = "";
    analyzeBtn.classList.add("btn-disabled");
    document.getElementById("result-area").innerHTML = "";
  });

  analyzeBtn.addEventListener("click", async () => {
    if (!currentBlob) {
      showToast(Lang.key("select_image_first"));
      return;
    }
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = Lang.key("analyzing");

    document.getElementById("result-area").innerHTML =
      `<div class="center"><div class="spinner" style="border-top-color:var(--primary); width:38px; height:38px; margin:24px auto;"></div>
       <p class="muted">${Lang.key("analyzing")}</p></div>`;

    try {
      const data = await Api.predict(currentBlob);
      renderResult(data);
    } catch (err) {
      const msg = err && err.isNetwork
        ? Lang.key("cannot_connect")
        : (err && err.message ? err.message : Lang.key("network_error"));
      document.getElementById("result-area").innerHTML =
        `<div class="card"><p class="muted" style="white-space:pre-wrap">${msg}</p></div>`;
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = Lang.key("analyze_plant");
    }
  });
}

/* ---------- Render the AI result ---------- */
function renderResult(data) {
  const area = document.getElementById("result-area");
  if (!area) return;

  const status = (data.status || "Analysis Not Available");
  const now = new Date();
  const timeStr = now.toLocaleString();

  const vconf = data.plant_validation_confidence;
  const pct = data.affected_area_pct;
  const sev = data.severity;
  const thr = data.severity_thresholds || { low_max: 10, med_max: 40 };
  const decision = data.spray_decision || null;
  const message = data.message || "";

  // ---- One clear, HEALTH-FIRST headline ---------------------------------
  let heroClass, heroEmoji, heroTitle;
  switch (status) {
    case "Healthy":
      heroClass = "healthy"; heroEmoji = "\U0001f7e2";
      heroTitle = Lang.key("plant_is_healthy");
      break;
    case "Diseased":
      heroClass = "diseased"; heroEmoji = "\U0001f534";
      heroTitle = Lang.key("disease_detected");
      break;
    case "Not a Plant":
      heroClass = "notplant"; heroEmoji = "\u274c";
      heroTitle = Lang.key("not_a_plant");
      break;
    case "Low Plant Confidence":
      heroClass = "unknown"; heroEmoji = "\u26a0\ufe0f";
      heroTitle = Lang.key("low_confidence");
      break;
    default: // "Analysis Not Available" and anything unknown - never guess
      heroClass = "unknown"; heroEmoji = "\u26a0\ufe0f";
      heroTitle = Lang.key("analysis_not_available");
      break;
  }

  const stoppedNote = data.pipeline_stopped
    ? `<p class="muted" style="margin-top:8px;">⚠ ${Lang.key("pipeline_stopped")}</p>` : "";

  const plantValLine = (vconf != null)
    ? `<div class="muted" style="margin-top:4px;">${Lang.key("plant_validation_confidence")}: ${Number(vconf).toFixed(1)}%</div>`
    : "";

  // ---- PLANT HEALTH ANALYSIS block (Healthy / Diseased only) --------------
  let analysisCard = "";
  if (status === "Healthy" || status === "Diseased") {
    const sevClass = (sev == null) ? "unknown"
      : (sev === "High") ? "high" : (sev === "Medium") ? "medium" : "low";
    const sevEmoji = (sev == null) ? "❓"
      : (sev === "High") ? "🔴" : (sev === "Medium") ? "🟡" : "🟢";
    const sevKey = "severity_" + String(sev || "unknown").toLowerCase();
    const sevLabel = Lang.key(sevKey);
    const healthLabel = (status === "Healthy")
      ? Lang.key("health_status_healthy") : Lang.key("health_status_diseased");

    const pctLine = (pct != null)
      ? `<div style="display:flex;justify-content:space-between;"><span>${Lang.key("affected_leaf_area")}</span><b>${Number(pct).toFixed(1)}%</b></div>`
      : "";
    const w = (pct != null) ? Math.min(Math.max(Number(pct), 0), 100) : 0;

    const decisionLine = decision
      ? `<div style="margin-top:10px;border-top:1px dashed var(--border,#ddd);padding-top:8px;">
           <div class="card-title" style="margin-bottom:4px;">🚿 ${Lang.key("spray_decision")}</div>
           <div style="font-size:1.05rem;font-weight:700;">${decision.label}</div>
           <div class="muted" style="margin-top:2px;">${decision.message || ""}</div>
         </div>`
      : "";

    analysisCard = `
    <div class="card">
      <div class="card-title">🌿 ${Lang.key("plant_health_analysis")}</div>
      <div style="display:flex;justify-content:space-between;"><span>${Lang.key("health_status")}</span><b>${healthLabel}</b></div>
      ${pctLine}
      <div style="display:flex;justify-content:space-between;margin-top:4px;"><span>${Lang.key("severity_label")}</span><b><span class="pill ${sevClass}">${sevEmoji} ${sevLabel}</span></b></div>
      <div class="sev-progress" style="margin-top:8px;"><div class="sev-progress-fill ${sevClass}" style="width:${w}%"></div></div>
      <div class="sev-scale"><span>${Lang.key("severity_low")} 0-${thr.low_max}%</span><span>${Lang.key("severity_medium")} ${thr.low_max}-${thr.med_max}%</span><span>${Lang.key("severity_high")} &gt;${thr.med_max}%</span></div>
      <div class="muted" style="margin-top:6px;font-size:.82rem;">${data.severity_note || ""}</div>
      ${decisionLine}
    </div>`;
  }

  // ---- Original photo + real AI overlay side by side ----------------------
  let overlayRow = "";
  if (data.overlay_data_uri) {
    const orig = currentBlob ? URL.createObjectURL(currentBlob) : "";
    overlayRow = `
      <div class="card">
        <div class="card-title">🖼 ${Lang.key("detection_overlay")}</div>
        <div style="display:flex;gap:8px;">
          <div style="flex:1;"><div class="muted" style="font-size:.8rem;margin-bottom:2px;">${Lang.key("original_photo")}</div>
            ${orig ? `<img class="overlay-img" src="${orig}" alt="${Lang.key("original_photo")}">` : ""}
          </div>
          <div style="flex:1;"><div class="muted" style="font-size:.8rem;margin-bottom:2px;">${Lang.key("ai_overlay")}</div>
            <img class="overlay-img" src="${data.overlay_data_uri}" alt="${Lang.key("ai_overlay")}">
          </div>
        </div>
        <div class="muted" style="font-size:.8rem;margin-top:4px;">${Lang.key("overlay_caption")}</div>
      </div>`;
  }

  // ---- Pipeline stage trace (reports the real steps that ran) --------------
  const stageMap = {
    "Plant Validation": "stage_validator",
    "Health Segmentation": "stage_segmentation",
  };
  let pipelineHtml = "";
  if (Array.isArray(data.pipeline) && data.pipeline.length) {
    pipelineHtml = data.pipeline.map(sp => {
      const label = Lang.key(stageMap[sp.stage] || sp.stage);
      const res = sp.result || "—";
      const extra = (sp.affected_area_pct != null)
        ? ` · ${sp.affected_area_pct}%`
        : (sp.confidence != null)
          ? ` · ${Number(sp.confidence).toFixed(1)}%` : "";
      return `<div class="pipeline-row"><span>${label}</span><span class="muted">${res}${extra}</span></div>`;
    }).join("");
    pipelineHtml = `
      <div class="card">
        <div class="card-title">🧬 ${Lang.key("stage_result")}</div>
        ${pipelineHtml}
      </div>`;
  }

  // ---- Notes ----------------------------------------------------------------
  let noteCard = "";
  if (status === "Not a Plant" || status === "Low Plant Confidence"
      || status === "Analysis Not Available") {
    noteCard = `<div class="card">${message ? `<p>${message}</p>` : ""}</div>`;
  } else {
    const warning = (status === "Healthy" || status === "Diseased")
      ? `<p class="muted" style="margin-top:8px;">${Lang.key("pesticide_warning")}</p>` : "";
    const healthyMsg = (status === "Healthy")
      ? `<p class="muted">${Lang.key("no_disease_symptoms")}</p>` : "";
    noteCard = `<div class="card">
      <p style="white-space:pre-wrap">${message || ""}</p>
      ${healthyMsg}
      ${warning}
      <div class="muted" style="font-size:.8rem;margin-top:6px;">${Lang.key("detection_time")}: ${timeStr}</div>
    </div>`;
  }

  area.innerHTML = `
    <div class="result-hero ${heroClass}">
      <div class="emoji">${heroEmoji}</div>
      <h2>${heroTitle}</h2>
      ${plantValLine}
      ${stoppedNote}
    </div>

    ${analysisCard}

    ${overlayRow}

    ${pipelineHtml}

    ${noteCard}

    <div class="center" style="padding:6px 16px;">
      <button class="btn btn-outline btn-sm" onclick="window.location.href='history.html'">${Lang.key("history")}</button>
    </div>
  `;

  // Save to history with thumbnail
  saveToHistory(data, timeStr);
}

/* ---------- Save scan result (with thumbnail) to localStorage ---------- */
function saveToHistory(data, timeStr) {
  const url = currentBlob ? URL.createObjectURL(currentBlob) : "";
  const status = data.status || "Unknown";
  const record = {
    id: Date.now(),
    date: new Date().toISOString(),
    timeLabel: timeStr,
    crop: "Unknown",
    plant: "Unknown",
    plant_validation_confidence: data.plant_validation_confidence,
    status: status,
    severity: data.severity || null,
    affected_area_pct: data.affected_area_pct ?? null,
    spray_decision: (data.spray_decision && data.spray_decision.code) || null,
    thumb: url,
  };
  HistoryStore.add(record);
  showToast(Lang.key("save_to_history"));
}