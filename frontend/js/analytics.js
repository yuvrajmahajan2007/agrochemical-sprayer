/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - Analytics page
   Charts built ONLY from real scan history (localStorage).
   ============================================================ */

let chartCrop = null, chartDisease = null;

async function initAnalytics() {
  if (!requireAuth()) return;
  initNav("analytics");

  const s = HistoryStore.stats();
  const box = document.getElementById("analytics-box");

  if (s.total === 0) {
    box.innerHTML = `<div class="card center"><p class="muted">${Lang.key("no_analytics_data")}</p></div>`;
    return;
  }

  // Summary stats
  box.innerHTML = `
    <div class="stats-grid">
      <div class="stat blue"><div class="num">${s.total}</div><div class="lbl">${Lang.key("total_scans")}</div></div>
      <div class="stat green"><div class="num">${s.healthyPct}%</div><div class="lbl">${Lang.key("healthy_pct")}</div></div>
      <div class="stat red"><div class="num">${s.diseasedPct}%</div><div class="lbl">${Lang.key("diseased_pct")}</div></div>
      <div class="stat warn"><div class="num">${s.unknownPct}%</div><div class="lbl">${Lang.key("unknown_pct")}</div></div>
    </div>

    <div class="card">
      <div class="card-title">📊 ${Lang.key("disease_distribution")}</div>
      <div class="chart-box"><canvas id="ch-disease"></canvas></div>
    </div>

    <div class="card">
      <div class="card-title">🌾 ${Lang.key("crop_distribution")}</div>
      <div class="chart-box"><canvas id="ch-crop"></canvas></div>
    </div>

    <div class="card">
      <div class="card-title">🏆 ${Lang.key("most_detected_crop")}</div>
      <div style="font-size:1.4rem;font-weight:800;">${s.mostDetectedCrop || "—"}</div>
    </div>
  `;

  /* Very small Chart.js-free bar renderer to keep things dependency-free.
     Draw a simple horizontal bar using divs (responsive, no external lib). */
  drawBars("ch-disease", s.diseaseCount, "#e74c3c");
  drawBars("ch-crop", s.cropCount, "#1e7e34");
}

function drawBars(canvasId, data, color) {
  const cv = document.getElementById(canvasId);
  if (!cv) return;
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = Math.max(1, ...entries.map(e => e[1]));
  cv.outerHTML = `<div>
    ${entries.map(([k, v]) => `
      <div style="margin:6px 0;">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
          <span>${k}</span><span>${v}</span>
        </div>
        <div style="background:#eef2ee;border-radius:6px;height:12px;margin-top:3px;">
          <div style="width:${(v / max) * 100}%;height:100%;background:${color};border-radius:6px;"></div>
        </div>
      </div>`).join("")}
  </div>`;
}