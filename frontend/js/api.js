/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - API client
   Communicates with the Flask backend which runs the REAL AI model.

   Backend base URL resolution (phone-safe, never localhost/127.0.0.1):
     1. explicit override the farmer saved in Settings (localStorage)
     2. backend-provided base URL from /api/config
        (env SPRAYBOT_API_BASE_URL > request.Host of the served page)
     3. window.location.origin  -> the phone loads the page from the
        laptop's LAN IP, so this is already http://<laptop-ip>:5000
   ============================================================ */

const Api = {
  KEY_URL: "AI-Driven Precision Agrochemical Sprayer_backend_url",
  _detectedUrl: null,

  get savedUrl() {
    const saved = localStorage.getItem(this.KEY_URL);
    return (saved && saved.trim()) ? saved.trim().replace(/\/+$/, "") : "";
  },

  /* Resolve the backend base URL (async: may consult /api/config). */
  async resolveBackendUrl() {
    if (this.savedUrl) return this.savedUrl;                 // 1. manual override
    if (this._detectedUrl) return this._detectedUrl;         // cached detect
    try {
      // 2. same-origin, always reachable, tells us the real LAN base URL
      const r = await fetch("./api/config", { cache: "no-store" });
      if (r.ok) {
        const d = await r.json();
        if (d && d.base_url) {
          this._detectedUrl = String(d.base_url).replace(/\/+$/, "");
          return this._detectedUrl;
        }
      }
    } catch (e) { /* offline -> fall through */ }
    // 3. host that actually served this page (laptop LAN IP on phone)
    return window.location.origin;
  },

  async getBackendUrl() {
    return this.resolveBackendUrl();
  },

  setBackendUrl(url) {
    this._detectedUrl = url.replace(/\/+$/, "");
    localStorage.setItem(this.KEY_URL, this._detectedUrl);
  },

  get confidenceThreshold() {
    const t = parseFloat(localStorage.getItem("AI-Driven Precision Agrochemical Sprayer_threshold"));
    return isNaN(t) ? 70.0 : t;
  },

  setConfidenceThreshold(t) {
    localStorage.setItem("AI-Driven Precision Agrochemical Sprayer_threshold", String(t));
  },

  async _fetch(path, options = {}) {
    const base = await this.resolveBackendUrl();
    return fetch(base + path, options);
  },

  /* ---------- health / status ---------- */
  async status() {
    const r = await this._fetch("/status");
    if (!r.ok) throw new Error("status_failed");
    return r.json();
  },

  /* ---------- supported crops ---------- */
  async supportedCrops() {
    const r = await this._fetch("/supported_crops");
    if (!r.ok) throw new Error("crops_failed");
    return r.json();
  },

  /* ---------- REAL prediction ---------- */
  async predict(imageBlob) {
    const fd = new FormData();
    fd.append("image", imageBlob, "plant.jpg");
    let r;
    try {
      r = await this._fetch("/predict", { method: "POST", body: fd });
    } catch (e) {
      const nerr = new Error("network_unreachable");
      nerr.isNetwork = true;
      throw nerr;
    }
    const data = await r.json().catch(() => null);
    if (!r.ok || !data) {
      const err = new Error(data && data.message ? data.message : "predict_failed");
      throw err;
    }
    return data;
  },
};

/* ---------- connectivity test ---------- */
function testBackend() {
  return Api.status()
    .then(r => ({ ok: true, data: r }))
    .catch(() => ({ ok: false }));
}