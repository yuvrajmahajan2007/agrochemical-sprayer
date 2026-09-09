/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - app glue: splash, navigation, toast, history store
   ============================================================ */

/* ---------- History persistence (localStorage) ---------- */
const HistoryStore = {
  KEY: "AI-Driven Precision Agrochemical Sprayer_scan_history",

  _read() {
    try { return JSON.parse(localStorage.getItem(this.KEY) || "[]"); }
    catch (e) { return []; }
  },
  _write(list) { localStorage.setItem(this.KEY, JSON.stringify(list)); },

  add(record) {
    const list = this._read();
    list.unshift(record);
    if (list.length > 500) list.length = 500;
    this._write(list);
  },
  all() { return this._read(); },
  clear() { localStorage.removeItem(this.KEY); },

  /* today's count */
  todayCount() {
    const today = new Date().toDateString();
    return this.all().filter(r => new Date(r.date).toDateString() === today).length;
  },

  /* aggregate stats from real history */
  stats() {
    const list = this.all();
    const counts = { Healthy: 0, Diseased: 0, Unknown: 0 };
    const cropCount = {};
    const diseaseCount = {};
    list.forEach(r => {
      const st = r.status || "Unknown";
      if (counts[st] !== undefined) counts[st]++; else counts.Unknown++;
      cropCount[r.crop] = (cropCount[r.crop] || 0) + 1;
      if (r.disease) diseaseCount[r.disease] = (diseaseCount[r.disease] || 0) + 1;
    });
    const total = list.length;
    const pct = c => total ? Math.round((c / total) * 1000) / 10 : 0;
    let mostCrop = null, mostCropCount = 0;
    Object.entries(cropCount).forEach(([c, n]) => { if (n > mostCropCount) { mostCropCount = n; mostCrop = c; } });
    return {
      total,
      healthy: counts.Healthy,
      diseased: counts.Diseased,
      unknown: counts.Unknown,
      healthyPct: pct(counts.Healthy),
      diseasedPct: pct(counts.Diseased),
      unknownPct: pct(counts.Unknown),
      mostDetectedCrop: mostCrop,
      diseaseCount,
      cropCount,
      today: this.todayCount(),
    };
  },
};

/* ---------- Toast ---------- */
function showToast(msg) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 3000);
}

/* ---------- Splash (index.html) ---------- */
function runSplash() {
  const splash = document.getElementById("splash");
  if (!splash) return;
  setTimeout(() => {
    splash.classList.add("hide");
    setTimeout(() => {
      location.href = "language.html";
    }, 500);
  }, 2200);
}

/* ---------- Auth guard for protected pages ---------- */
function requireAuth() {
  if (!Auth.isLoggedIn()) {
    location.href = "login.html";
    return false;
  }
  return true;
}

/* ---------- Bottom nav active state ---------- */
function initNav(pageId) {
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.page === pageId);
  });
}

/* ---------- Settings helpers reused across pages ---------- */
const SettingsStore = {
  get lang() { return Lang.current; },
  get notifications() { return localStorage.getItem("AI-Driven Precision Agrochemical Sprayer_notifications") !== "0"; },
  setNotifications(v) { localStorage.setItem("AI-Driven Precision Agrochemical Sprayer_notifications", v ? "1" : "0"); },
};

/* ---------- generic fetch wrapper with net error handling ---------- */
async function appFetch(fn) {
  try {
    return await fn();
  } catch (e) {
    showToast(Lang.key("network_error"));
    throw e;
  }
}