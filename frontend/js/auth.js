/* ============================================================
   AI-Driven Precision Agrochemical Sprayer - Prototype Authentication (localStorage)
   ============================================================ */

const Auth = {
  KEY_USERS: "AI-Driven Precision Agrochemical Sprayer_users",
  KEY_SESSION: "AI-Driven Precision Agrochemical Sprayer_session",

  /* ---------- storage helpers ---------- */
  _readUsers() {
    try {
      return JSON.parse(localStorage.getItem(this.KEY_USERS) || "{}");
    } catch (e) {
      return {};
    }
  },
  _writeUsers(users) {
    localStorage.setItem(this.KEY_USERS, JSON.stringify(users));
  },

  /* ---------- registration ---------- */
  register({ name, mobile, password }) {
    const users = this._readUsers();
    if (users[mobile]) {
      return { ok: false, message: "auth_failed_existing" };
    }
    users[mobile] = { name, mobile, password, created_at: Date.now() };
    this._writeUsers(users);
    return { ok: true, message: "register_success" };
  },

  /* ---------- login ---------- */
  login(mobile, password, remember = true) {
    const users = this._readUsers();
    const user = users[mobile];
    if (!user || user.password !== password) {
      return { ok: false, message: "auth_failed" };
    }
    if (remember) {
      localStorage.setItem(this.KEY_SESSION, JSON.stringify({
        mobile, name: user.name, ts: Date.now(),
      }));
    } else {
      sessionStorage.setItem(this.KEY_SESSION, JSON.stringify({
        mobile, name: user.name, ts: Date.now(),
      }));
    }
    return { ok: true, message: "login_success", user };
  },

  logout() {
    localStorage.removeItem(this.KEY_SESSION);
    sessionStorage.removeItem(this.KEY_SESSION);
  },

  /* ---------- session ---------- */
  session() {
    let s = null;
    try { s = JSON.parse(localStorage.getItem(this.KEY_SESSION)); } catch (e) {}
    if (!s) { try { s = JSON.parse(sessionStorage.getItem(this.KEY_SESSION)); } catch (e) {} }
    return s;
  },

  isLoggedIn() {
    return !!this.session();
  },

  currentUser() {
    const s = this.session();
    if (!s) return null;
    const users = this._readUsers();
    return users[s.mobile] || null;
  },

  updateProfile({ name, mobile }) {
    const s = this.session();
    if (!s) return { ok: false };
    const users = this._readUsers();
    if (users[s.mobile]) {
      users[s.mobile].name = name;
      if (mobile && mobile !== s.mobile) {
        const newUsers = {};
        Object.keys(users).forEach(k => {
          if (k === s.mobile) {
            newUsers[mobile] = { ...users[k], name, mobile };
          } else {
            newUsers[k] = users[k];
          }
        });
        localStorage.setItem(this.KEY_USERS, JSON.stringify(newUsers));
        s.mobile = mobile;
      } else {
        this._writeUsers(users);
      }
      s.name = name;
      localStorage.setItem(this.KEY_SESSION, JSON.stringify(s));
    }
    return { ok: true };
  },
};

/* ---------- helpers for validation ---------- */
function validMobile(m) {
  return /^\d{10}$/.test(m);
}
function validPassword(p) {
  return p && p.length >= 4;
}

/* ---------- page init ---------- */
function initAuthPage() {
  const mode = new URLSearchParams(location.search).get("mode");
  const isLogin = mode !== "register";
  document.querySelectorAll("#tab-login, #tab-register").forEach(el => {
    el.classList.toggle("active",
      (el.id === "tab-login") === isLogin);
  });
}