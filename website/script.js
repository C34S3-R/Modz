// Modz 3D Lab - frontend control panel for the general 3D processing server.
// The browser is only the interface: all heavy processing happens on the
// backend. This file talks to the REST API and renders state.

// ---------------------------------------------------------------------------
// API layer
// ---------------------------------------------------------------------------

const API_BASE = "/api";

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && data.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch { /* ignore body parse errors */ }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res.text();
}

function fmtSpeed(bytesPerSec) {
  return `${fmtBytes(bytesPerSec)}/s`;
}

// POST multipart with live upload progress. fetch() has no upload-progress
// API, so large project files use XHR here. onProgress(loaded, total) fires
// while bytes leave the browser; the promise resolves when the server has
// fully received AND saved the files.
function apiUpload(path, fd, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${path}`);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* non-JSON body */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      let detail = `${xhr.status} ${xhr.statusText}`;
      if (data && data.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      reject(new Error(detail));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.ontimeout = () => reject(new Error("Upload timed out"));
    if (xhr.upload && typeof onProgress === "function") {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) onProgress(ev.loaded, ev.total);
      };
    }
    xhr.send(fd);
  });
}

const API_ENDPOINTS = {
  serverStatus: () => api("/status"),
  listProjects: () => api("/projects"),
  getProject: (id) => api(`/projects/${id}`),
  createProject: (meta, files) => {
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    for (const f of files) if (f) fd.append("files", f, f.name);
    return api("/projects", { method: "POST", body: fd });
  },
  addFiles: (id, files, onProgress) => {
    const fd = new FormData();
    for (const f of files) if (f) fd.append("files", f, f.name);
    return apiUpload(`/projects/${id}/upload`, fd, onProgress);
  },
  createProjectWithProgress: (meta, files, onProgress) => {
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    for (const f of files) if (f) fd.append("files", f, f.name);
    return apiUpload("/projects", fd, onProgress);
  },
  startProcessing: (id) => api(`/projects/${id}/process`, { method: "POST" }),
  pauseProcessing: (id) => api(`/projects/${id}/pause`, { method: "POST" }),
  cancelProcessing: (id) => api(`/projects/${id}/cancel`, { method: "POST" }),
  resumeProcessing: (id) => api(`/projects/${id}/resume`, { method: "POST" }),
  retryProcessing: (id) => api(`/projects/${id}/retry`, { method: "POST" }),
  projectStatus: (id) => api(`/projects/${id}/status`),
  projectLogs: (id) => api(`/projects/${id}/logs`),
  projectFiles: (id) => api(`/projects/${id}/files`),
  saveOptions: (id, options) => api(`/projects/${id}/options`, { method: "PUT", body: JSON.stringify({ options }) }),
  importSources: (id, urls) => api(`/projects/${id}/import`, { method: "POST", body: JSON.stringify({ urls }) }),
  searchImages: (q, source = "all") => api(`/search/images?q=${encodeURIComponent(q)}&source=${encodeURIComponent(source)}`),
  sourceCheck: (id) => api(`/projects/${id}/source-check`),
  advisor: (id) => api(`/projects/${id}/advisor`),
  deleteFile: (id, path) => api(`/projects/${id}/files?path=${encodeURIComponent(path)}`, { method: "DELETE" }),
  downloadFile: (id, path) => `${API_BASE}/projects/${id}/files/download?path=${encodeURIComponent(path)}`,
  modelInfo: (id) => api(`/projects/${id}/model`),
  lights: (id) => api(`/projects/${id}/lights`),
  updateLight: (id, light) => api(`/projects/${id}/lights`, { method: "PUT", body: JSON.stringify(light) }),
  deleteLight: (id, name) => api(`/projects/${id}/lights/${encodeURIComponent(name)}`, { method: "DELETE" }),
  lightAnalysis: (id) => api(`/projects/${id}/lights/analysis`),
  exportChecklist: (id) => api(`/projects/${id}/export/checklist`),
  buildExport: (id) => api(`/projects/${id}/export`, { method: "POST" }),
  cancelExport: (id) => api(`/projects/${id}/export/cancel`, { method: "POST" }),
  exportStatus: (id) => api(`/projects/${id}/export/status`),
  settings: () => api("/settings"),
  saveSettings: (data) => api("/settings", { method: "PUT", body: JSON.stringify(data) }),
  clearLogs: () => api("/logs", { method: "DELETE" }),
  downloadLogs: () => `${API_BASE}/logs/download`,
};

// Alias used across all views: API.listProjects(), API.createProject(), ...
const API = API_ENDPOINTS;

// Real-time updates: WebSocket when available, SSE fallback, polling last.
const Live = {
  ws: null,
  es: null,
  listeners: new Set(),
  connect() {
    if (this.ws || this.es) return;
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onmessage = (ev) => this.receive(ev.data);
      ws.onerror = () => ws.close();
      ws.onclose = () => {
        if (this.ws === ws) this.ws = null;
        this.connectSSE();
      };
      this.ws = ws;
    } catch {
      this.connectSSE();
    }
  },
  connectSSE() {
    if (this.es || this.ws) return;
    try {
      const es = new EventSource(`${API_BASE}/events`);
      es.onmessage = (ev) => this.receive(ev.data);
      es.onerror = () => {
        es.close();
        if (this.es === es) this.es = null;
        setTimeout(() => this.connect(), 3000);
      };
      this.es = es;
    } catch {
      setTimeout(() => this.connect(), 3000);
    }
  },
  receive(raw) {
    try {
      const event = JSON.parse(raw);
      this.listeners.forEach((fn) => fn(event));
    } catch { /* bad frame */ }
  },
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  projects: [],
  currentProject: null,
  projectStatus: null,
  logs: [],
  lights: [],
  settings: {},
  server: null,
  route: "dashboard",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  // Boolean attributes (disabled, checked, ...) are presence-based: a
  // false value must remove the attribute, not render disabled="false".
  const BOOL_ATTRS = new Set(["disabled", "checked", "readonly", "required",
    "hidden", "selected", "multiple", "autofocus", "open"]);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") { if (v != null && v !== "") node.className = v; }
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === null || v === undefined || (BOOL_ATTRS.has(k) && v === false)) continue;
    else if (BOOL_ATTRS.has(k) && v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message, kind = "info") {
  const box = el("div", { class: `toast toast-${kind}` }, message);
  document.getElementById("toasts").append(box);
  setTimeout(() => box.classList.add("toast-out"), 3800);
  setTimeout(() => box.remove(), 4200);
}

function fmtBytes(n) {
  if (!n && n !== 0) return "--";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function elapsed(seconds) {
  if (seconds == null) return "--";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function progressBar(percent, status) {
  const p = Math.max(0, Math.min(100, percent ?? 0));
  // Freeze the crawl for terminal/idle states so a finished bar doesn't
  // look like it is still building.
  let stateClass = "";
  if (status === "failed" || status === "error") stateClass = " is-error";
  else if (status === "paused") stateClass = " is-paused";
  else if (status === "completed" || status === "done") stateClass = " is-done";
  else if (status && !["running", "processing", "queued"].includes(status)) stateClass = " is-idle";
  return el("div", { class: "progress" + stateClass, "data-pct": Math.round(p) },
    el("div", { style: `width:${p}%` }));
}

function statusDot(status) {
  const map = { running: "st-run", done: "st-done", completed: "st-done", failed: "st-fail", error: "st-fail", cancelled: "st-fail", paused: "st-pause", skipped: "st-skip", pending: "st-wait", queued: "st-wait" };
  return el("span", { class: `dot ${map[status] || "st-wait"}` });
}

// Source-quality report from GET /api/projects/{id}/source-check.
// Same gate the Video Validation stage enforces: ok:false means the
// pipeline will stop until better videos/images are supplied.
function sourceCheckView(rep) {
  const box = el("div");
  const head = el("div", { class: "row-between" },
    el("span", { class: rep.ok ? "st-done" : "st-fail" },
      `${rep.ok ? "✓ Good source" : "✕ Weak source"} (${rep.score || "--"})`),
    el("span", { class: "muted" }, rep.source === "images" ? "image set" : "video"),
  );
  box.append(head);
  for (const c of rep.checks || []) {
    box.append(el("div", { class: c.ok ? "st-done" : "st-fail" }, `${c.ok ? "✓" : "✕"} ${c.name} — ${c.detail}`));
  }
  for (const w of rep.warnings || []) box.append(el("div", { class: "muted" }, `! ${w}`));
  if (!rep.ok && rep.guidance) box.append(el("div", { class: "muted mt" }, rep.guidance));
  return box;
}

// AI advisor report from GET /api/projects/{id}/advisor. Read-only:
// rule findings always render, local-LLM summary appended when available.
function advisorView(rep) {
  const box = el("div", { class: "card mt" },
    el("div", { class: "panel-label" },
      `AI Advisor ${rep.ai_available ? `(local ${rep.ai_model})` : "(rules only — Ollama offline)"}`));
  box.append(el("div", { class: "muted" },
    `Type: ${rep.project_type} · Preset: ${rep.preset} · Suggested: ${rep.suggested_preset} · Status: ${rep.status || "idle"}`));
  box.append(el("div", { class: "panel-label mt" }, "Findings"));
  for (const f of rep.findings || []) box.append(el("div", {}, `• ${f}`));
  box.append(el("div", { class: "panel-label mt" }, "Next actions"));
  for (const a of rep.actions || []) box.append(el("div", {}, `→ ${a}`));
  if (rep.ai_summary) box.append(el("div", { class: "mt" }, el("em", {}, rep.ai_summary)));
  return box;
}

const PIPELINE_STAGES = [
  { key: "validation", label: "Video Validation" },
  { key: "extraction", label: "Frame Extraction" },
  { key: "filtering", label: "Frame Filtering" },
  { key: "preprocessing", label: "Preprocessing" },
  { key: "meshroom", label: "Meshroom Reconstruction" },
  { key: "mesh_validation", label: "Mesh Validation" },
  { key: "blender", label: "Blender Processing" },
  { key: "lights", label: "Light Detection" },
  { key: "light_creation", label: "Light Creation" },
  { key: "light_validation", label: "Light Validation" },
  { key: "bussid_prep", label: "Export Preparation (BUSSID optional)" },
  { key: "export", label: "Export" },
  { key: "package", label: "Package" },
];

const STAGE_ICONS = {
  pending: "○", waiting: "○", queued: "○",
  running: "◐", active: "◐",
  done: "✓", completed: "✓",
  failed: "✕", error: "✕", cancelled: "✕",
  skipped: "–", paused: "‖",
};
const STAGE_CLASSES = {
  pending: "st-wait", waiting: "st-wait", queued: "st-wait", idle: "st-wait",
  running: "st-run", active: "st-run",
  done: "st-done", completed: "st-done",
  failed: "st-fail", error: "st-fail", cancelled: "st-fail",
  skipped: "st-skip", paused: "st-pause",
};

// Never index STAGE_CLASSES raw: unknown statuses must still render.
function stageClass(status) {
  return STAGE_CLASSES[status] || "st-wait";
}

function stageState(stage) {
  return (stage && stage.status) || "waiting";
}

function pipelineList(stages, onPick) {
  const list = el("ul", { class: "pipeline" });
  const items = stages && stages.length ? stages : PIPELINE_STAGES.map((s) => ({ key: s.key, label: s.label, status: "pending" }));
  for (const stage of items) {
    const st = stageState(stage);
    const li = el("li", { class: stageClass(st) },
      el("span", { class: "pipe-icon" }, STAGE_ICONS[st] || "○"),
      el("span", { class: "pipe-label" }, stage.label || stage.key || stage.operation || "Stage"),
      el("span", { class: "pipe-state" }, st),
    );
    if (onPick) li.addEventListener("click", () => onPick(stage));
    list.append(li);
  }
  return list;
}

function stageDetail(stage) {
  if (!stage) return null;
  const st = stageState(stage);
  const box = el("div", { class: "card mt" });
  box.append(el("h3", {}, stage.label || stage.key || "Stage"));
  box.append(
    el("div", { class: "kv" }, el("span", {}, "Status"), el("span", { class: stageClass(st) }, st)),
  );
  for (const [k, v] of Object.entries(stage)) {
    if (["key", "label", "status"].includes(k) || v == null) continue;
    box.append(el("div", { class: "kv" }, el("span", {}, k.replace(/_/g, " ")), el("span", {}, String(v))));
  }
  return box;
}

// ---------------------------------------------------------------------------
// Construction progress hero
// ---------------------------------------------------------------------------

// Last percentage shown per hero key. Views rebuild on every live push, so
// the count-up animates from the previous value across rebuilds instead of
// restarting at 0. Keyed by project so multiple active jobs don't fight.
const pctStore = {};

function animatePct(node, target, key) {
  const to = Math.max(0, Math.min(100, Math.round(target ?? 0)));
  const from = pctStore[key] ?? 0;
  pctStore[key] = to;
  if (from === to) { node.textContent = `${to}%`; return; }
  const t0 = performance.now();
  const dur = 550;
  const step = (now) => {
    if (!node.isConnected) return; // superseded by a newer render
    const k = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    node.textContent = `${Math.round(from + (to - from) * eased)}%`;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function buildHero({ status, stage, pct, sub, compact, pctKey }) {
  const p = Math.max(0, Math.min(100, pct ?? 0));
  const st = status || "idle";
  const key = pctKey != null ? String(pctKey) : "default";
  const pctNode = el("div", { class: "build-pct" }, "0%");
  const hero = el("div", {
    class: "build-hero" + (compact ? " build-hero-compact" : ""),
    "data-status": st,
  },
    el("div", { class: "build-hero-row" },
      el("div", { class: "build-pct-wrap" },
        pctNode,
        el("span", { class: "build-pct-label" }, "complete"),
      ),
      el("div", { class: "build-meta" },
        el("span", { class: `build-status ${stageClass(st)}` }, statusDot(st), st),
        el("div", { class: "build-stage" },
          el("span", { class: "build-stage-dot" }),
          stage || (["running", "queued"].includes(st) ? "waiting for worker…" : "idle"),
        ),
        sub ? el("div", { class: "build-sub" }, sub) : null,
      ),
    ),
    progressBar(p, st),
  );
  // Count-up runs after insertion so the transition is visible.
  requestAnimationFrame(() => animatePct(pctNode, p, key));
  return hero;
}

// ---------------------------------------------------------------------------
// Log panel component
// ---------------------------------------------------------------------------

function logPanel(entries, { maxHeight } = {}) {
  const panel = el("div", { class: "log-panel" });
  if (maxHeight) panel.style.maxHeight = `${maxHeight}px`;
  const pre = el("pre");
  if (!entries || entries.length === 0) {
    pre.textContent = "No log entries yet.";
  } else {
    pre.textContent = entries
      .map((e) => {
        const line = `[${e.time}] ${e.message}`;
        if (e.level === "error") return line;
        return line;
      })
      .join("\n");
  }
  panel.append(pre);
  panel.scrollTop = panel.scrollHeight;
  return panel;
}

function logActions(entries, onClear) {
  const download = () => {
    const blob = new Blob([entries.map((e) => `[${e.time}] ${e.message}`).join("\n")], { type: "text/plain" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "process.log" });
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return el("div", { class: "row mt" },
    el("button", { class: "btn", onclick: onClear }, "Clear"),
    el("button", { class: "btn", onclick: download }, "Download Log"),
    el("button", {
      class: "btn",
      onclick: () => navigator.clipboard.writeText(entries.map((e) => `[${e.time}] ${e.message}`).join("\n")).then(() => toast("Log copied", "ok")),
    }, "Copy"),
  );
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

const views = {};

// ----- Dashboard -----

views.dashboard = function renderDashboard() {
  const v = el("div");
  const s = state.server;

  const hero = el("div", { class: "card" },
    el("h2", {}, "Vehicle Processing Server"),
    el("p", { class: "muted" }, "Local processing powered by your server"),
    el("div", { class: "row mt" },
      el("span", { class: `dot ${s?.online ? "dot-on" : "dot-off"}` }),
      el("span", { class: "stat-num" }, s?.online ? "ONLINE" : "OFFLINE"),
    ),
  );

  const stats = el("div", { class: "grid grid-3 mt" });
  const metrics = [
    ["CPU", s ? `${Math.round(s.cpu_percent ?? 0)}%` : "--", s ? (s.cpu_percent ?? 0) / 100 : null],
    ["RAM", s ? `${(s.ram_used_gb ?? 0).toFixed(1)} / ${(s.ram_total_gb ?? 0).toFixed(1)} GB` : "--",
      s && s.ram_total_gb ? s.ram_used_gb / s.ram_total_gb : null],
    ["Storage", s ? `${(s.disk_used_gb ?? 0).toFixed(0)} / ${(s.disk_total_gb ?? 0).toFixed(0)} GB` : "--",
      s && s.disk_total_gb ? s.disk_used_gb / s.disk_total_gb : null],
    ["Uptime", s ? elapsed(s.uptime_seconds) : "--", null],
  ];
  for (const [label, value, ratio] of metrics) {
    stats.append(el("div", { class: "card metric-card" },
      el("div", { class: "stat-num" }, value),
      el("div", { class: "stat-label" }, label),
      ratio != null ? el("div", { class: "meter" },
        el("div", { style: `width:${Math.max(0, Math.min(1, ratio)) * 100}%` })) : null,
    ));
  }

  const actions = el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Quick Actions"),
    el("div", { class: "row mt" },
      el("button", { class: "btn btn-primary", onclick: openNewProjectModal }, "New Project"),
      el("button", { class: "btn", onclick: () => (location.hash = "#/projects") }, "Upload Model"),
      el("button", { class: "btn", onclick: () => (location.hash = "#/projects") }, "Upload Reference"),
      el("button", { class: "btn", onclick: continueProcessing }, "Continue Processing"),
      el("button", { class: "btn", onclick: viewLastResult }, "View Last Result"),
    ),
  );

  const current = renderCurrentProjectCard();
  if (current) v.append(hero, stats, actions, current);
  else v.append(hero, stats, actions);

  return v;
};

function renderCurrentProjectCard() {
  const p = state.projects.find((x) => ["queued", "processing", "running"].includes(x.status));
  if (!p) return null;
  const card = el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Current Project"),
    el("h3", {}, p.name),
    buildHero({
      status: p.status || "idle",
      stage: p.current_stage,
      pct: p.progress,
      pctKey: p.id,
      compact: true,
      sub: `Started ${p.started_at ? fmtTime(p.started_at) : "--"} · Elapsed ${elapsed(p.elapsed_seconds)}`,
    }),
    el("div", { class: "mt" }, el("button", { class: "btn", onclick: () => openProject(p.id) }, "Open Project")),
  );
  return card;
}

async function continueProcessing() {
  const p = state.projects.find((x) => ["queued", "paused", "failed"].includes(x.status));
  if (!p) return toast("No project to continue", "warn");
  try {
    await API.startProcessing(p.id);
    toast(`Processing resumed for ${p.name}`, "ok");
    openProject(p.id);
  } catch (e) { toast(e.message, "err"); }
}

async function viewLastResult() {
  const done = state.projects.filter((x) => x.status === "done" || x.status === "completed").pop();
  if (!done) return toast("No completed projects yet", "warn");
  openProject(done.id);
}

// ----- Projects -----

views.projects = function renderProjects() {
  const v = el("div");
  const head = el("div", { class: "row-between" },
    el("span", { class: "muted" }, `${state.projects.length} project(s)`),
    el("button", { class: "btn btn-primary", onclick: openNewProjectModal }, "New Project"),
  );
  v.append(head);

  if (state.projects.length === 0) {
    v.append(el("div", { class: "card mt muted" }, "No projects yet. Create one to get started."));
    return v;
  }

  const table = el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Name"), el("th", {}, "Status"), el("th", {}, "Stage"), el("th", {}, "Progress"), el("th", {}, "Created"), el("th", {}, ""),
    )),
  );
  const tbody = el("tbody");
  for (const p of state.projects) {
    tbody.append(el("tr", {},
      el("td", {}, p.name),
      el("td", { class: stageClass(stageState(p)) }, p.status || "idle"),
      el("td", {}, p.current_stage || "--"),
      el("td", {}, `${Math.round(p.progress ?? 0)}%`),
      el("td", { class: "faint" }, p.created_at ? new Date(p.created_at * 1000).toLocaleString() : "--"),
      el("td", {}, el("button", { class: "btn", onclick: () => openProject(p.id) }, "Open")),
    ));
  }
  table.append(tbody);
  v.append(el("div", { class: "card mt" }, table));
  return v;
};

function openNewProjectModal() {
  document.getElementById("modal-overlay").classList.remove("hidden");
}

function closeNewProjectModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
  document.getElementById("new-project-form").reset();
  const prog = document.getElementById("upload-progress");
  if (prog) prog.style.display = "none";
}

async function submitNewProject(event) {
  event.preventDefault();
  const form = event.target;
  const fd = new FormData(form);
  const videoUrl = (fd.get("video_url") || "").toString().trim();
  const meta = {
    name: fd.get("name"),
    description: fd.get("description") || "",
    video_url: videoUrl || undefined,
    project_type: fd.get("project_type") || "general",
    preset: fd.get("preset") || "small",
    options: {
      photogrammetry: form.photogrammetry.checked,
      lights: form.lights.checked,
      blender: form.blender.checked,
      bussid: form.bussid.checked,
      mask_background: form.mask_background.checked,
      force_weak_source: form.force_weak_source ? form.force_weak_source.checked : false,
    },
  };
  const files = ["video", "model", "images"]
    .map((k) => (k === "images" ? [...form.images.files] : form[k].files[0]))
    .flat()
    .filter(Boolean);

  // A build without a reference video fails in seconds at Video Validation.
  // Warn now instead of wasting an upload + a doomed pipeline run.
  // An online URL counts as a video source: the server fetches it itself.
  const hasVideo = videoUrl
    || files.some((f) => (f.type || "").startsWith("video/") || /\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(f.name || ""));
  if (!hasVideo && meta.options.photogrammetry) {
    if (!confirm("No reference video selected — the build will fail at Video Validation. Create the project anyway?")) return;
  }

  const btn = form.querySelector('button[type="submit"]');
  const prog = document.getElementById("upload-progress");
  const bar = document.getElementById("upload-bar");
  const status = document.getElementById("upload-status");
  const totalBytes = files.reduce((n, f) => n + (f.size || 0), 0);
  const t0 = performance.now();
  const setBar = (loaded, total) => {
    const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
    bar.style.width = pct.toFixed(1) + "%";
    const secs = Math.max(0.1, (performance.now() - t0) / 1000);
    status.textContent = total > 0
      ? `${fmtBytes(loaded)} / ${fmtBytes(total)} (${pct.toFixed(0)}%) · ${fmtSpeed(loaded / secs)}`
      : `${fmtBytes(loaded)} sent…`;
  };
  btn.disabled = true;
  btn.textContent = "Uploading...";
  prog.style.display = "";
  setBar(0, totalBytes);
  let modalClosed = false;
  let savingPill = null;
  const showSaving = () => {
    if (savingPill) return;
    savingPill = document.createElement("div");
    savingPill.textContent = "Saving on server…";
    savingPill.style.cssText = "position:fixed;bottom:18px;right:18px;z-index:100;padding:10px 16px;border-radius:10px;background:#17203a;color:#e8eefc;font-size:13px;border:1px solid #2c3a5e;box-shadow:0 8px 24px rgba(0,0,0,.4)";
    document.body.append(savingPill);
  };
  const hideSaving = () => { if (savingPill) { savingPill.remove(); savingPill = null; } };
  // Close the modal the instant all bytes leave the browser — waiting for
  // the server reply (disk write + DB) is what kept it hanging open.
  const onBytesOut = (loaded, total) => {
    const done = total || totalBytes;
    setBar(loaded, done);
    if (done > 0 && loaded >= done && !modalClosed) {
      modalClosed = true;
      closeNewProjectModal();
      showSaving();
      toast("Upload complete — saving on server…", "ok");
    }
  };
  try {
    const project = await API.createProjectWithProgress(meta, files, onBytesOut);
    hideSaving();
    toast(`Project "${project.name}" created`, "ok");
    closeNewProjectModal();
    await loadProjects();
    // Auto-build: upload is saved, so enqueue the processing pipeline right
    // away, then land on the Processing board where the live build progress
    // (stage + bar, pushed over the socket) is front and center.
    let buildRunning = false;
    try {
      await API.startProcessing(project.id);
      buildRunning = true;
      toast("Build started — watch live progress", "ok");
    } catch (e) {
      toast(`Auto-start failed: ${e.message} — start it manually from the project view.`, "err");
    }
    await loadProjects();
    if (buildRunning) location.hash = "#/processing";
    else openProject(project.id);
  } catch (e) {
    toast(`Create failed: ${e.message}`, "err");
  } finally {
    hideSaving();
    btn.disabled = false;
    btn.textContent = "Create Project";
    prog.style.display = "none";
    bar.style.width = "0%";
  }
}

function renderAddFiles(p) {
  const wrap = el("div", { class: "mt" });
  const input = el("input", {
    type: "file", multiple: "",
    accept: ".mp4,.mov,.mkv,.avi,.webm,.m4v,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.obj,.fbx,.glb,.gltf,.blend,.stl,.ply",
  });
  const bar = el("div", { style: "width:0%" });
  const barWrap = el("div", { class: "progress", style: "display:none" }, bar);
  const status = el("div", { class: "muted" }, "");
  const btn = el("button", { class: "btn btn-primary" }, "Upload Files");
  btn.addEventListener("click", async () => {
    const files = [...input.files];
    if (!files.length) { toast("Choose files first", "warn"); return; }
    btn.disabled = true;
    barWrap.style.display = "";
    const total = files.reduce((n, f) => n + (f.size || 0), 0);
    const t0 = performance.now();
    try {
      await API.addFiles(p.id, files, (loaded, done0) => {
        const done = done0 || total;
        const pct = done > 0 ? Math.min(100, (loaded / done) * 100) : 0;
        bar.style.width = pct.toFixed(1) + "%";
        const secs = Math.max(0.1, (performance.now() - t0) / 1000);
        status.textContent = done > 0
          ? `${fmtBytes(loaded)} / ${fmtBytes(done)} (${pct.toFixed(0)}%) · ${fmtSpeed(loaded / secs)}`
          : `${fmtBytes(loaded)} sent…`;
      });
      toast(`${files.length} file(s) added to "${p.name}"`, "ok");
      input.value = "";
      await refreshProject(p.id);
    } catch (e) {
      toast(`Upload failed: ${e.message}`, "err");
    } finally {
      btn.disabled = false;
      barWrap.style.display = "none";
      bar.style.width = "0%";
      status.textContent = "";
    }
  });
  wrap.append(el("div", { class: "row" }, input, btn), barWrap, status);
  return wrap;
}

// ----- Project detail -----

views.project = function renderProjectDetail() {
  const p = state.currentProject;
  if (!p) return el("div", { class: "card muted" }, "Project not found.");
  const st = state.projectStatus || {};
  const v = el("div");

  v.append(el("div", { class: "card" },
    el("div", { class: "row-between" },
      el("h2", {}, p.name),
      el("div", { class: "row" },
        statusDot(st.status || p.status),
        el("span", { class: stageClass(stageState(st)) }, st.status || p.status || "idle"),
      ),
    ),
    p.description ? el("p", { class: "muted" }, p.description) : null,
    buildHero({
      status: st.status || p.status || "idle",
      stage: st.current_stage || p.current_stage || null,
      pct: st.progress ?? p.progress ?? 0,
      pctKey: p.id,
      sub: `${p.started_at ? "Started " + fmtTime(p.started_at) : "Not started"} · Elapsed ${elapsed(st.elapsed_seconds ?? p.elapsed_seconds)}`,
    }),
  ));

  // Inputs
  const inputs = (p.inputs || []).map((f) => f.name || f);
  const sourceBox = el("div", { id: "source-check-result", class: "muted mt" }, "");
  const advisorBox = el("div", { id: "advisor-result", class: "mt" }, "");
  const urlInput = el("input", { type: "url", placeholder: "https://...mp4 or YouTube link", style: "flex:1" });
  // Free online hunt (Openverse + Wikimedia Commons, no key): search by
  // matatu name, tick the photos that show the same vehicle, import them.
  const huntInput = el("input", { type: "text", placeholder: "Search free photos: e.g. Syndicate matatu", style: "flex:1" });
  const huntResults = el("div", { id: "hunt-results", class: "mt" });
  const huntGrid = el("div", { class: "hunt-grid" });
  huntResults.append(huntGrid);
  async function runHunt() {
    const q = (huntInput.value || "").trim();
    if (q.length < 2) return toast("Type at least 2 characters to search", "warn");
    huntGrid.replaceChildren(el("div", { class: "muted" }, "Searching free sources..."));
    let data;
    try { data = await API.searchImages(q); }
    catch (e) { huntGrid.replaceChildren(el("div", { class: "muted" }, `Search failed: ${e.message}`)); return; }
    huntGrid.replaceChildren();
    if (!data.results || !data.results.length) {
      huntGrid.append(el("div", { class: "muted" }, `No free photos found for "${data.query}". Try another name.`));
      return;
    }
    const picked = new Set();
    const imgBtn = el("button", { class: "btn btn-primary mt" }, "Import Selected (0)");
    for (const r of data.results) {
      const img = el("img", { src: r.thumbnail_url || r.image_url, alt: r.title, loading: "lazy" });
      const box = el("input", { type: "checkbox" });
      box.addEventListener("change", () => {
        if (box.checked) picked.add(r.image_url);
        else picked.delete(r.image_url);
        imgBtn.textContent = `Import Selected (${picked.size})`;
      });
      const card = el("label", { class: "hunt-card" }, box, img,
        el("div", { class: "hunt-title" }, r.title),
        el("div", { class: "muted" }, `${r.source} · ${r.license}`));
      huntGrid.append(card);
    }
    imgBtn.addEventListener("click", async () => {
      if (!picked.size) return toast("Tick at least one photo first", "warn");
      try {
        await API.importSources(p.id, [...picked]);
        toast(`${picked.size} photo(s) imported`, "ok");
        refreshProject(p.id);
      } catch (e) { toast(e.message, "err"); }
    });
    huntGrid.append(imgBtn);
  }
  v.append(el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Inputs"),
    inputs.length ? el("div", { class: "file-tree" }, inputs.map((n) => el("div", { class: "file" }, n)))
      : el("div", { class: "muted" }, "No input files."),
    el("div", { class: "row mt" },
      urlInput,
      el("button", {
        class: "btn",
        onclick: async () => {
          const url = (urlInput.value || "").trim();
          if (!url) return toast("Paste an online video/image URL first", "warn");
          try {
            await API.importSources(p.id, [url]);
            toast("Online source imported", "ok");
            urlInput.value = "";
            refreshProject(p.id);
          } catch (e) { toast(e.message, "err"); }
        },
      }, "Import URL"),
      el("button", {
        class: "btn",
        onclick: async () => {
          sourceBox.textContent = "Checking source quality...";
          try {
            const rep = await API.sourceCheck(p.id);
            sourceBox.replaceChildren(sourceCheckView(rep));
          } catch (e) { sourceBox.textContent = `Source check failed: ${e.message}`; }
        },
      }, "Check Source Quality"),
      el("button", {
        class: "btn",
        onclick: async () => {
          advisorBox.textContent = "Advisor thinking (local 1.5B model, up to ~2 min)...";
          try {
            const rep = await API.advisor(p.id);
            advisorBox.replaceChildren(advisorView(rep));
          } catch (e) { advisorBox.textContent = `Advisor failed: ${e.message}`; }
        },
      }, "AI Advisor"),
    ),
    el("div", { class: "row mt" },
      huntInput,
      el("button", { class: "btn", onclick: () => runHunt() }, "Find Photos Online"),
    ),
    huntResults,
    (function forceToggle() {
      const box = el("input", {
        type: "checkbox",
        onchange: async (ev) => {
          try {
            await API.saveOptions(p.id, { force_weak_source: ev.target.checked });
            toast(ev.target.checked
              ? "Weak-source runs allowed - retry processing to run anyway"
              : "Source gate re-enabled", ev.target.checked ? "warn" : "ok");
            refreshProject(p.id);
          } catch (e) { toast(e.message, "err"); refreshProject(p.id); }
        },
      });
      box.checked = !!((p.options || {}).force_weak_source);
      return el("label", { class: "check mt" }, box,
        " Run even if the source check fails (weak video may produce a partial model)");
    })(),
    sourceBox,
    advisorBox,
  ));

  // Pipeline
  const detailBox = el("div");
  const pipelineCard = el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Processing Pipeline"),
    pipelineList(st.stages, (stage) => {
      detailBox.replaceChildren(stageDetail(stage) || el("div"));
    }),
  );
  v.append(pipelineCard, detailBox);

  // Controls
  const isRunning = ["running", "processing", "queued"].includes(st.status || p.status);
  v.append(el("div", { class: "row mt" },
    el("button", {
      class: "btn btn-primary",
      disabled: isRunning,
      onclick: async () => {
        try { await API.startProcessing(p.id); toast("Processing started", "ok"); refreshProject(p.id); }
        catch (e) { toast(e.message, "err"); }
      },
    }, "Start Processing"),
    el("button", {
      class: "btn",
      disabled: !isRunning,
      onclick: async () => {
        try { await API.pauseProcessing(p.id); toast("Paused", "warn"); refreshProject(p.id); }
        catch (e) { toast(e.message, "err"); }
      },
    }, "Pause"),
    el("button", {
      class: "btn btn-danger",
      onclick: async () => {
        if (!confirm("Cancel processing for this project?")) return;
        try { await API.cancelProcessing(p.id); toast("Cancelled", "warn"); refreshProject(p.id); }
        catch (e) { toast(e.message, "err"); }
      },
    }, "Cancel"),
    el("button", { class: "btn", onclick: () => (location.hash = `#/logs?project=${p.id}`) }, "View Logs"),
  ));

  // Error display
  if (st.error) {
    v.append(el("div", { class: "error-box mt" },
      el("h3", {}, (st.error.stage || "Stage") + " failed"),
      el("div", { class: "kv" }, el("span", {}, "Stage"), el("span", {}, st.error.stage || "--")),
      el("div", { class: "kv" }, el("span", {}, "Exit code"), el("span", {}, String(st.error.exit_code ?? "--"))),
      el("div", { class: "kv" }, el("span", {}, "Possible cause"), el("span", {}, st.error.cause || "See full log.")),
      el("div", { class: "row mt" },
        el("button", { class: "btn", onclick: () => (location.hash = `#/logs?project=${p.id}`) }, "View full log"),
        el("button", { class: "btn", onclick: () => (location.hash = "#/settings") }, "Change Settings"),
        el("button", { class: "btn", onclick: () => (location.hash = "#/projects") }, "Return to Projects"),
      ),
    ));
  }

  // Files
  const filesCard = el("div", { class: "card mt" }, el("div", { class: "panel-label" }, "Project Files"));
  const tree = st.files || p.files;
  if (tree && tree.length) {
    for (const group of tree) {
      filesCard.append(el("div", { class: "file-tree" },
        el("div", { class: "dir" }, group.name),
        (group.files || []).map((f) => {
          const name = f.name || f;
          const row = el("div", { class: "file-row" },
            el("span", { class: "file" }, name),
            el("span", { class: "row" },
              el("a", { class: "btn btn-ghost", href: API.downloadFile(p.id, f.path || name), download: "" }, "Download"),
              el("button", {
                class: "btn btn-ghost",
                onclick: async () => {
                  const isInput = group.name === "INPUT";
                  if (isInput && !confirm(`Delete original input file "${name}"? This cannot be undone.`)) return;
                  try { await API.deleteFile(p.id, f.path || name); toast("Deleted", "warn"); refreshProject(p.id); }
                  catch (e) { toast(e.message, "err"); }
                },
              }, "Delete"),
            ),
          );
          return row;
        }),
      ));
    }
  } else {
    filesCard.append(el("div", { class: "muted" }, "No files yet."));
  }
  filesCard.append(renderAddFiles(p));
  v.append(filesCard);

  return v;
};

// ----- Processing overview -----

views.processing = function renderProcessing() {
  const v = el("div");
  const active = state.projects.filter((p) => ["queued", "processing", "running", "paused"].includes(p.status));
  if (active.length === 0) {
    v.append(el("div", { class: "card muted" }, "No active processing jobs."));
    return v;
  }
  for (const p of active) {
    v.append(el("div", { class: "card" + (v.childNodes.length ? " mt" : "") },
      el("div", { class: "row-between" },
        el("h3", {}, p.name),
        el("span", { class: stageClass(stageState(p)) }, p.status),
      ),
      buildHero({
        status: p.status,
        stage: p.current_stage,
        pct: p.progress,
        pctKey: p.id,
        compact: true,
        sub: p.started_at ? `Started ${fmtTime(p.started_at)} · Elapsed ${elapsed(p.elapsed_seconds)}` : null,
      }),
      el("div", { class: "mt" }, el("button", { class: "btn", onclick: () => openProject(p.id) }, "Open Project")),
    ));
  }
  return v;
};

// ----- 3D Preview -----

views.preview = function renderPreview() {
  const v = el("div");
  const info = el("div", { class: "card" }, el("div", { class: "panel-label" }, "Model Information"),
    el("div", { id: "model-info", class: "muted" }, "No model loaded."));
  v.append(
    el("div", { class: "card viewer-wrap" },
      el("canvas", { id: "viewer-canvas" }),
      el("div", { class: "viewer-placeholder", id: "viewer-placeholder" }, "No model loaded. Open a project with a model to preview it."),
      el("div", { class: "viewer-hint" }, "Drag to rotate - scroll to zoom - right-drag to pan"),
    ),
    el("div", { class: "row mt" },
      el("button", { class: "btn", onclick: () => Viewer.setMode("solid") }, "Solid"),
      el("button", { class: "btn", onclick: () => Viewer.setMode("wireframe") }, "Wireframe"),
      el("button", { class: "btn", onclick: () => Viewer.setMode("material") }, "Materials"),
      el("button", { class: "btn", onclick: () => Viewer.reset() }, "Reset View"),
      (function modelPicker() {
        const done = (state.projects || []).filter(
          (x) => x.status === "done" || x.status === "completed");
        const sel = el("select", { id: "viewer-project-select", class: "btn" },
          ...done.map((x) => el("option", { value: String(x.id) },
            `${x.name} (${x.status})`)));
        if (state.currentProject && done.some((x) => x.id === state.currentProject.id)) {
          sel.value = String(state.currentProject.id);
        }
        const btn = el("button", { class: "btn btn-primary" }, "Load Model");
        btn.disabled = !done.length;
        btn.addEventListener("click", async () => {
            const id = sel.value;
            if (!id) return toast("No completed project to load", "warn");
            const target = `#/preview?project=${id}`;
            // A hash change re-renders the view (fresh canvas) and the
            // preview auto-loads the project model itself; only load
            // directly when the hash is unchanged (no re-render happens).
            if (location.hash === target) {
              await loadProject(id);
              await Viewer.loadFromCurrentProject();
            } else {
              location.hash = target;
            }
          });
        if (!done.length) {
          sel.replaceChildren(el("option", { value: "" }, "No completed projects yet"));
        }
        return el("span", { class: "row", style: "gap:6px" },
          el("span", { class: "muted" }, "Model:"),
          sel, btn);
      })(),
    ),
    info,
  );
  // Init viewer after DOM insert. Live pushes re-render this view and
  // rebuild the canvas: re-show an already-loaded model, otherwise
  // auto-load the open project's model so the page is never a dead
  // placeholder waiting for a button click.
  setTimeout(() => {
    Promise.resolve(Viewer.init(document.getElementById("viewer-canvas")))
      .then(() => {
        const ph = document.getElementById("viewer-placeholder");
        if (Viewer.model) {
          ph?.remove();
          if (Viewer.lastInfo) Viewer.renderInfo(Viewer.lastInfo);
        } else if (state.currentProject) {
          Viewer.loadFromCurrentProject();
        } else if (state.projects && state.projects.length) {
          // Fresh session with no project context: fall back to the most
          // recently updated project so the preview is never a dead
          // placeholder waiting for a click.
          window.VehicleLabFallback = "ran:" + state.projects[0].id;
          loadProject(state.projects[0].id).then(() => Viewer.loadFromCurrentProject());
        }
      })
      .catch(() => { /* init failures are reported in the placeholder */ });
  }, 0);
  return v;
};

// Lightweight Three.js viewer. Loaded lazily from CDN via import map.
const Viewer = {
  inited: false,
  renderer: null, scene: null, camera: null, controls: null, model: null, mode: "material", canvas: null, rafId: null, liveryMap: null, lastInfo: null,

  async init(canvas) {
    if (!canvas || (this.inited && this.canvas === canvas)) return;
    if (this.inited && this.canvas !== canvas && this.renderer) {
      if (this.rafId) cancelAnimationFrame(this.rafId);
      this.renderer.dispose();
      this.rafId = null;
      this.inited = false;
    }
    this.canvas = canvas;
    try {
      const THREE = await import("three");
      const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
      this.THREE = THREE;

      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0a0d11);

      this.camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
      this.camera.position.set(3, 2, 5);

      const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.0);
      const dir = new THREE.DirectionalLight(0xffffff, 1.2);
      dir.position.set(4, 6, 3);
      this.scene.add(hemi, dir);

      const grid = new THREE.GridHelper(20, 20, 0x334455, 0x1c2430);
      this.scene.add(grid);

      this.controls = new OrbitControls(this.camera, canvas);
      this.controls.enableDamping = true;

      // A live push can rebuild the canvas mid-session: put the loaded
      // model straight back into the fresh scene instead of showing an
      // empty grid.
      if (this.model) {
        this.scene.add(this.model);
        this.setMode(this.mode);
      }

      this.inited = true;
      const loop = () => {
        this.rafId = requestAnimationFrame(loop);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      };
      this.rafId = requestAnimationFrame(loop);
      window.addEventListener("resize", () => {
        this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
      });
    } catch (e) {
      const ph = document.getElementById("viewer-placeholder");
      if (ph) ph.textContent = "3D viewer unavailable (Three.js failed to load).";
    }
  },

  clear() {
    if (this.model) {
      this.scene.remove(this.model);
      this.model = null;
    }
  },

  async loadFromCurrentProject() {
    const p = state.currentProject;
    if (!p) return toast("Open a project first", "warn");
    try {
      const info = await API.modelInfo(p.id);
      if (!info || !info.url) return toast("No model available for this project", "warn");
      await this.loadURL(info.url, info.format || "obj");
      this.lastInfo = info;
      this.renderInfo(info);
      document.getElementById("viewer-placeholder")?.remove();
    } catch (e) {
      toast(`Model load failed: ${e.message}`, "err");
    }
  },

  renderInfo(info) {
    const host = document.getElementById("model-info");
    if (!host || !info) return;
    const box = el("div");
    const rows = [
      ["Vertices", info.vertices?.toLocaleString()],
      ["Faces", info.faces?.toLocaleString()],
      ["Materials", info.materials],
      ["Objects", info.objects],
      ["File size", fmtBytes(info.file_size)],
    ];
    for (const [k, val] of rows) box.append(el("div", { class: "kv" }, el("span", {}, k), el("span", {}, val ?? "--")));
    host.replaceChildren(box);
  },

  async loadURL(url, format) {
    if (!this.THREE && this.canvas) await this.init(this.canvas);
    const THREE = this.THREE;
    this.clear();
    const normalized = String(format || "").toLowerCase().replace(/^\./, "");
    let obj;
    if (normalized === "glb" || normalized === "gltf") {
      const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
      const gltf = await new GLTFLoader().loadAsync(url);
      obj = gltf.scene;
    } else if (normalized === "fbx") {
      const { FBXLoader } = await import("three/addons/loaders/FBXLoader.js");
      obj = await new FBXLoader().loadAsync(url);
    } else if (normalized === "stl") {
      const { STLLoader } = await import("three/addons/loaders/STLLoader.js");
      const data = await fetch(url).then((response) => response.arrayBuffer());
      obj = new STLLoader().parse(data);
    } else if (normalized === "ply") {
      const { PLYLoader } = await import("three/addons/loaders/PLYLoader.js");
      const data = await fetch(url).then((response) => response.arrayBuffer());
      obj = new PLYLoader().parse(data);
    } else if (normalized === "obj") {
      const { OBJLoader } = await import("three/addons/loaders/OBJLoader.js");
      obj = await new OBJLoader().loadAsync(url);
      // Pipeline models ship their UV atlas (texture.png) beside the obj.
      // The importer never reads mtllib, so load the atlas directly and
      // let "Materials" mode show the real livery.
      this.liveryMap = null;
      try {
        const texUrl = url.replace(/[^/]*$/, "texture.png");
        const tex = await new THREE.TextureLoader().loadAsync(texUrl);
        tex.colorSpace = THREE.SRGBColorSpace;
        this.liveryMap = tex;
      } catch { /* model without a sibling atlas: keep default material */ }
    } else {
      throw new Error(`Preview format '${normalized || "unknown"}' is not supported`);
    }
    this.model = obj;
    this.scene.add(obj);
    this.setMode(this.mode);
    // Frame the model
    const bbox = new THREE.Box3().setFromObject(obj);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size * 0.8, size * 0.3, size * 0.8));
  },

  setMode(mode) {
    this.mode = mode;
    if (!this.model) return;
    this.model.traverse((child) => {
      if (!child.isMesh) return;
      if (mode === "wireframe") {
        child.material.wireframe = true;
        return;
      }
      child.material.wireframe = false;
      if (mode === "solid") {
        child.material.color?.set?.(0x8fa3c8);
        child.material.map = null;
        child.material.needsUpdate = true;
      } else if (mode === "material" && this.liveryMap) {
        // Restore the project's UV atlas (solid mode detached it).
        child.material.map = this.liveryMap;
        child.material.color?.set?.(0xffffff);
        child.material.needsUpdate = true;
      }
      // material mode without an atlas keeps original materials
    });
    // Software-GL / throttled clients can take a beat to produce the next
    // frame: paint the mode change immediately so toggles never look dead.
    if (this.renderer && this.inited) {
      this.renderer.render(this.scene, this.camera);
    }
  },

  reset() {
    if (!this.model) return;
    const bbox = new THREE.Box3().setFromObject(this.model);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size * 0.8, size * 0.3, size * 0.8));
  },

  // Toggle configured light materials/objects in the preview.
  setLightPreset(name) {
    const presets = {
      "ALL OFF": [],
      "HEADLIGHTS": ["headlight"],
      "BRAKE": ["brake", "tail"],
      "LEFT INDICATOR": ["indicator_l", "indicator_fl", "indicator_rl"],
      "RIGHT INDICATOR": ["indicator_r", "indicator_fr", "indicator_rr"],
      "HAZARD": ["indicator"],
      "REVERSE": ["reverse"],
      "FOG": ["fog"],
    };
    const tokens = presets[name] || [];
    let found = 0;
    if (this.model) {
      this.model.traverse((child) => {
        const matName = (child.material?.name || "").toLowerCase();
        const objName = (child.name || "").toLowerCase();
        const match = tokens.some((t) => matName.includes(t) || objName.includes(t));
        if (child.isMesh && matName.includes("light")) {
          const on = match && name !== "ALL OFF";
          if (child.material.emissive) {
            child.material.emissiveIntensity = on ? 2.0 : 0.0;
            found += on ? 1 : 0;
          }
        }
      });
    }
    toast(found ? `${name}: ${found} light object(s) lit` : `${name} preview applied`, found ? "ok" : "info");
  },
};

// Diagnostic handle for automated smoke tests (module scope is otherwise
// invisible to CDP probes).
window.VehicleLabViewer = Viewer;
window.VehicleLabState = state;

// ----- Lights -----

const LIGHT_CATEGORIES = ["HEADLIGHTS", "TAIL LIGHTS", "BRAKE LIGHTS", "INDICATORS", "REVERSE", "FOG", "DRL"];
const LIGHT_PRESETS = ["ALL OFF", "HEADLIGHTS", "BRAKE", "LEFT INDICATOR", "RIGHT INDICATOR", "HAZARD", "REVERSE", "FOG"];

views.lights = function renderLights() {
  const v = el("div");
  const p = state.currentProject;
  if (!p) {
    v.append(el("div", { class: "card muted" }, "Open a project to inspect its lights."));
    return v;
  }

  // 3D preview area (reuse viewer)
  v.append(el("div", { class: "card viewer-wrap" },
    el("canvas", { id: "viewer-canvas" }),
    el("div", { class: "viewer-hint" }, "Light preview shows configured light materials/objects"),
  ));
  setTimeout(() => Viewer.init(document.getElementById("viewer-canvas")), 0);

  v.append(el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Preview Controls"),
    el("div", { class: "row mt" },
      LIGHT_PRESETS.map((name) => el("button", {
        class: "btn",
        onclick: () => Viewer.setLightPreset?.(name) ?? toast(`Preset "${name}" sent`, "info"),
      }, name)),
    ),
  ));

  // Light list grouped by category
  const listCard = el("div", { class: "card mt" }, el("div", { class: "panel-label" }, "Detected Lights"));
  if (state.lights.length === 0) {
    listCard.append(el("div", { class: "muted" }, "No lights detected yet. Run the pipeline with light analysis enabled."));
  } else {
    for (const cat of LIGHT_CATEGORIES) {
      const items = state.lights.filter((l) => (l.category || "").toUpperCase() === cat);
      if (items.length === 0) continue;
      listCard.append(el("div", { class: "file-tree" }, el("div", { class: "dir" }, cat)));
      for (const light of items) {
        listCard.append(el("div", { class: "file-row" },
          el("span", {},
            el("span", { class: `dot ${light.status === "Configured" ? "dot-on" : "dot-run"}` }), " ",
            el("strong", {}, light.name),
            el("div", { class: "faint" }, `Material: ${light.material || "--"} - Status: ${light.status || "Detected"}`),
          ),
          el("button", { class: "btn", onclick: () => openLightEditor(light) }, "Edit"),
        ));
      }
    }
  }
  v.append(listCard);

  // Reference analysis
  API.lightAnalysis(p.id).then((analysis) => {
    if (!analysis) return;
    const card = el("div", { class: "card mt" }, el("div", { class: "panel-label" }, "Reference Analysis"));
    for (const [k, ok] of Object.entries(analysis.detected || {})) {
      card.append(el("div", {}, ok ? `✓ ${k} detected` : `✕ ${k} not detected`));
    }
    for (const warn of analysis.warnings || []) {
      card.append(el("div", { class: "mono", style: "color: var(--warn)" }, `⚠ ${warn}`));
    }
    card.append(el("div", { class: "mt" }, el("button", { class: "btn", onclick: () => toast("Review mode: select a light below", "info") }, "Review Lights")));
    v.append(card);
  }).catch(() => { /* analysis not available yet */ });

  return v;
};

function openLightEditor(light) {
  const overlay = el("div", { class: "modal-overlay", id: "light-editor-overlay" });
  const form = el("form", {},
    el("h3", {}, light.name),
    ["X", "Y", "Z"].map((axis, i) => el("label", {}, `Position ${axis}`,
      el("input", { type: "number", step: "0.001", name: `pos${i}`, value: light.position?.[i] ?? 0 })),
    ),
    ["X", "Y", "Z"].map((axis, i) => el("label", {}, `Rotation ${axis}`,
      el("input", { type: "number", step: "1", name: `rot${i}`, value: light.rotation?.[i] ?? 0 })),
    ),
    ["X", "Y", "Z"].map((axis, i) => el("label", {}, `Scale ${axis}`,
      el("input", { type: "number", step: "0.1", name: `scl${i}`, value: light.scale?.[i] ?? 1 })),
    ),
    el("label", {}, "Material",
      el("select", { name: "material" },
        ["LightWhite", "LuzAnaranjada", "LightRed", "LightBlue", "LightYellow", "LightGreen"].map((m) =>
          el("option", { value: m, ...(light.material === m ? { selected: "" } : {}) }, m))),
    ),
    el("label", {}, "Object", el("input", { type: "text", name: "object", value: light.object || light.name })),
    el("div", { class: "modal-actions" },
      el("button", { type: "button", class: "btn btn-danger", onclick: async () => {
        if (!confirm(`Delete light ${light.name}?`)) return;
        try { await API.deleteLight(state.currentProject.id, light.name); toast("Light deleted", "warn"); overlay.remove(); refreshLights(); }
        catch (e) { toast(e.message, "err"); }
      } }, "Delete"),
      el("button", { type: "button", class: "btn btn-ghost", onclick: () => overlay.remove() }, "Reset"),
      el("button", { type: "submit", class: "btn btn-primary" }, "Apply"),
    ),
  );
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const updated = {
      ...light,
      position: [0, 1, 2].map((i) => parseFloat(fd.get(`pos${i}`))),
      rotation: [0, 1, 2].map((i) => parseFloat(fd.get(`rot${i}`))),
      scale: [0, 1, 2].map((i) => parseFloat(fd.get(`scl${i}`))),
      material: fd.get("material"),
      object: fd.get("object"),
    };
    try {
      await API.updateLight(state.currentProject.id, updated);
      toast(`${light.name} updated`, "ok");
      overlay.remove();
      refreshLights();
    } catch (e) { toast(e.message, "err"); }
  });
  const modal = el("div", { class: "modal" }, form);
  overlay.append(modal);
  document.body.append(overlay);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
}

async function refreshLights() {
  const p = state.currentProject;
  if (!p) return;
  try { state.lights = await API.lights(p.id) || []; } catch { state.lights = []; }
  if (state.route === "lights") renderRoute();
}

// ----- Exports -----

views.exports = function renderExports() {
  const v = el("div");
  const p = state.currentProject;
  if (!p) {
    v.append(el("div", { class: "card muted" }, "Open a project to build its export."));
    return v;
  }

  const checklistCard = el("div", { class: "card" }, el("div", { class: "panel-label" }, "Export Checklist"));
  const list = el("div", { id: "export-checklist", class: "muted" }, "Checking...");
  checklistCard.append(list);

  const buildCard = el("div", { class: "card mt" },
    el("button", {
      class: "btn btn-primary",
      onclick: async () => {
        try {
          await API.buildExport(p.id);
          toast("Export started", "ok");
          pollExport(p.id);
        } catch (e) { toast(e.message, "err"); }
      },
    }, "Build BUSSID Mod"),
  );

  const progressCard = el("div", { class: "card mt", id: "export-progress", style: "display:none" },
    el("div", { class: "panel-label" }, "Building Export"),
    el("div", { id: "export-bar-wrap" }),
    el("div", { class: "kv mt" }, el("span", {}, "Current operation"), el("span", { id: "export-op" }, "--")),
  );

  const doneCard = el("div", { class: "card mt", id: "export-done", style: "display:none" },
    el("h3", {}, "Export Complete"),
    el("div", { class: "mono", id: "export-file" }, ""),
    el("div", { class: "mt" }, el("a", { class: "btn btn-primary", id: "export-download", download: "" }, "Download")),
  );

  v.append(checklistCard, buildCard, progressCard, doneCard);

  API.exportChecklist(p.id).then((checks) => {
    const box = el("div");
    for (const c of checks || []) {
      box.append(el("div", { class: c.ok ? "st-done" : "st-fail" }, `${c.ok ? "✓" : "✕"} ${c.name}`));
    }
    list.replaceChildren(box);
  }).catch(() => list.replaceChildren(el("div", { class: "muted" }, "Checklist unavailable.")));

  return v;
};

async function pollExport(projectId) {
  const progressCard = document.getElementById("export-progress");
  const doneCard = document.getElementById("export-done");
  if (!progressCard) return;
  progressCard.style.display = "";
  const timer = setInterval(async () => {
    let st;
    try { st = await API.exportStatus(projectId); } catch { return; }
    document.getElementById("export-bar-wrap").replaceChildren(progressBar(st.progress, st.status));
    document.getElementById("export-op").textContent = st.operation || "--";
    if (st.status === "done" || st.status === "completed") {
      clearInterval(timer);
      progressCard.style.display = "none";
      doneCard.style.display = "";
      document.getElementById("export-file").textContent = st.filename || "export.zip";
      const a = document.getElementById("export-download");
      a.href = st.download_url || API.downloadFile(projectId, st.filename || "export.zip");
      a.textContent = `Download ${st.filename || "export.zip"}`;
    } else if (st.status === "failed") {
      clearInterval(timer);
      toast(`Export failed: ${st.error || "unknown error"}`, "err");
    }
  }, 1500);
}

// ----- Logs -----

views.logs = function renderLogs() {
  const v = el("div");
  const panel = logPanel(state.logs);
  v.append(el("div", { class: "card" },
    el("div", { class: "row-between" },
      el("div", { class: "panel-label" }, "Process Log"),
      logActions(state.logs, () => { state.logs = []; renderRoute(); }),
    ),
    panel,
  ));
  return v;
};

// ----- Settings -----

views.settings = function renderSettings() {
  const v = el("div");
  const s = state.settings || {};
  const form = el("form", { id: "settings-form" },
    el("label", {}, "Meshroom executable path", el("input", { type: "text", name: "meshroom_path", value: s.meshroom_path || "" })),
    el("label", {}, "Blender executable path", el("input", { type: "text", name: "blender_path", value: s.blender_path || "" })),
    el("label", {}, "FFmpeg path", el("input", { type: "text", name: "ffmpeg_path", value: s.ffmpeg_path || "" })),
    el("label", {}, "Project directory", el("input", { type: "text", name: "project_dir", value: s.project_dir || "" })),
    el("label", {}, "Output directory", el("input", { type: "text", name: "output_dir", value: s.output_dir || "" })),
    el("label", {}, "Maximum concurrent jobs",
      el("input", { type: "number", name: "max_concurrent_jobs", min: "1", max: "4", value: s.max_concurrent_jobs ?? 1 })),
    el("div", { class: "modal-actions" },
      el("button", { type: "submit", class: "btn btn-primary" }, "Save Settings"),
    ),
  );
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    try {
      await API.saveSettings({
        meshroom_path: fd.get("meshroom_path"),
        blender_path: fd.get("blender_path"),
        ffmpeg_path: fd.get("ffmpeg_path"),
        project_dir: fd.get("project_dir"),
        output_dir: fd.get("output_dir"),
        max_concurrent_jobs: parseInt(fd.get("max_concurrent_jobs"), 10) || 1,
      });
      toast("Settings saved", "ok");
    } catch (e) { toast(e.message, "err"); }
  });
  v.append(el("div", { class: "card" },
    el("div", { class: "panel-label" }, "Processing Settings"),
    el("p", { class: "muted" }, "Single-user server. Keep concurrent jobs at 1 to avoid overloading the machine."),
    form,
  ));
  return v;
};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const ROUTE_TITLES = {
  dashboard: "Dashboard",
  projects: "Projects",
  project: "Project",
  processing: "Processing",
  preview: "3D Preview",
  lights: "Lights",
  exports: "Exports",
  logs: "Logs",
  settings: "Settings",
};

function parseHash() {
  const hash = location.hash.replace(/^#\/?/, "") || "dashboard";
  const [name, query] = hash.split("?");
  const parts = name.split("/");
  const params = new URLSearchParams(query || "");
  return { name: parts[0] || "dashboard", arg: parts[1] || null, params };
}

async function renderRoute() {
  const { name, arg, params } = parseHash();
  state.route = name;

  // Resolve project context
  if (name === "project" && arg) {
    await loadProject(arg);
  } else if (["preview", "lights", "exports"].includes(name) && params.get("project")) {
    await loadProject(params.get("project"));
  } else if (!["preview", "lights", "exports"].includes(name)) {
    state.currentProject = name === "projects" ? state.currentProject : state.currentProject;
  }

  const view = document.getElementById("view");
  view.replaceChildren();
  const render = views[name] || views.dashboard;
  view.append(render());

  // Entrance animation only when the route actually changed: live WS pushes
  // re-render the same route and must not restart the animation.
  if (renderRoute._last !== name) {
    renderRoute._last = name;
    view.classList.remove("route-enter");
    void view.offsetWidth; // restart the CSS animation
    view.classList.add("route-enter");
  }

  document.getElementById("page-title").textContent = ROUTE_TITLES[name] || "Dashboard";
  document.querySelectorAll("#nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === name);
  });
  document.getElementById("sidebar").classList.remove("open");
}

function openProject(id) {
  location.hash = `#/project/${id}`;
}

async function loadProject(id) {
  try {
    state.currentProject = await API.getProject(id);
    await refreshProject(id, false);
    try { state.lights = await API.lights(id) || []; } catch { state.lights = []; }
  } catch (e) {
    state.currentProject = null;
    toast(`Failed to load project: ${e.message}`, "err");
  }
}

async function refreshProject(id, rerender = true) {
  try {
    state.projectStatus = await API.projectStatus(id);
    try { state.logs = await API.projectLogs(id) || []; } catch { /* logs optional */ }
  } catch { state.projectStatus = null; }
  if (rerender && ["project", "logs", "processing", "dashboard"].includes(state.route)) renderRoute();
}

// ---------------------------------------------------------------------------
// Data loading + live updates
// ---------------------------------------------------------------------------

async function loadProjects() {
  try { state.projects = await API.listProjects() || []; }
  catch { state.projects = []; }
}

async function loadSettings() {
  try { state.settings = await API.settings() || {}; }
  catch { state.settings = {}; }
}

async function loadServerStatus() {
  try {
    state.server = await API.serverStatus();
    const online = !!state.server?.online;
    document.getElementById("server-dot").className = `dot ${online ? "dot-on" : "dot-off"}`;
    document.getElementById("server-online-label").textContent = online ? "ONLINE" : "OFFLINE";
    document.getElementById("sb-cpu").textContent = `${Math.round(state.server?.cpu_percent ?? 0)}%`;
    document.getElementById("sb-ram").textContent = state.server
      ? `${(state.server.ram_used_gb ?? 0).toFixed(1)} / ${(state.server.ram_total_gb ?? 0).toFixed(1)} GB` : "--";
    document.getElementById("sb-disk").textContent = state.server
      ? `${(state.server.disk_used_gb ?? 0).toFixed(0)} / ${(state.server.disk_total_gb ?? 0).toFixed(0)} GB` : "--";
    const badge = document.getElementById("backend-badge");
    badge.textContent = online ? "Backend: online" : "Backend: offline";
    badge.className = `badge ${online ? "badge-ok" : "badge-warn"}`;
  } catch {
    document.getElementById("server-dot").className = "dot dot-off";
    document.getElementById("server-online-label").textContent = "OFFLINE";
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.getElementById("new-project-form").addEventListener("submit", submitNewProject);
document.getElementById("modal-close").addEventListener("click", closeNewProjectModal);
document.getElementById("modal-cancel").addEventListener("click", closeNewProjectModal);
document.getElementById("modal-overlay").addEventListener("click", (e) => {
  if (e.target.id === "modal-overlay") closeNewProjectModal();
});
document.getElementById("menu-btn").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

window.addEventListener("hashchange", renderRoute);

Live.on((msg) => {
  // Backend push: { project_id, stage, operation, progress, status, log?, error? }
  if (msg.type === "export_update") return;
  const currentId = state.currentProject && String(state.currentProject.id);
  if (msg.log && (!msg.project_id || !currentId || String(msg.project_id) === currentId)) {
    state.logs.push({ time: msg.time || fmtTime(Date.now() / 1000), message: msg.log, level: msg.level });
  }
  const p = state.projects.find((x) => String(x.id) === String(msg.project_id));
  if (p) {
    if (msg.status) p.status = msg.status;
    if (msg.stage) p.current_stage = msg.stage;
    if (msg.progress != null) p.progress = msg.progress;
  }
  if (currentId && String(msg.project_id) === currentId) {
    state.projectStatus = { ...state.projectStatus, ...msg };
  }
  if (["project", "processing", "dashboard", "logs"].includes(state.route)) renderRoute();
});

Promise.all([loadProjects(), loadSettings()]).then(renderRoute);
loadServerStatus();
setInterval(loadServerStatus, 5000);
Live.connect();
