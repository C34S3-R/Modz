// Vehicle Lab - frontend control panel for the vehicle processing server.
// The browser is only the interface: all heavy processing happens on the
// backend. This file talks to the REST API and renders state.

// ---------------------------------------------------------------------------
// API layer
// ---------------------------------------------------------------------------

const API = "/api";

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
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
  startProcessing: (id) => api(`/projects/${id}/process`, { method: "POST" }),
  pauseProcessing: (id) => api(`/projects/${id}/pause`, { method: "POST" }),
  cancelProcessing: (id) => api(`/projects/${id}/cancel`, { method: "POST" }),
  projectStatus: (id) => api(`/projects/${id}/status`),
  projectLogs: (id) => api(`/projects/${id}/logs`),
  projectFiles: (id) => api(`/projects/${id}/files`),
  deleteFile: (id, path) => api(`/projects/${id}/files?path=${encodeURIComponent(path)}`, { method: "DELETE" }),
  downloadFile: (id, path) => `${API}/projects/${id}/files/download?path=${encodeURIComponent(path)}`,
  modelInfo: (id) => api(`/projects/${id}/model`),
  lights: (id) => api(`/projects/${id}/lights`),
  updateLight: (id, light) => api(`/projects/${id}/lights`, { method: "PUT", body: JSON.stringify(light) }),
  deleteLight: (id, name) => api(`/projects/${id}/lights/${encodeURIComponent(name)}`, { method: "DELETE" }),
  lightAnalysis: (id) => api(`/projects/${id}/lights/analysis`),
  exportChecklist: (id) => api(`/projects/${id}/export/checklist`),
  buildExport: (id) => api(`/projects/${id}/export`, { method: "POST" }),
  exportStatus: (id) => api(`/projects/${id}/export/status`),
  settings: () => api("/settings"),
  saveSettings: (data) => api("/settings", { method: "PUT", body: JSON.stringify(data) }),
  clearLogs: () => api("/logs", { method: "DELETE" }),
  downloadLogs: () => `${API}/logs/download`,
};

// Real-time updates: WebSocket when available, SSE fallback, polling last.
const Live = {
  ws: null,
  listeners: new Set(),
  connect() {
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onmessage = (ev) => {
        try { this.listeners.forEach((fn) => fn(JSON.parse(ev.data))); } catch { /* bad frame */ }
      };
      ws.onclose = () => setTimeout(() => this.connect(), 3000);
      ws.onerror = () => ws.close();
      this.ws = ws;
    } catch {
      // WebSocket unavailable: fall back to SSE
      const es = new EventSource(`${API}/events`);
      es.onmessage = (ev) => {
        try { this.listeners.forEach((fn) => fn(JSON.parse(ev.data))); } catch { /* bad frame */ }
      };
    }
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
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message, kind = "info") {
  const box = el("div", { class: `toast toast-${kind}` }, message);
  document.getElementById("toasts").append(box);
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

function progressBar(percent) {
  const p = Math.max(0, Math.min(100, percent ?? 0));
  return el("div", { class: "progress" }, el("div", { style: `width:${p}%` }));
}

function statusDot(status) {
  const map = { running: "st-run", done: "st-done", completed: "st-done", failed: "st-fail", error: "st-fail" };
  return el("span", { class: `dot ${map[status] || "st-wait"}` });
}

const PIPELINE_STAGES = [
  { key: "validation", label: "Video Validation" },
  { key: "extraction", label: "Frame Extraction" },
  { key: "filtering", label: "Frame Filtering" },
  { key: "meshroom", label: "Meshroom Reconstruction" },
  { key: "mesh_validation", label: "Mesh Validation" },
  { key: "blender", label: "Blender Processing" },
  { key: "lights", label: "Light Detection" },
  { key: "bussid_prep", label: "BUSSID Preparation" },
  { key: "export", label: "Export" },
];

const STAGE_ICONS = { waiting: "○", running: "●", done: "✓", failed: "✕" };
const STAGE_CLASSES = { waiting: "st-wait", running: "st-run", done: "st-done", failed: "st-fail" };

function stageState(stage) {
  return (stage && stage.status) || "waiting";
}

function pipelineList(stages, onPick) {
  const list = el("ul", { class: "pipeline" });
  const items = stages && stages.length ? stages : PIPELINE_STAGES.map((s) => ({ key: s.key, label: s.label, status: "waiting" }));
  for (const stage of items) {
    const st = stageState(stage);
    const li = el("li", { class: STAGE_CLASSES[st] }, el("span", { class: "pipe-icon" }, STAGE_ICONS[st] || "○"), stage.label);
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
    el("div", { class: "kv" }, el("span", {}, "Status"), el("span", { class: STAGE_CLASSES[st] }, st)),
  );
  for (const [k, v] of Object.entries(stage)) {
    if (["key", "label", "status"].includes(k) || v == null) continue;
    box.append(el("div", { class: "kv" }, el("span", {}, k.replace(/_/g, " ")), el("span", {}, String(v))));
  }
  return box;
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
    ["CPU", s ? `${Math.round(s.cpu_percent ?? 0)}%` : "--"],
    ["RAM", s ? `${(s.ram_used_gb ?? 0).toFixed(1)} / ${(s.ram_total_gb ?? 0).toFixed(1)} GB` : "--"],
    ["Storage", s ? `${(s.disk_used_gb ?? 0).toFixed(0)} / ${(s.disk_total_gb ?? 0).toFixed(0)} GB` : "--"],
    ["Uptime", s ? elapsed(s.uptime_seconds) : "--"],
  ];
  for (const [label, value] of metrics) {
    stats.append(el("div", { class: "card" },
      el("div", { class: "stat-num" }, value),
      el("div", { class: "stat-label" }, label),
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
    el("div", { class: "kv" }, el("span", {}, "Status"), el("span", { class: STAGE_CLASSES[stageState(p)] }, p.status || "idle")),
    el("div", { class: "kv" }, el("span", {}, "Current stage"), el("span", {}, p.current_stage || "--")),
    el("div", { class: "mt" }, progressBar(p.progress)),
    el("div", { class: "row mt" },
      el("span", { class: "faint" }, `Started: ${p.started_at ? fmtTime(p.started_at) : "--"}`),
      el("span", { class: "faint" }, `Elapsed: ${elapsed(p.elapsed_seconds)}`),
    ),
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
      el("td", { class: STAGE_CLASSES[stageState(p)] }, p.status || "idle"),
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
}

async function submitNewProject(event) {
  event.preventDefault();
  const form = event.target;
  const fd = new FormData(form);
  const meta = {
    name: fd.get("name"),
    description: fd.get("description") || "",
    options: {
      photogrammetry: form.photogrammetry.checked,
      lights: form.lights.checked,
      blender: form.blender.checked,
      bussid: form.bussid.checked,
    },
  };
  const files = ["video", "model", "images"]
    .map((k) => (k === "images" ? [...form.images.files] : form[k].files[0]))
    .flat()
    .filter(Boolean);

  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.textContent = "Uploading...";
  try {
    const project = await API.createProject(meta, files);
    toast(`Project "${project.name}" created`, "ok");
    closeNewProjectModal();
    await loadProjects();
    openProject(project.id);
  } catch (e) {
    toast(`Create failed: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Create Project";
  }
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
        el("span", { class: STAGE_CLASSES[stageState(st)] }, st.status || p.status || "idle"),
      ),
    ),
    p.description ? el("p", { class: "muted" }, p.description) : null,
    el("div", { class: "kv" }, el("span", {}, "Stage"), el("span", {}, st.current_stage || p.current_stage || "--")),
    el("div", { class: "kv" }, el("span", {}, "Progress"), el("span", {}, `${Math.round(st.progress ?? p.progress ?? 0)}%`)),
    el("div", { class: "mt" }, progressBar(st.progress ?? p.progress ?? 0)),
  ));

  // Inputs
  const inputs = (p.inputs || []).map((f) => f.name || f);
  v.append(el("div", { class: "card mt" },
    el("div", { class: "panel-label" }, "Inputs"),
    inputs.length ? el("div", { class: "file-tree" }, inputs.map((n) => el("div", { class: "file" }, n)))
      : el("div", { class: "muted" }, "No input files."),
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
        el("span", { class: STAGE_CLASSES[stageState(p)] }, p.status),
      ),
      el("div", { class: "kv" }, el("span", {}, "Stage"), el("span", {}, p.current_stage || "--")),
      el("div", { class: "mt" }, progressBar(p.progress)),
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
      el("button", { class: "btn", onclick: () => Viewer.loadFromCurrentProject() }, "Load Project Model"),
    ),
    info,
  );
  // Init viewer after DOM insert
  setTimeout(() => Viewer.init(document.getElementById("viewer-canvas")), 0);
  return v;
};

// Lightweight Three.js viewer. Loaded lazily from CDN via import map.
const Viewer = {
  inited: false,
  renderer: null, scene: null, camera: null, controls: null, model: null, mode: "solid",

  async init(canvas) {
    if (this.inited || !canvas) return;
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

      this.inited = true;
      const loop = () => {
        requestAnimationFrame(loop);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      };
      loop();
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
      const box = el("div");
      const rows = [
        ["Vertices", info.vertices?.toLocaleString()],
        ["Faces", info.faces?.toLocaleString()],
        ["Materials", info.materials],
        ["Objects", info.objects],
        ["File size", fmtBytes(info.file_size)],
      ];
      for (const [k, val] of rows) box.append(el("div", { class: "kv" }, el("span", {}, k), el("span", {}, val ?? "--")));
      document.getElementById("model-info").replaceChildren(box);
      document.getElementById("viewer-placeholder")?.remove();
    } catch (e) {
      toast(`Model load failed: ${e.message}`, "err");
    }
  },

  async loadURL(url, format) {
    const THREE = this.THREE;
    this.clear();
    let obj;
    if (format === "glb" || format === "gltf") {
      const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
      const gltf = await new GLTFLoader().loadAsync(url);
      obj = gltf.scene;
    } else {
      const { OBJLoader } = await import("three/addons/loaders/OBJLoader.js");
      obj = await new OBJLoader().loadAsync(url);
    }
    this.model = obj;
    this.scene.add(obj);
    this.setMode(this.mode);
    // Frame the model
    const bbox = new THREE.Box3().setFromObject(obj);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size, size * 0.6, size));
  },

  setMode(mode) {
    this.mode = mode;
    if (!this.model) return;
    this.model.traverse((child) => {
      if (!child.isMesh) return;
      if (mode === "wireframe") {
        child.material.wireframe = true;
      } else {
        child.material.wireframe = false;
        if (mode === "solid") {
          child.material.color?.set?.(0x8fa3c8);
          child.material.map = null;
        }
        // material mode keeps original materials
      }
    });
  },

  reset() {
    if (!this.model) return;
    const bbox = new THREE.Box3().setFromObject(this.model);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size, size * 0.6, size));
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
    el("div", { class: "panel-label" }, "Building Mod"),
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
    document.getElementById("export-bar-wrap").replaceChildren(progressBar(st.progress));
    document.getElementById("export-op").textContent = st.operation || "--";
    if (st.status === "done" || st.status === "completed") {
      clearInterval(timer);
      progressCard.style.display = "none";
      doneCard.style.display = "";
      document.getElementById("export-file").textContent = st.filename || "export.bussidmod";
      const a = document.getElementById("export-download");
      a.href = st.download_url || API.downloadFile(projectId, st.filename || "export.bussidmod");
      a.textContent = `Download ${st.filename || "export.bussidmod"}`;
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
  if (msg.log) state.logs.push({ time: msg.time || fmtTime(Date.now() / 1000), message: msg.log, level: msg.level });
  const p = state.projects.find((x) => String(x.id) === String(msg.project_id));
  if (p) {
    if (msg.status) p.status = msg.status;
    if (msg.stage) p.current_stage = msg.stage;
    if (msg.progress != null) p.progress = msg.progress;
  }
  if (state.currentProject && String(state.currentProject.id) === String(msg.project_id)) {
    state.projectStatus = { ...state.projectStatus, ...msg };
  }
  if (["project", "processing", "dashboard", "logs"].includes(state.route)) renderRoute();
});

loadProjects().then(renderRoute);
loadServerStatus();
setInterval(loadServerStatus, 5000);
Live.connect();
