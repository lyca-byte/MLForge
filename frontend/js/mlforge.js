// MLForge — shared state management and utilities
// All pages share this module via window.MLForge

window.MLForge = window.MLForge || {};

// ── Persistent state ──────────────────────────────────────────
const STORAGE_KEY = 'mlforge_state_v1';

const defaultState = {
  dataset: null,       // { name, task, classes, info }
  model: null,         // { layers: [...] }
  training: null,      // { epochs, batch_size, learning_rate, optimizer, validation_split }
  job: null,           // { job_id, status, history, metrics, model_path }
};

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...defaultState, ...JSON.parse(raw) } : { ...defaultState };
  } catch { return { ...defaultState }; }
}

function saveState(state) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch {}
}

const state = loadState();

MLForge.state = state;

MLForge.setState = function(patch) {
  Object.assign(state, patch);
  saveState(state);
};

MLForge.resetState = function() {
  Object.assign(state, defaultState);
  saveState(state);
};

// ── API base ──────────────────────────────────────────────────
// MLForge.API_BASE = 'http://localhost:8000';
MLForge.API_BASE =
    window.MLForgeConfig?.API_BASE || 'http://localhost:8000';

MLForge.api = async function(method, path, body, opts = {}) {
  const url = MLForge.API_BASE + path;
  const headers = {};
  let bodyContent;

  if (body instanceof FormData) {
    bodyContent = body;
  } else if (body) {
    headers['Content-Type'] = 'application/json';
    bodyContent = JSON.stringify(body);
  }

  const res = await fetch(url, {
    method,
    headers,
    body: bodyContent,
    signal: opts.signal,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || JSON.stringify(err);
    } catch {}
    throw new Error(detail);
  }

  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  if (opts.blob) return res.blob();
  return res.text();
};

// ── Navigation ────────────────────────────────────────────────
MLForge.pages = [
  { id: 'home',     label: 'Home',     href: 'index.html',         icon: '🏠' },
  { id: 'dataset',  label: 'Dataset',  href: 'dataset.html',       icon: '🗄️' },
  { id: 'model',    label: 'Model',    href: 'model-builder.html', icon: '🧠' },
  { id: 'training', label: 'Train',    href: 'training.html',      icon: '⚡' },
  { id: 'evaluate', label: 'Evaluate', href: 'evaluation.html',    icon: '📊' },
  { id: 'predict',  label: 'Predict',  href: 'prediction.html',    icon: '🔮' },
  { id: 'export',   label: 'Export',   href: 'export.html',        icon: '📦' },
];

MLForge.renderNav = function(activeId) {
  const steps = MLForge.pages.filter(p => p.id !== 'home');
  const order = ['dataset','model','training','evaluate','predict','export'];
  const activeIdx = order.indexOf(activeId);

  const stepsHtml = steps.map((p, i) => {
    const isActive    = p.id === activeId;
    const isCompleted = order.indexOf(p.id) < activeIdx;
    const cls = isActive ? 'active' : (isCompleted ? 'completed' : '');
    const checkmark = isCompleted ? '✓' : String(i + 1);
    return `
      <a href="${p.href}" class="nav-step ${cls}">
        <span class="step-num">${checkmark}</span>
        <span class="label">${p.label}</span>
      </a>
      ${i < steps.length - 1 ? '<span class="nav-sep">›</span>' : ''}
    `;
  }).join('');

  const progressPct = activeIdx < 0 ? 0 : Math.round(((activeIdx) / (order.length - 1)) * 100);

  return `
    <nav class="navbar">
      <a href="index.html" class="navbar-brand">ML<span>Forge</span></a>
      <div class="nav-steps">${stepsHtml}</div>
    </nav>
    <div class="workflow-progress">
      <div class="workflow-progress-fill" style="width:${progressPct}%"></div>
    </div>
  `;
};

// ── Toast notifications ───────────────────────────────────────
MLForge.toast = function(msg, type = 'info', duration = 3500) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = `
      position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
      display:flex;flex-direction:column;gap:0.5rem;max-width:340px;
    `;
    document.body.appendChild(container);
  }

  const colors = { info: '#38bdf8', success: '#22c55e', warning: '#f59e0b', error: '#ef4444' };
  const icons  = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };

  const el = document.createElement('div');
  el.style.cssText = `
    background:#1a1d27;border:1px solid #2e3347;border-left:3px solid ${colors[type]};
    border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#e2e8f0;
    display:flex;gap:0.5rem;align-items:flex-start;
    animation:fadeInUp 0.2s ease;
  `;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  container.appendChild(el);

  // inject keyframe once
  if (!document.getElementById('toast-anim')) {
    const s = document.createElement('style');
    s.id = 'toast-anim';
    s.textContent = '@keyframes fadeInUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}';
    document.head.appendChild(s);
  }

  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, duration);
};

// ── Formatting helpers ────────────────────────────────────────
MLForge.pct = v => (v * 100).toFixed(1) + '%';
MLForge.fmtNum = v => typeof v === 'number' ? v.toLocaleString() : v;

// ── Layer meta (for model builder) ───────────────────────────
MLForge.LAYER_META = {
  Conv2D: {
    label: 'Conv2D',
    color: '#6366f1',
    fields: [
      { key: 'filters',     label: 'Filters',     type: 'number', default: 32,    min: 1,  max: 512 },
      { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3,     min: 1,  max: 7 },
      { key: 'activation',  label: 'Activation',  type: 'select', default: 'relu', options: ['relu','sigmoid','tanh','linear'] },
      { key: 'padding',     label: 'Padding',     type: 'select', default: 'same', options: ['same','valid'] },
    ],
    explain: 'A convolutional layer scans the image with small learned filters to detect edges, textures, and patterns. Each filter produces a "feature map" highlighting where that pattern appears.',
  },
  MaxPooling2D: {
    label: 'MaxPooling2D',
    color: '#22c55e',
    fields: [
      { key: 'pool_size', label: 'Pool Size', type: 'number', default: 2, min: 1, max: 4 },
    ],
    explain: 'Max pooling shrinks the spatial size of feature maps by keeping only the strongest activation in each region. This reduces computation and helps the model generalize.',
  },
  AveragePooling2D: {
    label: 'AveragePooling2D',
    color: '#38bdf8',
    fields: [
      { key: 'pool_size', label: 'Pool Size', type: 'number', default: 2, min: 1, max: 4 },
    ],
    explain: 'Average pooling shrinks feature maps by computing the average instead of the maximum. It retains broader information across the pooling window.',
  },
  Flatten: {
    label: 'Flatten',
    color: '#f59e0b',
    fields: [],
    explain: 'Flatten converts the 2D feature maps from convolutional/pooling layers into a single 1D vector so Dense layers can process them.',
  },
  Dense: {
    label: 'Dense',
    color: '#a855f7',
    fields: [
      { key: 'units',      label: 'Units',      type: 'number', default: 128, min: 1, max: 1024 },
      { key: 'activation', label: 'Activation', type: 'select', default: 'relu', options: ['relu','sigmoid','tanh','softmax','linear'] },
    ],
    explain: 'A fully connected (dense) layer connects every input neuron to every output neuron. It combines learned features to produce classification decisions.',
  },
  Dropout: {
    label: 'Dropout',
    color: '#ef4444',
    fields: [
      { key: 'rate', label: 'Rate', type: 'number', default: 0.25, min: 0.01, max: 0.9, step: 0.05 },
    ],
    explain: 'Dropout randomly deactivates a fraction of neurons during each training step. This prevents the model from memorizing training data (overfitting).',
  },
  BatchNormalization: {
    label: 'BatchNorm',
    color: '#ec4899',
    fields: [],
    explain: 'Batch normalization standardizes layer outputs during training, which can stabilize learning, allow higher learning rates, and reduce sensitivity to initialization.',
  },
};

MLForge.OPTIMIZERS = ['adam', 'sgd', 'rmsprop'];
MLForge.ACTIVATIONS = ['relu', 'sigmoid', 'tanh'];
