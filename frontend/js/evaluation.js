// MLForge — Evaluation page logic
(function () {
  'use strict';

  document.getElementById('nav-container').innerHTML = MLForge.renderNav('evaluate');

  const state = MLForge.state;
  const job   = state.job;

  // if (!job?.status === 'completed' && !job?.metrics) {
  if (job?.status !== 'completed' || !job?.metrics) {
    document.getElementById('no-results').style.display = 'flex';
    document.getElementById('eval-content').style.display = 'none';
    return;
  }

  if (!job?.metrics || !job?.history) {
    document.getElementById('no-results').style.display = 'flex';
    document.getElementById('eval-content').style.display = 'none';
    return;
  }

  // ── Metric cards ──────────────────────────────────────────
  function renderMetrics() {
    const m = job.metrics;
    const items = [
      { label: 'Accuracy',  value: (m.accuracy * 100).toFixed(1) + '%', color: '#818cf8', explain: 'Correct predictions / total samples' },
      { label: 'Precision', value: (m.precision * 100).toFixed(1) + '%', color: '#22c55e', explain: 'True positives / (TP + FP)' },
      { label: 'Recall',    value: (m.recall * 100).toFixed(1) + '%', color: '#38bdf8', explain: 'True positives / (TP + FN)' },
      { label: 'F1-Score',  value: (m.f1 * 100).toFixed(1) + '%', color: '#f59e0b', explain: 'Harmonic mean of precision & recall' },
    ];

    document.getElementById('metrics-grid').innerHTML = items.map(i => `
      <div class="metric-card">
        <div class="mc-label">${i.label}</div>
        <div class="mc-value" style="color:${i.color}">${i.value}</div>
        <div class="mc-explain">${i.explain}</div>
      </div>
    `).join('');
  }

  // ── Training curves ───────────────────────────────────────
  function renderCurves() {
    const h = job.history;
    const n = h.accuracy?.length || 0;
    const labels = Array.from({ length: n }, (_, i) => String(i + 1));

    const chartOpts = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#7c8a9e', font: { size: 11 } } },
        tooltip: { backgroundColor: '#1a1d27', borderColor: '#2e3347', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#7c8a9e', font: { size: 10 } }, grid: { color: '#2e3347' } },
        y: { ticks: { color: '#7c8a9e', font: { size: 10 } }, grid: { color: '#2e3347' } }
      }
    };

    new Chart(document.getElementById('acc-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Train Acc', data: h.accuracy || [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.3, fill: true, pointRadius: 3 },
          { label: 'Val Acc',   data: h.val_accuracy || [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.08)', tension: 0.3, pointRadius: 3, borderDash: [4,2] },
        ]
      },
      options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, min: 0, max: 1 } } }
    });

    new Chart(document.getElementById('loss-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Train Loss', data: h.loss || [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.08)', tension: 0.3, fill: true, pointRadius: 3 },
          { label: 'Val Loss',   data: h.val_loss || [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.06)', tension: 0.3, pointRadius: 3, borderDash: [4,2] },
        ]
      },
      options: chartOpts
    });
  }

  // ── Confusion matrix ──────────────────────────────────────
  function renderConfusionMatrix() {
    const cm = job.metrics.confusion_matrix;
    const classLabels = state.dataset?.class_labels || {};
    const classes = state.dataset?.classes || [];

    if (!cm) {
      document.getElementById('cm-container').innerHTML = '<p class="text-muted text-sm">Confusion matrix not available.</p>';
      return;
    }

    // Find max for colour scaling
    const flat = cm.flat();
    const maxVal = Math.max(...flat);

    const headers = ['', ...classes.map(c => `<th>${classLabels[c] ?? c}</th>`)].join('');
    const rows = cm.map((row, i) => {
      const cells = row.map(v => {
        const pct = maxVal > 0 ? v / maxVal : 0;
        const cls = v === 0 ? 'cm-cell-zero' : pct > 0.6 ? 'cm-cell-high' : pct > 0.2 ? 'cm-cell-med' : 'cm-cell-low';
        return `<td class="${cls}">${v}</td>`;
      }).join('');
      return `<tr><th>${classLabels[classes[i]] ?? classes[i]}</th>${cells}</tr>`;
    }).join('');

    document.getElementById('cm-container').innerHTML = `
      <table class="cm-table">
        <thead>
          <tr>
            <th style="border:none;background:transparent"></th>
            ${classes.map(c => `<th style="color:var(--accent-hover)">${classLabels[c] ?? c}</th>`).join('')}
          </tr>
          <tr><th colspan="${classes.length + 1}" style="padding:0.2rem;font-size:0.65rem;color:var(--muted);text-align:left;border:none">← Predicted →</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="text-xs text-muted mt-1">Rows = True class · Columns = Predicted class</p>
    `;
  }

  // ── Per-class metrics ─────────────────────────────────────
  function renderPerClass() {
    const pcm = job.metrics.per_class;
    if (!pcm) {
      document.getElementById('per-class-container').innerHTML = '<p class="text-muted text-sm">Per-class metrics not available.</p>';
      return;
    }

    const classLabels = state.dataset?.class_labels || {};
    const rows = Object.entries(pcm).map(([cls, m]) => `
      <tr>
        <td>${classLabels[cls] ?? cls}</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td>${(m.f1 * 100).toFixed(1)}%</td>
        <td>${MLForge.fmtNum(m.support)}</td>
      </tr>
    `).join('');

    document.getElementById('per-class-container').innerHTML = `
      <table class="table per-class-table">
        <thead>
          <tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // ── Model info ────────────────────────────────────────────
  function renderModelInfo() {
    const info = job.metrics.model_info;
    if (!info) {
      document.getElementById('model-info-grid').innerHTML = '<p class="text-muted text-sm">Model info not available.</p>';
      return;
    }

    const items = [
      { label: 'Total Layers',    value: info.num_layers },
      { label: 'Total Params',    value: MLForge.fmtNum(info.total_params) },
      { label: 'Trainable',       value: MLForge.fmtNum(info.trainable_params) },
      { label: 'Non-Trainable',   value: MLForge.fmtNum(info.non_trainable_params) },
    ];

    document.getElementById('model-info-grid').innerHTML = items.map(i => `
      <div class="stat-item">
        <div class="stat-label">${i.label}</div>
        <div class="stat-value">${i.value}</div>
      </div>
    `).join('');
  }

  // ── Init ──────────────────────────────────────────────────
  renderMetrics();
  renderCurves();
  renderConfusionMatrix();
  renderPerClass();
  renderModelInfo();

}());
