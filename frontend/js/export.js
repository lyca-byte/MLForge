// MLForge — Export page logic
(function () {
  'use strict';

  document.getElementById('nav-container').innerHTML = MLForge.renderNav('export');

  const state = MLForge.state;

  // ── Guard ─────────────────────────────────────────────────
  if (!state.job?.status || state.job.status !== 'completed') {
    document.getElementById('no-model').style.display = 'flex';
    document.getElementById('export-content').style.display = 'none';
    return;
  }

  // ── Download helper ───────────────────────────────────────
  window.downloadExport = async function(type) {
    const jobId = state.job?.job_id;
    if (!jobId) { MLForge.toast('No trained model found.', 'error'); return; }

    const statusEl = document.getElementById(`status-${type}`);
    if (statusEl) {
      statusEl.innerHTML = '<span style="color:var(--muted)">Preparing…</span>';
    }

    try {
      const blob = await MLForge.api('GET', `/api/export/${type}/${jobId}`, null, { blob: true });

      const ext = type === 'keras' ? '.keras' : type === 'tflite' ? '.tflite' : '.json';
      const filename = `mlforge_${type}_${jobId.slice(0, 8)}${ext}`;

      const url = URL.createObjectURL(blob);
      const a   = document.createElement('a');
      a.href     = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      if (statusEl) {
        statusEl.innerHTML = '<span style="color:var(--success)">✓ Downloaded</span>';
      }
      MLForge.toast(`Downloaded ${filename}`, 'success');

    } catch (err) {
      if (statusEl) {
        statusEl.innerHTML = `<span style="color:var(--danger)">✗ ${err.message}</span>`;
      }
      MLForge.toast(`Export failed: ${err.message}`, 'error');
    }
  };

}());
