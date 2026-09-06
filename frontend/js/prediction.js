// MLForge — Prediction page logic
(function () {
  'use strict';

  document.getElementById('nav-container').innerHTML = MLForge.renderNav('predict');

  const state = MLForge.state;
  let selectedFile = null;

  // ── Guard ─────────────────────────────────────────────────
  if (!state.job?.status || state.job.status !== 'completed') {
    document.getElementById('no-model').style.display = 'flex';
    document.getElementById('pred-content').style.display = 'none';
    return;
  }

  // ── Model context ─────────────────────────────────────────
  const ds = state.dataset;
  const modelCtx = document.getElementById('model-context');
  if (ds) {
    modelCtx.innerHTML = `
      Dataset: <strong>${ds.name}</strong> &nbsp;·&nbsp;
      Task: <strong>${ds.task}</strong> &nbsp;·&nbsp;
      Classes: <strong>${ds.classes.map(c => ds.class_labels?.[c] ?? c).join(', ')}</strong><br>
      Input size: <strong>${ds.input_shape.join('×')}</strong>
    `;
  }

  // ── Drag and drop ─────────────────────────────────────────
  window.handleDragOver = function(e) {
    e.preventDefault();
    document.getElementById('upload-zone').classList.add('drag-over');
  };
  window.handleDragLeave = function() {
    document.getElementById('upload-zone').classList.remove('drag-over');
  };
  window.handleDrop = function(e) {
    e.preventDefault();
    document.getElementById('upload-zone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  };
  window.handleFileSelect = function(e) {
    const file = e.target.files[0];
    if (file) loadFile(file);
  };

  function loadFile(file) {
    const allowed = ['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'image/webp'];
    if (!allowed.includes(file.type)) {
      MLForge.toast('Unsupported file type. Please upload PNG, JPG, BMP, or WebP.', 'error');
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = document.getElementById('preview-img');
      img.src = ev.target.result;
      document.getElementById('img-info').textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
      document.getElementById('preview-section').style.display = 'block';
      document.getElementById('upload-zone').style.display = 'none';
      // hide result until re-predicted
      document.getElementById('result-panel').style.display = 'none';
      document.getElementById('result-placeholder').style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }

  window.clearImage = function() {
    selectedFile = null;
    document.getElementById('preview-img').src = '';
    document.getElementById('preview-section').style.display = 'none';
    document.getElementById('upload-zone').style.display = 'block';
    document.getElementById('result-panel').style.display = 'none';
    document.getElementById('result-placeholder').style.display = 'flex';
    document.getElementById('file-input').value = '';
  };

  // ── Run prediction ────────────────────────────────────────
  window.runPrediction = async function() {
    if (!selectedFile) return;

    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div> Predicting…';

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('job_id', state.job.job_id);

      const result = await MLForge.api('POST', '/api/prediction/predict', formData);
      showResult(result);
      MLForge.toast('Prediction complete!', 'success', 2000);

    } catch (err) {
      MLForge.toast(`Prediction failed: ${err.message}`, 'error');
      document.getElementById('result-placeholder').innerHTML = `
        <div style="text-align:center;color:var(--danger)">
          <div style="font-size:2rem;margin-bottom:0.5rem">❌</div>
          <p class="text-sm">Prediction error: ${err.message}</p>
        </div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = '🔮 Run Prediction';
    }
  };

  // ── Show results ──────────────────────────────────────────
  function showResult(result) {
    document.getElementById('result-placeholder').style.display = 'none';
    document.getElementById('result-panel').style.display = 'block';

    const classLabel = ds?.class_labels?.[result.predicted_class] ?? result.predicted_class;
    const confidence = (result.confidence * 100).toFixed(1);
    const confColor  = result.confidence > 0.8 ? 'var(--success)' : result.confidence > 0.5 ? 'var(--warning)' : 'var(--danger)';

    document.getElementById('pred-result-area').innerHTML = `
      <div class="prediction-result">
        <div class="pred-class">${classLabel}</div>
        <div class="pred-confidence" style="color:${confColor}">${confidence}% confident</div>
        <div class="pred-label">Predicted Class</div>
      </div>
    `;

    // Probability bars
    const probs   = result.probabilities || {};
    const classes = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
    const topClass = classes[0];

    document.getElementById('prob-bars').innerHTML = classes.map(cls => {
      const lbl  = ds?.class_labels?.[cls] ?? cls;
      const pct  = (probs[cls] * 100).toFixed(1);
      const isTop= cls == topClass; // loose compare (string vs number)
      return `
        <div class="prob-bar-row">
          <div class="prob-bar-label">${lbl}</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill ${isTop ? 'top' : ''}" style="width:${pct}%"></div>
          </div>
          <div class="prob-bar-pct">${pct}%</div>
        </div>
      `;
    }).join('');
  }

}());
