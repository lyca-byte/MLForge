// MLForge — Model Builder page logic
(function () {
  'use strict';

  document.getElementById('nav-container').innerHTML = MLForge.renderNav('model');

  // ── Guard: need dataset ───────────────────────────────────
  const ds = MLForge.state.dataset;
  if (!ds) {
    document.getElementById('dataset-context').textContent = 'No dataset selected.';
    const alert = document.createElement('div');
    alert.className = 'alert alert-warning';
    alert.innerHTML = '<span class="alert-icon">⚠️</span> Please select a dataset first.';
    document.querySelector('.builder-layout').prepend(alert);
  } else {
    document.getElementById('dataset-context').textContent =
      `Dataset: ${ds.name} · ${ds.task} · Classes: ${ds.classes.join(', ')} · Input: ${ds.input_shape.join('×')}`;
  }

  // ── Layers state ──────────────────────────────────────────
  let layers = [];
  let dragSrcIdx = null;

  // Restore from state
  if (MLForge.state.model?.layers) {
    layers = JSON.parse(JSON.stringify(MLForge.state.model.layers));
  }

  // ── Palette ───────────────────────────────────────────────
  function renderPalette() {
    const palette = document.getElementById('palette');
    palette.innerHTML = Object.entries(MLForge.LAYER_META).map(([type, meta]) => `
      <div class="palette-item" onclick="addLayer('${type}')">
        <span class="pi-dot" style="background:${meta.color}"></span>
        <span class="pi-label">${meta.label}</span>
        <span class="pi-add">+</span>
      </div>
    `).join('');
  }

  // ── Add layer ─────────────────────────────────────────────
  window.addLayer = function(type) {
    const meta = MLForge.LAYER_META[type];
    if (!meta) return;

    const layer = { type, id: Date.now() + Math.random() };
    meta.fields.forEach(f => { layer[f.key] = f.default; });
    layers.push(layer);
    renderLayerList();
    renderArchPreview();
    updateNextButton();
    MLForge.toast(`Added ${meta.label}`, 'success', 1500);
  };

  // ── Remove layer ──────────────────────────────────────────
  window.removeLayer = function(idx) {
    layers.splice(idx, 1);
    renderLayerList();
    renderArchPreview();
    updateNextButton();
  };

  // ── Move layer ────────────────────────────────────────────
  window.moveLayer = function(idx, dir) {
    const to = idx + dir;
    if (to < 0 || to >= layers.length) return;
    [layers[idx], layers[to]] = [layers[to], layers[idx]];
    renderLayerList();
    renderArchPreview();
    updateNextButton();
  };

  // ── Update layer field ────────────────────────────────────
  window.updateLayerField = function(idx, key, value) {
    const field = MLForge.LAYER_META[layers[idx].type]?.fields.find(f => f.key === key);
    if (!field) return;
    layers[idx][key] = field.type === 'number' ? Number(value) : value;
    renderArchPreview();
    updateNextButton();
    // re-render header summary without full re-render
    const summaryEl = document.getElementById(`lh-summary-${idx}`);
    if (summaryEl) summaryEl.textContent = layerSummary(layers[idx]);
  };

  // ── Layer summary string ──────────────────────────────────
  function layerSummary(layer) {
    const t = layer.type;
    if (t === 'Conv2D')            return `${layer.filters} filters · ${layer.kernel_size}×${layer.kernel_size} · ${layer.activation}`;
    if (t === 'MaxPooling2D')      return `pool ${layer.pool_size}×${layer.pool_size}`;
    if (t === 'AveragePooling2D')  return `pool ${layer.pool_size}×${layer.pool_size}`;
    if (t === 'Flatten')           return 'flattens feature maps';
    if (t === 'Dense')             return `${layer.units} units · ${layer.activation}`;
    if (t === 'Dropout')           return `rate ${layer.rate}`;
    if (t === 'BatchNormalization') return 'normalizes activations';
    return '';
  }

  // ── Render layer list ─────────────────────────────────────
  function renderLayerList() {
    const container = document.getElementById('layer-list');
    document.getElementById('layer-count-badge').textContent = `${layers.length} layer${layers.length !== 1 ? 's' : ''}`;

    if (layers.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="es-icon">🧩</div>
          <p>No layers yet. Add layers from the palette on the left to build your model.</p>
        </div>`;
      return;
    }

    container.innerHTML = layers.map((layer, idx) => {
      const meta   = MLForge.LAYER_META[layer.type];
      const fields = meta.fields.map(f => {
        if (f.type === 'select') {
          return `
            <div class="form-group">
              <label class="form-label">${f.label}</label>
              <select class="form-control" onchange="updateLayerField(${idx},'${f.key}',this.value)">
                ${f.options.map(o => `<option value="${o}" ${layer[f.key] === o ? 'selected' : ''}>${o}</option>`).join('')}
              </select>
            </div>`;
        }
        return `
          <div class="form-group">
            <label class="form-label">${f.label}</label>
            <input type="number" class="form-control"
              value="${layer[f.key]}"
              min="${f.min ?? ''}" max="${f.max ?? ''}" step="${f.step ?? 1}"
              onchange="updateLayerField(${idx},'${f.key}',this.value)">
          </div>`;
      }).join('');

      return `
        <div class="layer-card" id="layer-card-${idx}">
          <div class="layer-header" onclick="toggleLayer(${idx})">
            <span class="lh-drag" title="Use ↑↓ buttons to reorder">⠿</span>
            <span class="lh-dot" style="background:${meta.color}"></span>
            <span class="lh-name">${meta.label}</span>
            <span class="lh-summary" id="lh-summary-${idx}">${layerSummary(layer)}</span>
            <span class="lh-caret">›</span>
          </div>
          <div class="layer-body">
            ${fields ? `<div class="form-row">${fields}</div>` : ''}
            <div class="explain-box">${meta.explain}</div>
            <div class="layer-actions">
              <button class="btn btn-secondary btn-sm btn-icon" title="Move up"    onclick="moveLayer(${idx},-1)" ${idx === 0 ? 'disabled' : ''}>↑</button>
              <button class="btn btn-secondary btn-sm btn-icon" title="Move down"  onclick="moveLayer(${idx},+1)" ${idx === layers.length-1 ? 'disabled' : ''}>↓</button>
              <button class="btn btn-danger btn-sm" onclick="removeLayer(${idx})">Remove</button>
            </div>
          </div>
        </div>`;
    }).join('');

    renderValidation();
  }

  window.toggleLayer = function(idx) {
    const card = document.getElementById(`layer-card-${idx}`);
    if (card) card.classList.toggle('expanded');
  };

  // ── Templates ─────────────────────────────────────────────
  const TEMPLATES = {
    simple: [
      { type: 'Conv2D',       filters: 32, kernel_size: 3, activation: 'relu', padding: 'same' },
      { type: 'MaxPooling2D', pool_size: 2 },
      { type: 'Flatten' },
      { type: 'Dense',        units: 64, activation: 'relu' },
      { type: 'Dropout',      rate: 0.25 },
    ],
    deeper: [
      { type: 'Conv2D',           filters: 32,  kernel_size: 3, activation: 'relu', padding: 'same' },
      { type: 'Conv2D',           filters: 64,  kernel_size: 3, activation: 'relu', padding: 'same' },
      { type: 'MaxPooling2D',     pool_size: 2 },
      { type: 'BatchNormalization' },
      { type: 'Conv2D',           filters: 128, kernel_size: 3, activation: 'relu', padding: 'same' },
      { type: 'MaxPooling2D',     pool_size: 2 },
      { type: 'Flatten' },
      { type: 'Dense',            units: 128, activation: 'relu' },
      { type: 'Dropout',          rate: 0.5 },
    ],
    mlp: [
      { type: 'Flatten' },
      { type: 'Dense',   units: 256, activation: 'relu' },
      { type: 'Dropout', rate: 0.3 },
      { type: 'Dense',   units: 128, activation: 'relu' },
      { type: 'Dropout', rate: 0.3 },
    ],
  };

  window.loadTemplate = function(name) {
    const tmpl = TEMPLATES[name];
    if (!tmpl) return;
    layers = tmpl.map(l => ({ ...l, id: Date.now() + Math.random() }));
    renderLayerList();
    renderArchPreview();
    updateNextButton();
    MLForge.toast(`Loaded ${name} template`, 'info');
  };

  window.clearLayers = function() {
    layers = [];
    renderLayerList();
    renderArchPreview();
    updateNextButton();
  };

  // ── Architecture preview ──────────────────────────────────
  function renderArchPreview() {
    const container = document.getElementById('arch-preview');
    if (layers.length === 0) {
      container.innerHTML = `<div class="empty-state"><p>Add layers to see the architecture</p></div>`;
      return;
    }

    const dsInfo = MLForge.state.dataset;
    const inputShape = dsInfo ? `Input: ${dsInfo.input_shape.join('×')}` : 'Input';

    let html = `<div class="arch-summary">
      <div class="arch-node">
        <span class="an-dot" style="background:#57606a"></span>
        <span class="an-name">${inputShape}</span>
      </div>
      <div class="arch-connector">│</div>`;

    layers.forEach((layer, i) => {
      const meta = MLForge.LAYER_META[layer.type];
      html += `
        <div class="arch-node">
          <span class="an-dot" style="background:${meta.color}"></span>
          <span class="an-name">${meta.label}</span>
          <span class="an-params">${layerSummary(layer)}</span>
        </div>
        ${i < layers.length - 1 ? '<div class="arch-connector">│</div>' : ''}`;
    });

    // Output node
    const numClasses = dsInfo ? dsInfo.classes.length : '?';
    const outAct = dsInfo?.task === 'binary' ? 'sigmoid' : 'softmax';
    html += `
      <div class="arch-connector">│</div>
      <div class="arch-node">
        <span class="an-dot" style="background:#22c55e"></span>
        <span class="an-name">Output (auto)</span>
        <span class="an-params">${numClasses} units · ${outAct}</span>
      </div>
    </div>`;

    container.innerHTML = html;
  }

  // ── Validation ────────────────────────────────────────────
  function validateLayers() {
    const errors = [];
    if (layers.length === 0) {
      errors.push('Add at least one layer to build a model.');
      return errors;
    }

    const hasConv   = layers.some(l => l.type === 'Conv2D');
    const hasPool   = layers.some(l => ['MaxPooling2D','AveragePooling2D'].includes(l.type));
    const hasDense  = layers.some(l => l.type === 'Dense');
    const hasFlatten = layers.some(l => l.type === 'Flatten');
    const flatIdx   = layers.findIndex(l => l.type === 'Flatten');
    const firstConv = layers.findIndex(l => l.type === 'Conv2D');
    const firstDense= layers.findIndex(l => l.type === 'Dense');

    // Conv after Flatten
    if (hasFlatten && hasConv) {
      const lastConvIdx = layers.reduce((acc, l, i) => l.type === 'Conv2D' ? i : acc, -1);
      if (lastConvIdx > flatIdx) {
        errors.push('Conv2D layers must appear before Flatten. Conv cannot follow Flatten.');
      }
    }

    // Dense before Flatten (if any Flatten)
    if (hasFlatten && hasDense && firstDense < flatIdx) {
      errors.push('Dense layers should appear after Flatten to operate on a 1D input.');
    }

    // Conv without Flatten before Dense
    if (hasConv && hasDense && !hasFlatten) {
      errors.push('A Flatten layer is required between convolutional and Dense layers.');
    }

    return errors;
  }

  function renderValidation() {
    const errors  = validateLayers();
    const banner  = document.getElementById('validation-msg');
    if (errors.length === 0) {
      banner.style.display = 'none';
    } else {
      banner.style.display = 'block';
      banner.innerHTML = `<div class="alert alert-warning">
        <span class="alert-icon">⚠️</span>
        <div>${errors.map(e => `<div>• ${e}</div>`).join('')}</div>
      </div>`;
    }
  }

  function isModelValid() {
    return layers.length > 0 && validateLayers().length === 0;
  }

  // ── Next button ───────────────────────────────────────────
  function updateNextButton() {
    document.getElementById('next-btn').disabled = !isModelValid() || !ds;
    renderValidation();
  }

  window.goNext = function() {
    if (!isModelValid()) return;

    const training = {
      epochs:           parseInt(document.getElementById('cfg-epochs').value),
      batch_size:       parseInt(document.getElementById('cfg-batch').value),
      learning_rate:    parseFloat(document.getElementById('cfg-lr').value),
      optimizer:        document.getElementById('cfg-optimizer').value,
      validation_split: parseFloat(document.getElementById('cfg-val-split').value),
    };

    // Validate training params
    if (training.epochs < 1 || training.epochs > 50) {
      MLForge.toast('Epochs must be between 1 and 50.', 'error'); return;
    }
    if (training.learning_rate <= 0 || training.learning_rate > 1) {
      MLForge.toast('Learning rate must be between 0.00001 and 1.', 'error'); return;
    }

    // Strip id from layers before saving
    const layersClean = layers.map(({ id, ...l }) => l);

    MLForge.setState({
      model: { layers: layersClean },
      training,
      job: null,
    });

    window.location.href = 'training.html';
  };

  // ── Init ──────────────────────────────────────────────────
  renderPalette();
  renderLayerList();
  renderArchPreview();
  updateNextButton();

  // Restore training config
  const savedTraining = MLForge.state.training;
  if (savedTraining) {
    document.getElementById('cfg-epochs').value      = savedTraining.epochs;
    document.getElementById('cfg-batch').value       = savedTraining.batch_size;
    document.getElementById('cfg-lr').value          = savedTraining.learning_rate;
    document.getElementById('cfg-optimizer').value   = savedTraining.optimizer;
    document.getElementById('cfg-val-split').value   = savedTraining.validation_split;
  }

}());
