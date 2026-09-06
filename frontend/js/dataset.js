// MLForge — Dataset page logic
(function () {
  'use strict';

  // Render nav
  document.getElementById('nav-container').innerHTML = MLForge.renderNav('dataset');

  // ── Dataset registry ──────────────────────────────────────
  const DATASETS = [
    {
      id: 'mnist',
      name: 'MNIST',
      icon: '✏️',
      description: 'Handwritten digit recognition. 70,000 grayscale images of digits 0–9.',
      classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      classLabels: { 0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9' },
      input_shape: [28, 28, 1],
      train_samples: 60000,
      test_samples: 10000,
      badge: 'Supported',
      badgeCls: 'badge-success',
    },
    {
      id: 'fashion_mnist',
      name: 'Fashion-MNIST',
      icon: '👕',
      description: 'Clothing item classification. 70,000 grayscale fashion images. (Coming soon)',
      classes: [0,1,2,3,4,5,6,7,8,9],
      classLabels: {0:'T-shirt',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Boot'},
      input_shape: [28, 28, 1],
      train_samples: 60000,
      test_samples: 10000,
      badge: 'Coming Soon',
      badgeCls: 'badge-muted',
      disabled: true,
    },
    {
      id: 'cifar10',
      name: 'CIFAR-10',
      icon: '🐶',
      description: 'Object recognition in 32×32 colour images across 10 categories. (Coming soon)',
      classes: [0,1,2,3,4,5,6,7,8,9],
      classLabels: {0:'Airplane',1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'Dog',6:'Frog',7:'Horse',8:'Ship',9:'Truck'},
      input_shape: [32, 32, 3],
      train_samples: 50000,
      test_samples: 10000,
      badge: 'Coming Soon',
      badgeCls: 'badge-muted',
      disabled: true,
    },
  ];

  // ── State ─────────────────────────────────────────────────
  let selectedDataset = null;
  let selectedTask    = null;
  let selectedClasses = new Set();

  // Restore previous selection if any
  const prev = MLForge.state.dataset;
  if (prev) {
    selectedDataset = DATASETS.find(d => d.id === prev.name.toLowerCase().replace('-','_').replace(' ','_')) || null;
    if (selectedDataset) selectedDataset = selectedDataset; // already found
    selectedTask    = prev.task || null;
    selectedClasses = new Set(prev.classes || []);
  }

  // ── Render dataset cards ──────────────────────────────────
  function renderDatasets() {
    const grid = document.getElementById('dataset-grid');
    grid.innerHTML = DATASETS.map(ds => `
      <div class="dataset-card ${ds.disabled ? 'disabled' : ''} ${selectedDataset?.id === ds.id ? 'selected' : ''}"
           id="ds-${ds.id}"
           onclick="${ds.disabled ? '' : `selectDataset('${ds.id}')`}">
        <div class="ds-icon">${ds.icon}</div>
        <h3>${ds.name}</h3>
        <p>${ds.description}</p>
        <div class="ds-badge"><span class="badge ${ds.badgeCls}">${ds.badge}</span></div>
      </div>
    `).join('');
  }

  // ── Select dataset ────────────────────────────────────────
  window.selectDataset = function(id) {
    selectedDataset = DATASETS.find(d => d.id === id);
    selectedTask    = null;
    selectedClasses = new Set();

    renderDatasets();

    document.getElementById('task-card').style.display    = 'block';
    document.getElementById('classes-card').style.display = 'none';
    document.getElementById('summary-card').style.display = 'none';

    // Reset task UI
    document.getElementById('task-binary').classList.remove('selected');
    document.getElementById('task-multi').classList.remove('selected');

    updateNextButton();
  };

  // ── Select task ───────────────────────────────────────────
  window.selectTask = function(task) {
    selectedTask    = task;
    selectedClasses = new Set();

    document.getElementById('task-binary').classList.toggle('selected', task === 'binary');
    document.getElementById('task-multi').classList.toggle('selected', task === 'multiclass');

    document.getElementById('classes-card').style.display = 'block';
    document.getElementById('summary-card').style.display = 'none';

    renderClassPicker();
    updateNextButton();
  };

  // ── Class picker ──────────────────────────────────────────
  function renderClassPicker() {
    if (!selectedDataset || !selectedTask) return;

    const instruction = document.getElementById('class-instruction');
    if (selectedTask === 'binary') {
      instruction.textContent = 'Select exactly 2 classes for binary classification.';
    } else {
      instruction.textContent = 'Select 3 or more classes for multiclass classification.';
    }

    const picker = document.getElementById('class-picker');
    picker.innerHTML = selectedDataset.classes.map(c => {
      const label = selectedDataset.classLabels[c] || String(c);
      const isSel = selectedClasses.has(c);
      return `
        <div class="class-chip ${isSel ? 'selected' : ''}" id="chip-${c}" onclick="toggleClass(${c})">
          ${label}
        </div>
      `;
    }).join('');

    updateClassValidation();
  }

  window.toggleClass = function(cls) {
    if (selectedClasses.has(cls)) {
      selectedClasses.delete(cls);
    } else {
      // binary: max 2
      if (selectedTask === 'binary' && selectedClasses.size >= 2) {
        MLForge.toast('Binary classification allows exactly 2 classes. Deselect one first.', 'warning');
        return;
      }
      selectedClasses.add(cls);
    }

    // re-render chips
    selectedDataset.classes.forEach(c => {
      const chip = document.getElementById(`chip-${c}`);
      if (chip) chip.classList.toggle('selected', selectedClasses.has(c));
    });

    document.getElementById('class-count-badge').textContent = `${selectedClasses.size} selected`;
    updateClassValidation();
    renderSummary();
    updateNextButton();
  };

  function updateClassValidation() {
    const el  = document.getElementById('class-validation');
    const msg = document.getElementById('class-validation-msg');
    const n   = selectedClasses.size;

    if (selectedTask === 'binary' && n > 0 && n !== 2) {
      el.style.display = 'flex';
      msg.textContent = `Please select exactly 2 classes. Currently ${n} selected.`;
    } else if (selectedTask === 'multiclass' && n > 0 && n < 3) {
      el.style.display = 'flex';
      msg.textContent = `Please select at least 3 classes. Currently ${n} selected.`;
    } else {
      el.style.display = 'none';
    }
  }

  // ── Dataset summary ───────────────────────────────────────
  function renderSummary() {
    if (!isValid()) return;

    const ds = selectedDataset;
    document.getElementById('summary-card').style.display = 'block';

    const items = [
      { label: 'Dataset',        value: ds.name },
      { label: 'Task',           value: selectedTask === 'binary' ? 'Binary' : 'Multiclass' },
      { label: 'Classes',        value: [...selectedClasses].map(c => ds.classLabels[c]).join(', ') },
      { label: 'Num Classes',    value: selectedClasses.size },
      { label: 'Image Size',     value: `${ds.input_shape[0]}×${ds.input_shape[1]}px` },
      { label: 'Channels',       value: ds.input_shape[2] === 1 ? 'Grayscale' : 'RGB' },
      { label: 'Train Samples',  value: MLForge.fmtNum(ds.train_samples) },
      { label: 'Test Samples',   value: MLForge.fmtNum(ds.test_samples) },
    ];

    document.getElementById('ds-summary-grid').innerHTML = items.map(i => `
      <div class="ds-info-item">
        <div class="di-label">${i.label}</div>
        <div class="di-value">${i.value}</div>
      </div>
    `).join('');

    // Sample placeholder squares for selected classes
    const sampleGrid = document.getElementById('sample-grid');
    const classArr = [...selectedClasses];
    sampleGrid.innerHTML = classArr.flatMap(c => {
      const label = ds.classLabels[c];
      return Array(4).fill(0).map(() => `
        <div class="sample-img-placeholder" title="Class: ${label}">${label}</div>
      `);
    }).join('');
  }

  // ── Validation ────────────────────────────────────────────
  function isValid() {
    if (!selectedDataset || !selectedTask) return false;
    const n = selectedClasses.size;
    if (selectedTask === 'binary'     && n !== 2) return false;
    if (selectedTask === 'multiclass' && n  < 3)  return false;
    return true;
  }

  function updateNextButton() {
    document.getElementById('next-btn').disabled = !isValid();
    document.getElementById('class-count-badge').textContent = `${selectedClasses.size} selected`;
    if (isValid()) renderSummary();
  }

  // ── Navigate next ─────────────────────────────────────────
  window.goNext = function() {
    if (!isValid()) return;
    MLForge.setState({
      dataset: {
        name:          selectedDataset.name,
        id:            selectedDataset.id,
        task:          selectedTask,
        classes:       [...selectedClasses],
        class_labels:  Object.fromEntries([...selectedClasses].map(c => [c, selectedDataset.classLabels[c]])),
        input_shape:   selectedDataset.input_shape,
        train_samples: selectedDataset.train_samples,
        test_samples:  selectedDataset.test_samples,
      },
      // reset downstream when dataset changes
      model:    null,
      training: null,
      job:      null,
    });
    window.location.href = 'model-builder.html';
  };

  // ── Init ──────────────────────────────────────────────────
  renderDatasets();

  if (selectedDataset) {
    document.getElementById('task-card').style.display = 'block';
    if (selectedTask) {
      document.getElementById(`task-${selectedTask === 'binary' ? 'binary' : 'multi'}`).classList.add('selected');
      document.getElementById('classes-card').style.display = 'block';
      renderClassPicker();
      updateNextButton();
    }
  }

}());
