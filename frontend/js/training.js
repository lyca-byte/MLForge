// MLForge — Training page logic
(function () {
  'use strict';

  document.getElementById('nav-container').innerHTML = MLForge.renderNav('training');

  const state    = MLForge.state;
  let jobId      = null;
  let pollTimer  = null;
  let cancelled  = false;

  // ── Chart setup ───────────────────────────────────────────
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
      legend: { labels: { color: '#7c8a9e', font: { size: 11 } } },
      tooltip: { backgroundColor: '#1a1d27', borderColor: '#2e3347', borderWidth: 1 }
    },
    scales: {
      x: { ticks: { color: '#7c8a9e', font: { size: 10 } }, grid: { color: '#2e3347' } },
      y: { ticks: { color: '#7c8a9e', font: { size: 10 } }, grid: { color: '#2e3347' } }
    }
  };

  const accChart = new Chart(document.getElementById('acc-chart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Acc',  data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.3, fill: true, pointRadius: 3 },
        { label: 'Val Acc',    data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.08)', tension: 0.3, fill: true, pointRadius: 3, borderDash: [4,2] },
      ]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 1 } } }
  });

  const lossChart = new Chart(document.getElementById('loss-chart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Loss', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.08)', tension: 0.3, fill: true, pointRadius: 3 },
        { label: 'Val Loss',   data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.06)', tension: 0.3, fill: true, pointRadius: 3, borderDash: [4,2] },
      ]
    },
    options: chartDefaults
  });

  // ── Config display ────────────────────────────────────────
  function showConfigSummary() {
    const ds = state.dataset;
    const m  = state.model;
    const t  = state.training;

    const el = document.getElementById('config-display');
    const err = document.getElementById('config-error');

    if (!ds || !m || !t) {
      el.innerHTML = '';
      err.style.display = 'block';
      document.getElementById('config-error-msg').textContent = 'Dataset or model configuration is missing. Please go back and complete the setup.';
      document.getElementById('train-btn').disabled = true;
      return;
    }
    err.style.display = 'none';

    const items = [
      { label: 'Dataset',      value: ds.name },
      { label: 'Task',         value: ds.task },
      { label: 'Classes',      value: ds.classes.join(', ') },
      { label: 'Layers',       value: m.layers.length },
      { label: 'Epochs',       value: t.epochs },
      { label: 'Batch Size',   value: t.batch_size },
      { label: 'Learning Rate',value: t.learning_rate },
      { label: 'Optimizer',    value: t.optimizer },
    ];

    el.innerHTML = items.map(i => `
      <div class="stat-item">
        <div class="stat-label">${i.label}</div>
        <div class="stat-value" style="font-size:1rem">${i.value}</div>
      </div>`).join('');
  }

  // ── Logging ───────────────────────────────────────────────
  function log(msg, type = '') {
    const area = document.getElementById('log-area');
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    const ts = new Date().toLocaleTimeString();
    line.textContent = `[${ts}] ${msg}`;
    area.appendChild(line);
    area.scrollTop = area.scrollHeight;
  }

  // ── Status helpers ────────────────────────────────────────
  const statusColors = { idle: '#7c8a9e', queued: '#f59e0b', training: '#6366f1', completed: '#22c55e', failed: '#ef4444' };

  function setStatus(status, detail) {
    document.getElementById('status-dot').style.background = statusColors[status] || statusColors.idle;
    document.getElementById('status-label').textContent    = status.charAt(0).toUpperCase() + status.slice(1);
    document.getElementById('status-detail').textContent   = detail || '';
    document.getElementById('status-spinner').classList.toggle('hidden', !['queued','training'].includes(status));
  }

  // ── Chart update ──────────────────────────────────────────
  function updateCharts(history) {
    const epochs = history.accuracy?.length || 0;
    const labels = Array.from({ length: epochs }, (_, i) => String(i + 1));

    accChart.data.labels               = labels;
    accChart.data.datasets[0].data     = history.accuracy || [];
    accChart.data.datasets[1].data     = history.val_accuracy || [];
    accChart.update('none');

    lossChart.data.labels              = labels;
    lossChart.data.datasets[0].data    = history.loss || [];
    lossChart.data.datasets[1].data    = history.val_loss || [];
    lossChart.update('none');
  }

  // ── Live metrics ──────────────────────────────────────────
  function updateLiveMetrics(history) {
    const n = history.accuracy?.length || 0;
    if (n === 0) return;

    const acc    = history.accuracy[n-1];
    const valAcc = history.val_accuracy?.[n-1];
    const loss   = history.loss[n-1];
    const valLoss= history.val_loss?.[n-1];
    const total  = state.training?.epochs || 1;

    document.getElementById('live-acc').textContent     = (acc * 100).toFixed(1) + '%';
    document.getElementById('live-val-acc').textContent = valAcc != null ? (valAcc * 100).toFixed(1) + '%' : '—';
    document.getElementById('live-loss').textContent    = loss.toFixed(4);
    document.getElementById('live-val-loss').textContent= valLoss != null ? valLoss.toFixed(4) : '—';
    document.getElementById('epoch-counter').textContent= `${n} / ${total}`;
    document.getElementById('epoch-bar-fill').style.width = `${(n / total) * 100}%`;
  }

  // ── Polling ───────────────────────────────────────────────
  async function pollJob() {
    if (!jobId || cancelled) return;

    try {
      const data = await MLForge.api('GET', `/api/training/status/${jobId}`);

      setStatus(data.status, data.message || '');

      if (data.history) {
        updateCharts(data.history);
        updateLiveMetrics(data.history);
      }

      if (data.status === 'training' || data.status === 'queued') {
        pollTimer = setTimeout(pollJob, 2000);
      } else if (data.status === 'completed') {
        onTrainingComplete(data);
      } else if (data.status === 'failed') {
        onTrainingFailed(data.message || 'Unknown error.');
      }

    } catch (err) {
      log(`Polling error: ${err.message}`, 'error');
      // Retry a few times before giving up
      if (!cancelled) pollTimer = setTimeout(pollJob, 4000);
    }
  }

  // ── Start training ────────────────────────────────────────
  window.startTraining = async function() {
    const ds = state.dataset;
    const m  = state.model;
    const t  = state.training;

    if (!ds || !m || !t) {
      MLForge.toast('Missing dataset or model configuration.', 'error');
      return;
    }

    cancelled = false;
    document.getElementById('train-btn').disabled  = true;
    document.getElementById('cancel-btn').disabled = false;
    document.getElementById('eval-btn').style.display = 'none';
    document.getElementById('next-btn').style.display = 'none';
    document.getElementById('log-area').innerHTML  = '';

    // Reset charts
    accChart.data.labels = lossChart.data.labels = [];
    accChart.data.datasets.forEach(d => d.data = []);
    lossChart.data.datasets.forEach(d => d.data = []);
    accChart.update(); lossChart.update();

    setStatus('queued', 'Sending request to training server…');
    log('Starting training job…');

    try {
      const payload = {
        dataset:  ds,
        model:    m,
        training: t,
      };

      const res = await MLForge.api('POST', '/api/training/start', payload);
      jobId = res.job_id;
      log(`Job created: ${jobId}`);
      setStatus('queued', `Job ID: ${jobId} — waiting for server…`);
      MLForge.setState({ job: { job_id: jobId, status: 'queued' } });
      pollTimer = setTimeout(pollJob, 1500);

    } catch (err) {
      setStatus('failed', err.message);
      log(`Error: ${err.message}`, 'error');
      document.getElementById('train-btn').disabled  = false;
      document.getElementById('cancel-btn').disabled = true;

      if (err.message.includes('fetch') || err.message.toLowerCase().includes('failed')) {
        MLForge.toast('Cannot reach the training server. Is the backend running?', 'error', 6000);
      } else {
        MLForge.toast(err.message, 'error');
      }
    }
  };

  // ── Cancel training ───────────────────────────────────────
  window.cancelTraining = async function() {
    cancelled = true;
    clearTimeout(pollTimer);
    if (jobId) {
      try { await MLForge.api('POST', `/api/training/cancel/${jobId}`); } catch {}
    }
    setStatus('idle', 'Training cancelled.');
    log('Training cancelled by user.', 'error');
    document.getElementById('train-btn').disabled  = false;
    document.getElementById('cancel-btn').disabled = true;
  };

  // ── On complete ───────────────────────────────────────────
  function onTrainingComplete(data) {
    clearTimeout(pollTimer);
    log('Training complete!', 'success');
    setStatus('completed', `Finished — ${data.history?.accuracy?.length || '?'} epochs`);
    document.getElementById('train-btn').disabled  = false;
    document.getElementById('cancel-btn').disabled = true;
    document.getElementById('eval-btn').style.display = 'inline-flex';
    document.getElementById('next-btn').style.display = 'inline-flex';

    MLForge.setState({
      job: {
        job_id:     jobId,
        status:     'completed',
        history:    data.history,
        metrics:    data.metrics,
        model_path: data.model_path,
      }
    });

    MLForge.toast('Training complete! Proceed to evaluation.', 'success');
  }

  // ── On failed ─────────────────────────────────────────────
  function onTrainingFailed(msg) {
    clearTimeout(pollTimer);
    log(`Training failed: ${msg}`, 'error');
    setStatus('failed', msg);
    document.getElementById('train-btn').disabled  = false;
    document.getElementById('cancel-btn').disabled = true;
    MLForge.toast(`Training failed: ${msg}`, 'error', 6000);
  }

  // ── Restore state from previous run ──────────────────────
  function restoreIfCompleted() {
    const job = state.job;
    if (job?.status === 'completed' && job.history) {
      updateCharts(job.history);
      const n = job.history.accuracy?.length || 0;
      updateLiveMetrics(job.history);
      setStatus('completed', `Previous run: ${n} epochs`);
      document.getElementById('eval-btn').style.display  = 'inline-flex';
      document.getElementById('next-btn').style.display  = 'inline-flex';
      jobId = job.job_id;
      log('Restored results from previous training run.', 'success');
    }
  }

  // ── Init ──────────────────────────────────────────────────
  showConfigSummary();
  restoreIfCompleted();

}());
