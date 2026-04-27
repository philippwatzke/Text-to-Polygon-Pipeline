(function () {
  const map = L.map('map').setView([48.137, 11.575], 12);
  const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: '&copy; OpenStreetMap contributors',
  });
  const dopLayer = L.tileLayer.wms('https://geoservices.bayern.de/od/wms/dop/v1/dop20', {
    layers: 'by_dop20c',
    format: 'image/png',
    transparent: false,
    version: '1.1.1',
    maxZoom: 22,
    attribution: '&copy; Bayerische Vermessungsverwaltung, CC BY 4.0',
  });
  const baseLayers = {
    'OSM': osmLayer,
    'DOP20': dopLayer,
  };
  osmLayer.addTo(map);
  L.control.layers(baseLayers, null, {position: 'topright'}).addTo(map);

  const drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);
  map.addControl(new L.Control.Draw({
    draw: {
      polygon: false,
      polyline: false,
      circle: false,
      marker: false,
      circlemarker: false,
      rectangle: true,
    },
    edit: { featureGroup: drawnItems },
  }));

  const promptEl = document.getElementById('prompt');
  const presetEl = document.getElementById('preset');
  const statusEl = document.getElementById('status');
  const jobsEl = document.getElementById('jobs');
  const submitEl = document.getElementById('submit');
  const exportEl = document.getElementById('export');
  const scoreFilterEl = document.getElementById('score-filter');
  const scoreValueEl = document.getElementById('score-value');
  const showRejectedEl = document.getElementById('show-rejected');
  const rejectBelowScoreEl = document.getElementById('reject-below-score');
  const reviewStatsEl = document.getElementById('review-stats');
  const addCompareEl = document.getElementById('add-compare');
  const missedEstimateEl = document.getElementById('missed-estimate');
  const saveMissedEstimateEl = document.getElementById('save-missed-estimate');
  const compareListEl = document.getElementById('compare-list');

  let bbox = null;
  let currentJobId = null;
  let polygonLayer = null;
  let nodataLayer = null;
  let currentPolygons = null;
  let comparisonJobIds = loadComparisonJobIds();
  const pendingUpdates = new Map();
  let flushTimer = null;
  let isFlushing = false;
  const MAX_CLIENT_BUFFER_UPDATES = 100;

  map.on(L.Draw.Event.CREATED, (event) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(event.layer);
    const b = event.layer.getBounds();
    bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  });

  function storageKey(jobId) {
    return `job:${jobId}:pending-validations`;
  }

  function snapshotUpdates() {
    return [...pendingUpdates].map(([pid, validation]) => ({pid, validation}));
  }

  function persistPending(jobId) {
    sessionStorage.setItem(storageKey(jobId), JSON.stringify(snapshotUpdates()));
  }

  function clearPending(jobId) {
    sessionStorage.removeItem(storageKey(jobId));
  }

  function hydratePending(jobId) {
    pendingUpdates.clear();
    const raw = sessionStorage.getItem(storageKey(jobId));
    if (!raw) return;
    for (const item of JSON.parse(raw)) {
      pendingUpdates.set(item.pid, item.validation);
    }
  }

  function featureStyle(feature) {
    const accepted = feature.properties.validation !== 'REJECTED';
    const score = Number(feature.properties.score || 0);
    const underFilter = score < Number(scoreFilterEl.value || 0);
    return {
      color: accepted ? '#2563eb' : '#dc2626',
      fillColor: accepted ? '#60a5fa' : '#f87171',
      fillOpacity: underFilter ? 0.08 : 0.28,
      opacity: underFilter ? 0.35 : 1,
      weight: underFilter ? 1 : 2,
    };
  }

  function passesReviewFilter(feature) {
    const score = Number(feature.properties.score || 0);
    const validation = feature.properties.validation || 'ACCEPTED';
    if (score < Number(scoreFilterEl.value || 0)) return false;
    if (!showRejectedEl.checked && validation === 'REJECTED') return false;
    return true;
  }

  function reviewCounts() {
    const features = currentPolygons?.features || [];
    let accepted = 0;
    let rejected = 0;
    let visible = 0;
    let acceptedBelow = 0;
    const threshold = Number(scoreFilterEl.value || 0);
    for (const feature of features) {
      const isRejected = feature.properties.validation === 'REJECTED';
      const score = Number(feature.properties.score || 0);
      if (isRejected) rejected += 1;
      else {
        accepted += 1;
        if (score < threshold) acceptedBelow += 1;
      }
      if (passesReviewFilter(feature)) visible += 1;
    }
    return {total: features.length, accepted, rejected, visible, acceptedBelow};
  }

  function renderReviewStats() {
    scoreValueEl.value = Number(scoreFilterEl.value || 0).toFixed(2);
    const counts = reviewCounts();
    reviewStatsEl.textContent =
      `${counts.visible}/${counts.total} visible | ${counts.accepted} accepted | ${counts.rejected} rejected | ${counts.acceptedBelow} below score`;
  }

  function compareStorageKey() {
    return 'job-comparison-ids';
  }

  function loadComparisonJobIds() {
    try {
      const parsed = JSON.parse(localStorage.getItem(compareStorageKey()) || '[]');
      return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
    } catch (err) {
      return [];
    }
  }

  function saveComparisonJobIds() {
    localStorage.setItem(compareStorageKey(), JSON.stringify(comparisonJobIds));
  }

  function formatPercent(value) {
    return value === null || value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`;
  }

  function formatScore(value) {
    return value === null || value === undefined ? 'n/a' : Number(value).toFixed(3);
  }

  function addMetric(grid, label, value) {
    const labelEl = document.createElement('span');
    labelEl.textContent = label;
    const valueEl = document.createElement('span');
    valueEl.textContent = value;
    grid.append(labelEl, valueEl);
  }

  async function fetchJobSummary(jobId) {
    const res = await fetch(`/jobs/${jobId}/summary`);
    if (!res.ok) return null;
    return res.json();
  }

  async function renderComparison() {
    compareListEl.innerHTML = '';
    if (!comparisonJobIds.length) {
      compareListEl.textContent = 'No jobs selected';
      return;
    }

    const summaries = await Promise.all(comparisonJobIds.map(fetchJobSummary));
    const validIds = [];
    summaries.forEach((summary) => {
      if (!summary) return;
      validIds.push(summary.id);
      const card = document.createElement('article');
      card.className = 'compare-card';

      const header = document.createElement('header');
      const titleWrap = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'compare-title';
      title.textContent = summary.prompt;
      const subtitle = document.createElement('div');
      subtitle.className = 'compare-subtitle';
      subtitle.textContent = `${summary.tile_preset} | ${summary.status} | ${summary.id.slice(0, 8)}`;
      titleWrap.append(title, subtitle);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'x';
      remove.title = 'Remove from comparison';
      remove.onclick = () => {
        comparisonJobIds = comparisonJobIds.filter(id => id !== summary.id);
        saveComparisonJobIds();
        void renderComparison();
      };
      header.append(titleWrap, remove);

      const grid = document.createElement('div');
      grid.className = 'compare-grid';
      addMetric(grid, 'Accepted', `${summary.accepted}/${summary.total}`);
      addMetric(grid, 'Rejected', String(summary.rejected));
      addMetric(grid, 'Precision', formatPercent(summary.precision_review));
      addMetric(grid, 'Recall est.', formatPercent(summary.recall_estimate));
      addMetric(grid, 'Missed est.', summary.missed_estimate ?? 'n/a');
      addMetric(grid, 'Avg score', formatScore(summary.avg_score));

      const buckets = document.createElement('div');
      buckets.className = 'compare-subtitle';
      buckets.textContent =
        `<0.35 ${summary.score_buckets.lt_035} | 0.35-0.50 ${summary.score_buckets.gte_035_lt_05} | 0.50-0.70 ${summary.score_buckets.gte_05_lt_07} | >=0.70 ${summary.score_buckets.gte_07}`;

      card.append(header, grid, buckets);
      compareListEl.appendChild(card);
      if (summary.id === currentJobId) {
        missedEstimateEl.value = summary.missed_estimate ?? '';
      }
    });

    if (validIds.length !== comparisonJobIds.length) {
      comparisonJobIds = validIds;
      saveComparisonJobIds();
    }
  }

  function renderPolygons() {
    if (polygonLayer) polygonLayer.remove();
    if (!currentPolygons) {
      renderReviewStats();
      return;
    }
    polygonLayer = L.geoJSON(currentPolygons, {
      filter: passesReviewFilter,
      style: featureStyle,
      onEachFeature: (feature, layer) => {
        const score = Number(feature.properties.score || 0).toFixed(3);
        const validation = feature.properties.validation || 'ACCEPTED';
        layer.bindPopup(
          `Score: ${score}<br>Status: ${validation}<br>ID: ${feature.properties.id}`
        );
        layer.on('click', () => {
          feature.properties.validation =
            feature.properties.validation === 'ACCEPTED' ? 'REJECTED' : 'ACCEPTED';
          layer.setStyle(featureStyle(feature));
          layer.setPopupContent(
            `Score: ${score}<br>Status: ${feature.properties.validation}<br>ID: ${feature.properties.id}`
          );
          queueValidation(feature.properties.id, feature.properties.validation);
          renderPolygons();
        });
      },
    }).addTo(map);
    renderReviewStats();
  }

  async function refreshJobs() {
    const res = await fetch('/jobs');
    const jobs = await res.json();
    jobsEl.innerHTML = '';
    for (const job of jobs) {
      const el = document.createElement('div');
      el.className = 'job';
      const done = (job.tile_completed || 0) + (job.tile_failed || 0);
      el.textContent = `${job.prompt} - ${job.status} ${done}/${job.tile_total || '?'}`;
      if (job.status === 'FAILED') {
        el.title = [job.error_reason, job.error_message].filter(Boolean).join('\n\n');
        const reason = job.error_reason ? ` (${job.error_reason})` : '';
        el.textContent = `${job.prompt} - FAILED${reason}`;
      }
      el.onclick = () => openJob(job.id);
      jobsEl.appendChild(el);
    }
  }

  async function flushValidations() {
    if (isFlushing || !pendingUpdates.size || !currentJobId) return;
    isFlushing = true;
    const jobId = currentJobId;
    const snapshot = new Map(pendingUpdates);
    const updates = [...snapshot].map(([pid, validation]) => ({pid, validation}));
    let failed = false;
    try {
      const res = await fetch(`/jobs/${jobId}/polygons/validate_bulk`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({updates}),
      });
      if (!res.ok) throw new Error(`server ${res.status}`);
      for (const [pid, sentVal] of snapshot) {
        if (pendingUpdates.get(pid) === sentVal) pendingUpdates.delete(pid);
      }
      pendingUpdates.size ? persistPending(jobId) : clearPending(jobId);
      void renderComparison();
    } catch (err) {
      failed = true;
      persistPending(jobId);
    } finally {
      isFlushing = false;
      if (pendingUpdates.size && currentJobId === jobId) {
        if (flushTimer) clearTimeout(flushTimer);
        flushTimer = setTimeout(() => void flushValidations(), failed ? 3000 : 0);
      }
    }
  }

  function queueValidation(pid, validation) {
    if (!currentJobId) return;
    pendingUpdates.set(pid, validation);
    persistPending(currentJobId);
    if (isFlushing) return;
    if (pendingUpdates.size >= MAX_CLIENT_BUFFER_UPDATES) {
      void flushValidations();
      return;
    }
    if (flushTimer) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => void flushValidations(), 3000);
  }

  function queueValidationBatch(updates) {
    if (!currentJobId || !updates.length) return;
    for (const update of updates) {
      pendingUpdates.set(update.pid, update.validation);
    }
    persistPending(currentJobId);
    if (isFlushing) return;
    if (pendingUpdates.size >= MAX_CLIENT_BUFFER_UPDATES) {
      void flushValidations();
      return;
    }
    if (flushTimer) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => void flushValidations(), 1000);
  }

  function rejectAcceptedBelowScore() {
    if (!currentPolygons) return;
    const threshold = Number(scoreFilterEl.value || 0);
    const updates = [];
    for (const feature of currentPolygons.features) {
      const score = Number(feature.properties.score || 0);
      if (feature.properties.validation !== 'REJECTED' && score < threshold) {
        feature.properties.validation = 'REJECTED';
        updates.push({pid: feature.properties.id, validation: 'REJECTED'});
      }
    }
    if (!updates.length) {
      statusEl.textContent = `No accepted polygons below ${threshold.toFixed(2)}`;
      renderReviewStats();
      return;
    }
    queueValidationBatch(updates);
    renderPolygons();
    statusEl.textContent =
      `Marked ${updates.length} polygons below ${threshold.toFixed(2)} as rejected`;
  }

  async function openJob(jobId) {
    await flushValidations();
    currentJobId = jobId;
    hydratePending(jobId);
    if (polygonLayer) polygonLayer.remove();
    if (nodataLayer) nodataLayer.remove();
    const [polygons, nodata, summary] = await Promise.all([
      fetch(`/jobs/${jobId}/polygons`).then(r => r.json()),
      fetch(`/jobs/${jobId}/nodata`).then(r => r.json()),
      fetchJobSummary(jobId),
    ]);
    currentPolygons = polygons;
    missedEstimateEl.value = summary?.missed_estimate ?? '';
    renderPolygons();
    nodataLayer = L.geoJSON(nodata, {
      style: {color: '#111827', fillOpacity: 0.1, dashArray: '4 4'},
    }).addTo(map);
    void renderComparison();
  }

  scoreFilterEl.addEventListener('input', renderPolygons);
  showRejectedEl.addEventListener('change', renderPolygons);
  rejectBelowScoreEl.addEventListener('click', rejectAcceptedBelowScore);
  addCompareEl.addEventListener('click', () => {
    if (!currentJobId) {
      statusEl.textContent = 'Open a job first';
      return;
    }
    if (!comparisonJobIds.includes(currentJobId)) {
      comparisonJobIds.push(currentJobId);
      saveComparisonJobIds();
    }
    void renderComparison();
  });
  saveMissedEstimateEl.addEventListener('click', async () => {
    if (!currentJobId) {
      statusEl.textContent = 'Open a job first';
      return;
    }
    const raw = missedEstimateEl.value.trim();
    const missed = raw === '' ? null : Number(raw);
    if (missed !== null && (!Number.isInteger(missed) || missed < 0)) {
      statusEl.textContent = 'Missed estimate must be a non-negative integer';
      return;
    }
    const res = await fetch(`/jobs/${currentJobId}/missed_estimate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({missed_estimate: missed}),
    });
    statusEl.textContent = res.ok ? 'Estimate saved' : 'Estimate failed';
    await renderComparison();
  });

  submitEl.onclick = async () => {
    if (!bbox) {
      statusEl.textContent = 'Draw a rectangle first';
      return;
    }
    const res = await fetch('/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: promptEl.value,
        bbox_wgs84: bbox,
        tile_preset: presetEl.value,
      }),
    });
    if (!res.ok) {
      statusEl.textContent = await res.text();
      return;
    }
    const job = await res.json();
    currentJobId = job.id;
    statusEl.textContent = `Queued ${job.id}`;
    await refreshJobs();
  };

  exportEl.onclick = async () => {
    if (!currentJobId) return;
    await flushValidations();
    const res = await fetch(`/jobs/${currentJobId}/export`, {method: 'POST'});
    statusEl.textContent = res.ok ? 'Exported' : 'Export failed';
    await refreshJobs();
  };

  window.addEventListener('pagehide', () => {
    if (!pendingUpdates.size || !currentJobId) return;
    persistPending(currentJobId);
    fetch(`/jobs/${currentJobId}/polygons/validate_bulk`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        updates: snapshotUpdates().slice(0, MAX_CLIENT_BUFFER_UPDATES),
      }),
      keepalive: true,
    }).catch(() => {});
  });

  setInterval(refreshJobs, 3000);
  void refreshJobs();
  void renderComparison();
})();
