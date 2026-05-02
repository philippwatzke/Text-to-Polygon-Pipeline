(function () {
  const map = L.map('map', {zoomControl: false}).setView([48.137, 11.575], 12);

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
    OSM: osmLayer,
    DOP20: dopLayer,
  };
  dopLayer.addTo(map);
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
      rectangle: {
        shapeOptions: {
          color: '#2563eb',
          fillColor: '#2563eb',
          fillOpacity: 0.08,
          weight: 2,
        },
      },
    },
    edit: {featureGroup: drawnItems},
  }));

  const promptEl = document.getElementById('prompt');
  const presetEl = document.getElementById('preset');
  const presetButtonsEl = document.getElementById('preset-buttons');
  const statusEl = document.getElementById('status');
  const jobsEl = document.getElementById('jobs');
  const submitEl = document.getElementById('submit');
  const exportEl = document.getElementById('export');
  const scoreFilterEl = document.getElementById('score-filter');
  const scoreValueEl = document.getElementById('score-value');
  const segmentOpacityEl = document.getElementById('segment-opacity');
  const segmentOpacityValueEl = document.getElementById('segment-opacity-value');
  const showRejectedEl = document.getElementById('show-rejected');
  const rejectBelowScoreEl = document.getElementById('reject-below-score');
  const reviewStatsEl = document.getElementById('review-stats');
  const toggleMissedModeEl = document.getElementById('toggle-missed-mode');
  const missedStatsEl = document.getElementById('missed-stats');
  const addCompareEl = document.getElementById('add-compare');
  const compareListEl = document.getElementById('compare-list');
  const jobSearchEl = document.getElementById('job-search');
  const globalSearchEl = document.getElementById('global-search');
  const showFailedJobsEl = document.getElementById('show-failed-jobs');
  const showExportedJobsEl = document.getElementById('show-exported-jobs');
  const toggleAllJobsEl = document.getElementById('toggle-all-jobs');
  const jobCountEl = document.getElementById('job-count');
  const bboxAreaEl = document.getElementById('bbox-area');
  const simplifyToleranceEl = document.getElementById('simplify-tolerance');
  const orthogonalizeEl = document.getElementById('orthogonalize');
  const presetNameEl = document.getElementById('preset-name');
  const savePresetEl = document.getElementById('save-preset');
  const customPresetListEl = document.getElementById('preset-list');

  const kpiAcceptedEl = document.getElementById('kpi-accepted');
  const kpiRejectedEl = document.getElementById('kpi-rejected');
  const kpiMissedEl = document.getElementById('kpi-missed');
  const openReviewTitleEl = document.getElementById('open-review-title');
  const openReviewStatusEl = document.getElementById('open-review-status');
  const scoreHistogramEl = document.getElementById('score-histogram');
  const thresholdTitleEl = document.getElementById('threshold-title');
  const thresholdCopyEl = document.getElementById('threshold-copy');
  const workerStatusEl = document.getElementById('worker-status');
  const queueSummaryEl = document.getElementById('queue-summary');
  const exportStateEl = document.getElementById('export-state');
  const copyMetadataEl = document.getElementById('copy-metadata');

  const diagnostic = {
    title: document.getElementById('diagnostic-title'),
    subtitle: document.getElementById('diagnostic-subtitle'),
    state: document.getElementById('diagnostic-state'),
    score: document.getElementById('diagnostic-score'),
    ndvi: document.getElementById('diagnostic-ndvi'),
    ndsm: document.getElementById('diagnostic-ndsm'),
  };
  const detail = {
    precision: document.getElementById('error-precision'),
    recall: document.getElementById('error-recall'),
    rejectedShare: document.getElementById('error-rejected-share'),
    lowScore: document.getElementById('error-low-score'),
    failedTiles: document.getElementById('error-failed-tiles'),
    scoreRange: document.getElementById('error-score-range'),
    prompt: document.getElementById('detail-prompt'),
    bbox: document.getElementById('detail-bbox'),
    preset: document.getElementById('detail-preset'),
    vector: document.getElementById('detail-vector'),
    modality: document.getElementById('detail-modality'),
    status: document.getElementById('detail-status'),
    tiles: document.getElementById('detail-tiles'),
    revision: document.getElementById('detail-revision'),
    exportStale: document.getElementById('detail-export-stale'),
    created: document.getElementById('detail-created'),
    runtime: document.getElementById('detail-runtime'),
    model: document.getElementById('detail-model'),
    preprocess: document.getElementById('detail-preprocess'),
    git: document.getElementById('detail-git'),
  };

  let bbox = null;
  let currentJobId = null;
  let currentJobSummary = null;
  let polygonLayer = null;
  let nodataLayer = null;
  let missedLayer = null;
  let currentPolygons = null;
  let currentMissedObjects = null;
  let latestJobs = [];
  let missedMode = false;
  let showAllJobs = false;
  let comparisonJobIds = loadComparisonJobIds();
  const pendingUpdates = new Map();
  let flushTimer = null;
  let isFlushing = false;
  const MAX_CLIENT_BUFFER_UPDATES = 100;

  setTimeout(() => map.invalidateSize(), 0);

  function aoiStyle() {
    return {
      color: '#2563eb',
      fillColor: '#2563eb',
      fillOpacity: 0.08,
      weight: 2,
    };
  }

  function setDrawnBbox(values, {fit = false} = {}) {
    if (!Array.isArray(values) || values.length !== 4) return;
    bbox = values.map(Number);
    bboxAreaEl.textContent = `${bboxAreaKm2(bbox).toFixed(3)} km2`;
    drawnItems.clearLayers();
    const bounds = L.latLngBounds(
      [bbox[1], bbox[0]],
      [bbox[3], bbox[2]],
    );
    L.rectangle(bounds, aoiStyle()).addTo(drawnItems);
    if (fit) map.fitBounds(bounds.pad(0.12), {maxZoom: 18});
  }

  function updateBboxFromDrawnItems() {
    let layer = null;
    drawnItems.eachLayer((item) => {
      if (!layer) layer = item;
    });
    if (!layer?.getBounds) {
      bbox = null;
      bboxAreaEl.textContent = 'not drawn';
      return;
    }
    const b = layer.getBounds();
    bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    bboxAreaEl.textContent = `${bboxAreaKm2(bbox).toFixed(3)} km2`;
  }

  map.on(L.Draw.Event.CREATED, (event) => {
    drawnItems.clearLayers();
    if (event.layer.setStyle) {
      event.layer.setStyle(aoiStyle());
    }
    drawnItems.addLayer(event.layer);
    updateBboxFromDrawnItems();
    statusEl.textContent = 'AOI selected';
  });

  map.on(L.Draw.Event.EDITED, () => {
    updateBboxFromDrawnItems();
    statusEl.textContent = bbox ? 'AOI updated' : 'AOI removed';
  });

  map.on(L.Draw.Event.DELETED, () => {
    updateBboxFromDrawnItems();
    statusEl.textContent = 'AOI removed';
  });

  map.on('click', (event) => {
    if (!missedMode || !currentJobId) return;
    void addMissedObject(event.latlng);
  });

  presetButtonsEl?.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-preset]');
    if (!button) return;
    presetEl.value = button.dataset.preset;
    renderPresetButtons();
  });

  presetEl.addEventListener('change', renderPresetButtons);

  globalSearchEl?.addEventListener('input', () => {
    jobSearchEl.value = globalSearchEl.value;
    renderJobs();
  });

  function bboxAreaKm2(values) {
    if (!values) return 0;
    const [minLon, minLat, maxLon, maxLat] = values.map(Number);
    const meanLat = ((minLat + maxLat) / 2) * Math.PI / 180;
    const widthKm = Math.abs(maxLon - minLon) * 111.32 * Math.cos(meanLat);
    const heightKm = Math.abs(maxLat - minLat) * 110.57;
    return widthKm * heightKm;
  }

  function renderPresetButtons() {
    presetButtonsEl?.querySelectorAll('button[data-preset]').forEach((button) => {
      button.classList.toggle('active', button.dataset.preset === presetEl.value);
    });
  }

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
    const fill = Number(segmentOpacityEl.value || 0.28);
    return {
      color: accepted ? '#2563eb' : '#c24135',
      fillColor: accepted ? '#60a5fa' : '#f07f72',
      fillOpacity: underFilter ? Math.max(0.035, fill * 0.32) : fill,
      opacity: underFilter ? 0.36 : Math.min(1, fill + 0.45),
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
    segmentOpacityValueEl.value = Number(segmentOpacityEl.value || 0.28).toFixed(2);
    const counts = reviewCounts();
    const threshold = Number(scoreFilterEl.value || 0);
    reviewStatsEl.textContent = !currentPolygons
      ? 'Open a job to review detections.'
      : (threshold > 0
        ? `${counts.visible} detections shown after score filter`
        : 'Score filter is off');
    kpiAcceptedEl.textContent = String(counts.accepted);
    kpiRejectedEl.textContent = String(counts.rejected);
    renderScoreHistogram();
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

  function formatDateTime(value) {
    return value ? new Date(value).toLocaleString() : 'n/a';
  }

  function formatRuntime(started, finished) {
    if (!started) return 'n/a';
    const end = finished ? new Date(finished) : new Date();
    const seconds = Math.max(0, Math.round((end - new Date(started)) / 1000));
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const rest = seconds % 60;
    return `${minutes}m ${rest}s`;
  }

  function formatBbox(values) {
    if (!Array.isArray(values) || values.length !== 4) return 'n/a';
    const nums = values.map(Number);
    if (nums.some(value => !Number.isFinite(value))) return 'n/a';
    return `${nums[0].toFixed(5)}, ${nums[1].toFixed(5)} to ${nums[2].toFixed(5)}, ${nums[3].toFixed(5)}`;
  }

  function cleanFilter(filter) {
    if (!filter) return {};
    return Object.fromEntries(
      Object.entries(filter).filter(([, value]) => value !== null && value !== undefined),
    );
  }

  function formatModality(filter) {
    const active = cleanFilter(filter);
    const parts = [];
    if (active.ndvi_min !== undefined) parts.push(`NDVI >= ${active.ndvi_min}`);
    if (active.ndvi_max !== undefined) parts.push(`NDVI <= ${active.ndvi_max}`);
    if (active.ndsm_min !== undefined) parts.push(`nDSM >= ${active.ndsm_min}m`);
    if (active.ndsm_max !== undefined) parts.push(`nDSM <= ${active.ndsm_max}m`);
    return parts.length ? parts.join(', ') : 'none';
  }

  function formatVectorOptions(options) {
    if (!options) return 'none';
    const tolerance = options.simplification_tolerance_m;
    const simplify = tolerance !== null && tolerance !== undefined && tolerance !== ''
      ? `simplify ${tolerance}m`
      : 'no simplification';
    return `${simplify}, ortho ${options.orthogonalize ? 'on' : 'off'}`;
  }

  function applyModalityFilter(filter) {
    setNumberInput('ndvi-min', filter?.ndvi_min);
    setNumberInput('ndvi-max', filter?.ndvi_max);
    setNumberInput('ndsm-min', filter?.ndsm_min);
    setNumberInput('ndsm-max', filter?.ndsm_max);
  }

  function applyVectorOptions(options) {
    if (simplifyToleranceEl) {
      simplifyToleranceEl.value = options?.simplification_tolerance_m ?? '';
    }
    if (orthogonalizeEl) {
      orthogonalizeEl.checked = Boolean(options?.orthogonalize);
    }
  }

  function customPresetStorageKey() {
    return 'polysam:custom-job-presets';
  }

  function loadCustomPresets() {
    try {
      const parsed = JSON.parse(localStorage.getItem(customPresetStorageKey()) || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }

  function saveCustomPresets(presets) {
    localStorage.setItem(customPresetStorageKey(), JSON.stringify(presets));
  }

  function currentPresetPayload(name) {
    return {
      id: window.crypto?.randomUUID ? window.crypto.randomUUID() : String(Date.now()),
      name,
      prompt: promptEl.value.trim(),
      tilePreset: presetEl.value,
      modalityFilter: buildModalityFilter(),
      vectorOptions: buildVectorOptions(),
      createdAt: new Date().toISOString(),
    };
  }

  function presetSubtitle(preset) {
    const parts = [
      preset.tilePreset || 'medium',
      formatModality(preset.modalityFilter),
      formatVectorOptions(preset.vectorOptions),
    ].filter(Boolean);
    return parts.join(' | ');
  }

  function applyCustomPreset(preset) {
    promptEl.value = preset.prompt || '';
    presetEl.value = preset.tilePreset || 'medium';
    renderPresetButtons();
    applyModalityFilter(preset.modalityFilter || {});
    applyVectorOptions(preset.vectorOptions || {});
    statusEl.textContent = `Preset applied: ${preset.name}`;
  }

  function deleteCustomPreset(presetId) {
    const presets = loadCustomPresets().filter(preset => preset.id !== presetId);
    saveCustomPresets(presets);
    renderCustomPresets();
  }

  function renderCustomPresets() {
    if (!customPresetListEl) return;
    const presets = loadCustomPresets();
    customPresetListEl.innerHTML = '';
    if (!presets.length) {
      customPresetListEl.className = 'preset-empty';
      customPresetListEl.textContent = 'No custom presets yet';
      return;
    }
    customPresetListEl.className = 'preset-list';
    presets.forEach((preset) => {
      const card = document.createElement('article');
      card.className = 'preset-card';
      const title = document.createElement('b');
      title.textContent = preset.name || preset.prompt || 'Preset';
      const actions = document.createElement('div');
      actions.className = 'preset-actions';
      const apply = document.createElement('button');
      apply.type = 'button';
      apply.textContent = 'Apply';
      apply.onclick = () => applyCustomPreset(preset);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.onclick = () => deleteCustomPreset(preset.id);
      actions.append(apply, remove);
      const subtitle = document.createElement('span');
      subtitle.textContent = presetSubtitle(preset);
      card.append(title, actions, subtitle);
      customPresetListEl.append(card);
    });
  }

  function saveCurrentPreset() {
    const fallback = promptEl.value.trim() || 'Untitled preset';
    const name = (presetNameEl?.value || fallback).trim();
    const presets = loadCustomPresets();
    presets.unshift(currentPresetPayload(name));
    saveCustomPresets(presets.slice(0, 30));
    if (presetNameEl) presetNameEl.value = '';
    renderCustomPresets();
    statusEl.textContent = `Preset saved: ${name}`;
  }

  function jobName(job) {
    return job?.label || job?.prompt || job?.id || 'job';
  }

  function shortId(id) {
    return String(id || '').slice(0, 8);
  }

  function statusClass(status) {
    if (status === 'READY_FOR_REVIEW' || status === 'EXPORTED') return 'ready';
    if (status === 'FAILED') return 'failed';
    if (status === 'DOWNLOADING' || status === 'INFERRING') return 'warn';
    return '';
  }

  function jobMatchesFilters(job) {
    if (!showFailedJobsEl.checked && job.status === 'FAILED') return false;
    if (!showExportedJobsEl.checked && job.status === 'EXPORTED') return false;
    const q = jobSearchEl.value.trim().toLowerCase();
    if (!q) return true;
    return [
      job.label,
      job.prompt,
      job.status,
      job.id,
      job.error_reason,
    ].filter(Boolean).some(value => String(value).toLowerCase().includes(q));
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) return null;
    return res.json();
  }

  async function fetchJobSummary(jobId) {
    return fetchJson(`/jobs/${jobId}/summary`);
  }

  async function renderComparison() {
    compareListEl.innerHTML = '';
    if (!comparisonJobIds.length) {
      compareListEl.textContent = 'No jobs selected';
      return;
    }

    const summaries = await Promise.all(comparisonJobIds.map(fetchJobSummary));
    const validIds = [];
    const table = document.createElement('table');
    table.className = 'compare-table';
    const head = document.createElement('thead');
    head.innerHTML =
      '<tr><th>Job</th><th>Accepted</th><th>Rejected</th><th>Missed</th><th>Precision</th><th>Recall</th><th>Avg</th><th></th></tr>';
    const body = document.createElement('tbody');
    table.append(head, body);
    summaries.forEach((summary) => {
      if (!summary) return;
      validIds.push(summary.id);
      const row = document.createElement('tr');
      const nameCell = document.createElement('td');
      const title = document.createElement('div');
      title.className = 'compare-title';
      title.textContent = summary.label || summary.prompt;
      const subtitle = document.createElement('div');
      subtitle.className = 'compare-subtitle';
      subtitle.textContent = `${summary.tile_preset} | ${summary.status} | ${shortId(summary.id)}`;
      nameCell.append(title, subtitle);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'x';
      remove.title = 'Remove from comparison';
      remove.className = 'compare-remove';
      remove.onclick = () => {
        comparisonJobIds = comparisonJobIds.filter(id => id !== summary.id);
        saveComparisonJobIds();
        void renderComparison();
      };
      const removeCell = document.createElement('td');
      removeCell.append(remove);
      row.append(
        nameCell,
        cell(`${summary.accepted}/${summary.total}`),
        cell(String(summary.rejected)),
        cell(String(summary.missed_marked || 0)),
        cell(formatPercent(summary.precision_review)),
        cell(formatPercent(summary.recall_estimate)),
        cell(formatScore(summary.avg_score)),
        removeCell,
      );
      body.appendChild(row);
    });
    compareListEl.appendChild(table);

    if (validIds.length !== comparisonJobIds.length) {
      comparisonJobIds = validIds;
      saveComparisonJobIds();
    }
  }

  function cell(text) {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
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
        layer.on('click', () => {
          feature.properties.validation =
            feature.properties.validation === 'ACCEPTED' ? 'REJECTED' : 'ACCEPTED';
          queueValidation(feature.properties.id, feature.properties.validation);
          updateDiagnostic(feature);
          renderPolygons();
        });
        layer.on('mouseover', () => updateDiagnostic(feature));
      },
    }).addTo(map);
    renderReviewStats();
  }

  function updateDiagnostic(feature) {
    const props = feature?.properties || {};
    const score = props.score === undefined ? 'n/a' : Number(props.score).toFixed(3);
    diagnostic.title.textContent = props.id ? 'Polygon selected' : 'No polygon selected';
    diagnostic.subtitle.textContent = currentJobSummary
      ? `${currentJobSummary.prompt} / ${currentJobSummary.tile_preset}${props.id ? ` / id ${props.id}` : ''}`
      : 'Open a review job and click a detection.';
    diagnostic.state.textContent = props.validation || 'idle';
    diagnostic.state.className = `pill ${props.validation ? (props.validation === 'REJECTED' ? 'failed' : 'ready') : ''}`;
    diagnostic.score.textContent = score;
    diagnostic.ndvi.textContent = props.ndvi_mean === undefined ? 'n/a' : Number(props.ndvi_mean).toFixed(3);
    diagnostic.ndsm.textContent = props.ndsm_mean === undefined ? 'n/a' : `${Number(props.ndsm_mean).toFixed(2)} m`;
  }

  function renderMissedStats() {
    const count = currentMissedObjects?.features?.length || 0;
    missedStatsEl.textContent = `${count} marked | ${missedMode ? 'click map to add' : 'mode off'}`;
    kpiMissedEl.textContent = String(count);
    toggleMissedModeEl.classList.toggle('active-mode', missedMode);
    toggleMissedModeEl.textContent = missedMode ? 'Stop marking' : 'Mark misses';
  }

  function renderMissedObjects() {
    if (missedLayer) missedLayer.remove();
    if (!currentMissedObjects) {
      renderMissedStats();
      return;
    }
    missedLayer = L.geoJSON(currentMissedObjects, {
      pointToLayer: (_feature, latlng) => L.circleMarker(latlng, {
        radius: 6,
        color: '#b7791f',
        fillColor: '#f5b642',
        fillOpacity: 0.9,
        weight: 2,
      }),
      onEachFeature: (feature, layer) => {
        layer.bindPopup(`Missed object<br>ID: ${feature.properties.id}<br>Click marker to remove`);
        layer.on('click', (event) => {
          L.DomEvent.stopPropagation(event);
          void removeMissedObject(feature.properties.id);
        });
      },
    }).addTo(map);
    renderMissedStats();
  }

  async function refreshMissedObjects() {
    if (!currentJobId) return;
    currentMissedObjects = await fetchJson(`/jobs/${currentJobId}/missed_objects`);
    renderMissedObjects();
    currentJobSummary = await fetchJobSummary(currentJobId);
    renderJobDetail(currentJobSummary);
    void renderComparison();
  }

  async function addMissedObject(latlng) {
    const res = await fetch(`/jobs/${currentJobId}/missed_objects`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({lon: latlng.lng, lat: latlng.lat}),
    });
    if (!res.ok) {
      statusEl.textContent = 'Missed object save failed';
      return;
    }
    statusEl.textContent = 'Missed object marked';
    await refreshMissedObjects();
  }

  async function removeMissedObject(missedId) {
    if (!currentJobId) return;
    const res = await fetch(`/jobs/${currentJobId}/missed_objects/${missedId}`, {
      method: 'DELETE',
    });
    statusEl.textContent = res.ok ? 'Missed object removed' : 'Remove missed object failed';
    if (res.ok) await refreshMissedObjects();
  }

  async function refreshJobs() {
    const res = await fetch('/jobs');
    if (!res.ok) {
      workerStatusEl.textContent = 'API unavailable';
      return;
    }
    latestJobs = await res.json();
    renderJobs();
  }

  async function refreshHealth() {
    const res = await fetch('/system/health');
    if (!res.ok) {
      workerStatusEl.textContent = 'health unavailable';
      return;
    }
    const health = await res.json();
    const worker = health.worker || {};
    const queue = health.queue || {};
    const activeJob = latestJobs.find(job => job.id === worker.current_job_id);
    const workerLabel = worker.state === 'online'
      ? `${worker.heartbeat_state || 'worker'}${activeJob ? `: ${jobName(activeJob)}` : ''}`
      : worker.state;
    workerStatusEl.textContent = workerLabel || 'offline';

    const pending = Number(queue.pending || 0);
    const running = Number(queue.running || 0);
    const ready = Number(queue.ready_for_review || 0);
    const failed = Number(queue.failed || 0);
    if (running > 0 && activeJob) {
      queueSummaryEl.textContent = `running: ${jobName(activeJob)}`;
    } else if (pending > 0 || running > 0) {
      queueSummaryEl.textContent = `${running} running / ${pending} queued`;
    } else if (ready > 0 || failed > 0) {
      queueSummaryEl.textContent = `${ready} ready / ${failed} failed`;
    } else {
      queueSummaryEl.textContent = 'No active job';
    }
  }

  function renderJobs() {
    const filteredJobs = latestJobs.filter(jobMatchesFilters);
    const jobs = showAllJobs ? filteredJobs : filteredJobs.slice(0, 5);
    jobsEl.innerHTML = '';
    jobCountEl.textContent = `${jobs.length}/${filteredJobs.length}`;
    toggleAllJobsEl.hidden = filteredJobs.length <= 5;
    toggleAllJobsEl.textContent = showAllJobs ? 'Show latest 5' : 'Show all jobs';
    for (const job of jobs) {
      const done = (job.tile_completed || 0) + (job.tile_failed || 0);
      const total = job.tile_total || '?';
      const el = document.createElement('article');
      el.className = 'job';
      if (job.id === currentJobId) el.classList.add('active');
      if (job.status === 'FAILED') {
        el.title = [job.error_reason, job.error_message].filter(Boolean).join('\n\n');
      }

      const top = document.createElement('div');
      top.className = 'job-top';
      const titleWrap = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'job-title';
      title.textContent = jobName(job);
      const meta = document.createElement('div');
      meta.className = 'job-sub';
      meta.textContent = `${shortId(job.id)} / ${job.tile_preset} / ${new Date(job.created_at).toLocaleString()}`;
      titleWrap.append(title, meta);
      const status = document.createElement('span');
      status.className = `pill ${statusClass(job.status)}`;
      status.textContent = job.status === 'FAILED' && job.error_reason ? job.error_reason : job.status;
      top.append(titleWrap, status);

      const grid = document.createElement('div');
      grid.className = 'job-meta-grid';
      grid.append(
        metricCell(`${done}/${total}`, 'tiles'),
        metricCell(String(job.tile_failed || 0), 'failed'),
        metricCell(job.export_stale ? 'stale' : 'fresh', 'export'),
      );

      const progress = jobProgress(job);

      const actions = document.createElement('div');
      actions.className = 'job-actions';
      const note = document.createElement('span');
      note.className = 'hint';
      note.textContent = job.status === 'READY_FOR_REVIEW'
        ? 'Click detections to review.'
        : (job.status === 'FAILED' ? 'Open for error detail.' : 'Progress updates automatically.');
      const buttons = document.createElement('div');
      buttons.className = 'mini-buttons';
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.textContent = 'Copy';
      copy.onclick = (event) => {
        event.stopPropagation();
        void useJobSettings(job.id);
      };
      const rename = document.createElement('button');
      rename.type = 'button';
      rename.textContent = 'Rename';
      rename.onclick = (event) => {
        event.stopPropagation();
        void renameJob(job);
      };
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.onclick = (event) => {
        event.stopPropagation();
        void deleteJob(job);
      };
      buttons.append(copy, rename, remove);
      actions.append(note, buttons);

      el.append(top, grid, progress, actions);
      el.onclick = () => openJob(job.id);
      jobsEl.appendChild(el);
    }
  }

  function jobProgress(job) {
    const done = (job.tile_completed || 0) + (job.tile_failed || 0);
    const total = Number(job.tile_total || 0);
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    const wrap = document.createElement('div');
    wrap.className = 'job-progress';
    const head = document.createElement('div');
    head.className = 'job-progress-head';
    const label = document.createElement('span');
    label.textContent = progressLabel(job.status);
    const value = document.createElement('span');
    value.textContent = total > 0 ? `${pct}%` : 'waiting';
    const track = document.createElement('div');
    track.className = 'job-progress-track';
    const bar = document.createElement('span');
    bar.style.setProperty('--value', `${pct}%`);
    track.append(bar);
    head.append(label, value);
    wrap.append(head, track);
    return wrap;
  }

  function progressLabel(status) {
    if (status === 'PENDING') return 'queued';
    if (status === 'DOWNLOADING') return 'downloading imagery';
    if (status === 'INFERRING') return 'tile inference';
    if (status === 'READY_FOR_REVIEW') return 'ready for review';
    if (status === 'EXPORTED') return 'exported';
    if (status === 'FAILED') return 'failed';
    return 'job progress';
  }

  function metricCell(value, label) {
    const span = document.createElement('span');
    const b = document.createElement('b');
    b.textContent = value;
    span.append(b, document.createTextNode(label));
    return span;
  }

  async function useJobSettings(jobId) {
    const job = await fetchJson(`/jobs/${jobId}`);
    if (!job) {
      statusEl.textContent = 'Job settings could not be loaded';
      return;
    }
    promptEl.value = job.prompt || '';
    presetEl.value = job.tile_preset || 'medium';
    renderPresetButtons();
    applyModalityFilter(job.modality_filter || {});
    applyVectorOptions(job.vector_options || job.run_metadata?.vector_options);
    if (Array.isArray(job.bbox_wgs84) && job.bbox_wgs84.length === 4) {
      setDrawnBbox(job.bbox_wgs84, {fit: true});
    }
    statusEl.textContent = `Settings copied from ${jobName(job)}`;
  }

  function setNumberInput(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value === null || value === undefined ? '' : String(value);
  }

  async function deleteJob(job) {
    const name = jobName(job);
    if (!window.confirm(`Delete job "${name}"? This removes the job, review state, and exported files.`)) {
      return;
    }
    const res = await fetch(`/jobs/${job.id}`, {method: 'DELETE'});
    if (!res.ok) {
      statusEl.textContent = await res.text();
      return;
    }
    if (currentJobId === job.id) {
      currentJobId = null;
      currentJobSummary = null;
      currentPolygons = null;
      currentMissedObjects = null;
      if (polygonLayer) polygonLayer.remove();
      if (nodataLayer) nodataLayer.remove();
      if (missedLayer) missedLayer.remove();
      renderJobDetail(null);
      renderReviewStats();
      renderMissedStats();
      updateDiagnostic(null);
    }
    comparisonJobIds = comparisonJobIds.filter(id => id !== job.id);
    saveComparisonJobIds();
    statusEl.textContent = `Deleted ${name}`;
    await refreshJobs();
    void renderComparison();
  }

  async function renameJob(job) {
    const next = window.prompt('Job name', job.label || job.prompt || '');
    if (next === null) return;
    const res = await fetch(`/jobs/${job.id}/label`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label: next}),
    });
    statusEl.textContent = res.ok ? 'Job renamed' : 'Rename failed';
    await refreshJobs();
    void renderComparison();
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
      currentJobSummary = await fetchJobSummary(jobId);
      renderJobDetail(currentJobSummary);
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
    if (missedLayer) missedLayer.remove();

    const [polygons, nodata, missed, summary] = await Promise.all([
      fetchJson(`/jobs/${jobId}/polygons`),
      fetchJson(`/jobs/${jobId}/nodata`),
      fetchJson(`/jobs/${jobId}/missed_objects`),
      fetchJobSummary(jobId),
    ]);

    currentPolygons = polygons;
    currentMissedObjects = missed;
    currentJobSummary = summary;
    if (summary?.bbox_wgs84) {
      setDrawnBbox(summary.bbox_wgs84);
    }
    renderJobDetail(summary);
    renderPolygons();
    renderMissedObjects();
    updateDiagnostic(null);

    if (nodata) {
      nodataLayer = L.geoJSON(nodata, {
        style: {color: '#172431', fillOpacity: 0.08, dashArray: '4 4', weight: 1},
      }).addTo(map);
    }
    if (!fitToOpenJob(polygons, nodata, missed) && summary?.bbox_wgs84) {
      setDrawnBbox(summary.bbox_wgs84, {fit: true});
    }
    statusEl.textContent = summary
      ? `Opened ${jobName(summary)}`
      : `Opened ${jobName({id: jobId})} without review data`;
    renderJobs();
    void renderComparison();
  }

  function fitToOpenJob(...collections) {
    const layers = [];
    for (const collection of collections) {
      if (!collection?.features?.length) continue;
      layers.push(L.geoJSON(collection));
    }
    if (!layers.length) return false;
    const group = L.featureGroup(layers);
    const bounds = group.getBounds();
    if (!bounds.isValid()) return false;
    map.fitBounds(bounds.pad(0.12), {maxZoom: 19});
    return true;
  }

  function renderJobDetail(summary) {
    if (!summary) {
      openReviewTitleEl.textContent = 'Open review';
      openReviewStatusEl.textContent = 'no job';
      openReviewStatusEl.className = 'pill';
      detail.precision.textContent = 'n/a';
      detail.recall.textContent = 'n/a';
      detail.rejectedShare.textContent = 'n/a';
      detail.lowScore.textContent = 'n/a';
      detail.failedTiles.textContent = 'n/a';
      detail.scoreRange.textContent = 'n/a';
      detail.exportStale.textContent = 'n/a';
      detail.prompt.textContent = 'n/a';
      detail.bbox.textContent = 'n/a';
      detail.preset.textContent = 'n/a';
      detail.vector.textContent = 'n/a';
      detail.modality.textContent = 'n/a';
      detail.status.textContent = 'n/a';
      detail.tiles.textContent = 'n/a';
      detail.revision.textContent = 'n/a';
      detail.created.textContent = 'n/a';
      detail.runtime.textContent = 'n/a';
      detail.model.textContent = 'n/a';
      detail.preprocess.textContent = 'n/a';
      detail.git.textContent = 'n/a';
      exportStateEl.textContent = 'not exported';
      exportStateEl.className = 'pill';
      return;
    }
    const metadata = summary.run_metadata || {};
    const settings = metadata.settings || {};
    const vectorOptions = summary.vector_options || metadata.vector_options;
    const rejectedShare = summary.total ? summary.rejected / summary.total : null;
    const lowScore =
      Number(summary.score_buckets?.lt_035 || 0) + Number(summary.score_buckets?.gte_035_lt_05 || 0);
    const tileDone = Number(summary.tile_completed || 0) + Number(summary.tile_failed || 0);
    const tileTotal = summary.tile_total ?? 'n/a';
    openReviewTitleEl.textContent = summary.label || summary.prompt || 'Open review';
    openReviewStatusEl.textContent = summary.status;
    openReviewStatusEl.className = `pill ${statusClass(summary.status)}`;
    detail.precision.textContent = formatPercent(summary.precision_review);
    detail.recall.textContent = formatPercent(summary.recall_estimate);
    detail.rejectedShare.textContent = formatPercent(rejectedShare);
    detail.lowScore.textContent = `${lowScore} below 0.50`;
    detail.failedTiles.textContent = String(summary.tile_failed || 0);
    detail.scoreRange.textContent = `${formatScore(summary.min_score)} - ${formatScore(summary.max_score)} (avg ${formatScore(summary.avg_score)})`;
    detail.prompt.textContent = summary.prompt || 'n/a';
    detail.bbox.textContent = formatBbox(summary.bbox_wgs84);
    detail.preset.textContent = summary.tile_preset || 'n/a';
    detail.vector.textContent = formatVectorOptions(vectorOptions);
    detail.modality.textContent = formatModality(summary.modality_filter || metadata.modality_filter);
    detail.status.textContent = summary.status || 'n/a';
    detail.tiles.textContent = `${tileDone}/${tileTotal}`;
    detail.revision.textContent = String(summary.validation_revision ?? 'n/a');
    detail.exportStale.textContent = summary.export_stale
      ? `stale${summary.exported_revision !== null && summary.exported_revision !== undefined ? `, exported rev ${summary.exported_revision}` : ''}`
      : 'fresh';
    detail.created.textContent = formatDateTime(summary.created_at);
    detail.runtime.textContent = formatRuntime(summary.started_at, summary.finished_at);
    detail.model.textContent = settings.SAM3_MODEL_ID || 'n/a';
    detail.preprocess.textContent = settings.SAM_IMAGE_PREPROCESS || 'n/a';
    detail.git.textContent = metadata.git_commit_sha ? shortId(metadata.git_commit_sha) : 'n/a';
    exportStateEl.textContent = summary.export_stale ? 'stale' : 'fresh';
    exportStateEl.className = `pill ${summary.export_stale ? 'warn' : 'ready'}`;
    if (summary.missed_marked !== undefined) kpiMissedEl.textContent = String(summary.missed_marked || 0);
  }

  function renderScoreHistogram() {
    scoreHistogramEl.innerHTML = '';
    const buckets = currentJobSummary?.score_buckets;
    const rows = [
      ['<0.35', buckets?.lt_035 || 0, true],
      ['0.35-0.50', buckets?.gte_035_lt_05 || 0, true],
      ['0.50-0.70', buckets?.gte_05_lt_07 || 0, false],
      ['>=0.70', buckets?.gte_07 || 0, false],
    ];
    const max = Math.max(...rows.map(([, value]) => Number(value || 0)), 1);
    rows.forEach(([labelText, value, low]) => {
      const row = document.createElement('div');
      row.className = `score-bucket${low ? ' low' : ''}`;
      const label = document.createElement('span');
      label.textContent = labelText;
      const track = document.createElement('div');
      track.className = 'score-bucket-track';
      const bar = document.createElement('span');
      bar.style.setProperty('--value', `${Math.max(3, (Number(value || 0) / max) * 100)}%`);
      const count = document.createElement('b');
      count.textContent = String(value || 0);
      track.append(bar);
      row.append(label, track, count);
      scoreHistogramEl.appendChild(row);
    });
    thresholdTitleEl.textContent = currentJobSummary
      ? `Precision proxy: ${formatPercent(currentJobSummary.precision_review)}`
      : 'No review metrics yet';
    thresholdCopyEl.textContent = currentJobSummary
      ? `Recall estimate: ${formatPercent(currentJobSummary.recall_estimate)}. Buckets show model confidence before your visible score filter.`
      : 'Open a job to inspect accepted and rejected score buckets.';
  }

  scoreFilterEl.addEventListener('input', renderPolygons);
  segmentOpacityEl.addEventListener('input', renderPolygons);
  showRejectedEl.addEventListener('change', renderPolygons);
  jobSearchEl.addEventListener('input', renderJobs);
  showFailedJobsEl.addEventListener('change', renderJobs);
  showExportedJobsEl.addEventListener('change', renderJobs);
  savePresetEl?.addEventListener('click', saveCurrentPreset);
  toggleAllJobsEl.addEventListener('click', () => {
    showAllJobs = !showAllJobs;
    renderJobs();
  });
  rejectBelowScoreEl.addEventListener('click', rejectAcceptedBelowScore);
  toggleMissedModeEl.addEventListener('click', () => {
    if (!currentJobId) {
      statusEl.textContent = 'Open a job first';
      return;
    }
    missedMode = !missedMode;
    renderMissedStats();
  });
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
  copyMetadataEl?.addEventListener('click', async () => {
    if (!currentJobId) {
      statusEl.textContent = 'Open a job first';
      return;
    }
    const job = await fetchJson(`/jobs/${currentJobId}`);
    const payload = JSON.stringify(job?.run_metadata || {}, null, 2);
    await navigator.clipboard?.writeText(payload);
    statusEl.textContent = 'Run metadata copied';
  });

  function readNumberInput(id) {
    const el = document.getElementById(id);
    if (!el || el.value === '') return null;
    const n = Number(el.value);
    return Number.isFinite(n) ? n : null;
  }

  function buildModalityFilter() {
    return {
      ndvi_min: readNumberInput('ndvi-min'),
      ndvi_max: readNumberInput('ndvi-max'),
      ndsm_min: readNumberInput('ndsm-min'),
      ndsm_max: readNumberInput('ndsm-max'),
    };
  }

  function buildVectorOptions() {
    return {
      simplification_tolerance_m: readNumberInput('simplify-tolerance'),
      orthogonalize: Boolean(orthogonalizeEl?.checked),
    };
  }

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
        modality_filter: buildModalityFilter(),
        vector_options: buildVectorOptions(),
      }),
    });
    if (!res.ok) {
      statusEl.textContent = await res.text();
      return;
    }
    const job = await res.json();
    currentJobId = job.id;
    statusEl.textContent = `Queued ${promptEl.value.trim() || 'job'}`;
    await refreshJobs();
  };

  exportEl.onclick = async () => {
    if (!currentJobId) {
      statusEl.textContent = 'Open a job first';
      return;
    }
    await flushValidations();
    const res = await fetch(`/jobs/${currentJobId}/export`, {method: 'POST'});
    statusEl.textContent = res.ok ? 'Exported' : 'Export failed';
    await refreshJobs();
    currentJobSummary = await fetchJobSummary(currentJobId);
    renderJobDetail(currentJobSummary);
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

  renderPresetButtons();
  renderReviewStats();
  renderMissedStats();
  renderJobDetail(null);
  renderCustomPresets();
  setInterval(() => {
    void refreshJobs();
    void refreshHealth();
  }, 3000);
  void refreshJobs().then(() => refreshHealth());
  void renderComparison();
})();
