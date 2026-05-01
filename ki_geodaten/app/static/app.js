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
  const showFailedJobsEl = document.getElementById('show-failed-jobs');
  const showExportedJobsEl = document.getElementById('show-exported-jobs');
  const jobCountEl = document.getElementById('job-count');

  let bbox = null;
  let currentJobId = null;
  let polygonLayer = null;
  let nodataLayer = null;
  let missedLayer = null;
  let currentPolygons = null;
  let currentMissedObjects = null;
  let latestJobs = [];
  let missedMode = false;
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

  map.on('click', (event) => {
    if (!missedMode || !currentJobId) return;
    void addMissedObject(event.latlng);
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
    const fill = Number(segmentOpacityEl.value || 0.28);
    return {
      color: accepted ? '#2563eb' : '#dc2626',
      fillColor: accepted ? '#60a5fa' : '#f87171',
      fillOpacity: underFilter ? Math.max(0.03, fill * 0.3) : fill,
      opacity: underFilter ? 0.35 : Math.min(1, fill + 0.35),
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

  function jobName(job) {
    return job?.label || job?.prompt || job?.id || 'job';
  }

  function shortId(id) {
    return String(id || '').slice(0, 8);
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

  function renderMissedStats() {
    const count = currentMissedObjects?.features?.length || 0;
    missedStatsEl.textContent = `${count} marked | ${missedMode ? 'click map to add' : 'mode off'}`;
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
        color: '#f59e0b',
        fillColor: '#fbbf24',
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
    currentMissedObjects = await fetch(`/jobs/${currentJobId}/missed_objects`).then(r => r.json());
    renderMissedObjects();
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
    const jobs = await res.json();
    latestJobs = jobs;
    renderJobs();
  }

  function renderJobs() {
    const jobs = latestJobs.filter(jobMatchesFilters);
    jobsEl.innerHTML = '';
    jobCountEl.textContent = `${jobs.length}/${latestJobs.length}`;
    for (const job of jobs) {
      const el = document.createElement('div');
      el.className = 'job';
      if (job.id === currentJobId) el.classList.add('active');
      const done = (job.tile_completed || 0) + (job.tile_failed || 0);
      const titleWrap = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'job-title';
      title.textContent = jobName(job);
      const meta = document.createElement('div');
      meta.className = 'job-meta';
      meta.textContent = `${job.status} ${done}/${job.tile_total || '?'} | ${shortId(job.id)}`;
      titleWrap.append(title, meta);
      if (job.status === 'FAILED') {
        el.title = [job.error_reason, job.error_message].filter(Boolean).join('\n\n');
        meta.textContent = `FAILED${job.error_reason ? ` (${job.error_reason})` : ''} | ${shortId(job.id)}`;
      }
      const actions = document.createElement('div');
      actions.className = 'job-actions';
      const rename = document.createElement('button');
      rename.type = 'button';
      rename.textContent = 'Rename';
      rename.onclick = (event) => {
        event.stopPropagation();
        void renameJob(job);
      };
      actions.append(rename);
      el.append(titleWrap, actions);
      el.onclick = () => openJob(job.id);
      jobsEl.appendChild(el);
    }
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
      fetch(`/jobs/${jobId}/polygons`).then(r => r.json()),
      fetch(`/jobs/${jobId}/nodata`).then(r => r.json()),
      fetch(`/jobs/${jobId}/missed_objects`).then(r => r.json()),
      fetchJobSummary(jobId),
    ]);
    currentPolygons = polygons;
    currentMissedObjects = missed;
    renderPolygons();
    renderMissedObjects();
    nodataLayer = L.geoJSON(nodata, {
      style: {color: '#111827', fillOpacity: 0.1, dashArray: '4 4'},
    }).addTo(map);
    void renderComparison();
  }

  scoreFilterEl.addEventListener('input', renderPolygons);
  segmentOpacityEl.addEventListener('input', renderPolygons);
  showRejectedEl.addEventListener('change', renderPolygons);
  jobSearchEl.addEventListener('input', renderJobs);
  showFailedJobsEl.addEventListener('change', renderJobs);
  showExportedJobsEl.addEventListener('change', renderJobs);
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
  function readNumberInput(id) {
    const el = document.getElementById(id);
    if (!el || el.value === '') return null;
    const n = Number(el.value);
    return Number.isFinite(n) ? n : null;
  }

  function readBooleanInput(id) {
    const el = document.getElementById(id);
    return Boolean(el?.checked);
  }

  function buildVectorTopology() {
    return {
      simplify_tolerance_m: readNumberInput('simplify-tolerance') ?? 0,
      orthogonalize: readBooleanInput('orthogonalize'),
      orthogonalize_angle_tolerance_deg: readNumberInput('orthogonal-angle-tolerance') ?? 12,
      orthogonalize_max_area_delta_ratio: readNumberInput('orthogonal-area-delta') ?? 0.25,
      orthogonalize_max_shift_m: readNumberInput('orthogonal-max-shift') ?? 2,
    };
  }

  function buildModalityFilter() {
    return {
      ndvi_min: readNumberInput('ndvi-min'),
      ndvi_max: readNumberInput('ndvi-max'),
      ndsm_min: readNumberInput('ndsm-min'),
      ndsm_max: readNumberInput('ndsm-max'),
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
        vector_topology: buildVectorTopology(),
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
  renderMissedStats();
})();
