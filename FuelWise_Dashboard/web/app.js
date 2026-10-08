'use strict';
const $ = (id) => document.getElementById(id),
  fmt = (v, n = 1) =>
    v == null || !Number.isFinite(Number(v))
      ? '—'
      : Number(v).toLocaleString('zh-TW', { maximumFractionDigits: n, minimumFractionDigits: n });
const esc = (v) =>
  String(v ?? '—').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const sum = (a, k) => a.reduce((s, r) => s + (Number.isFinite(r[k]) ? r[k] : 0), 0);
let catalog,
  current,
  tripId,
  map,
  routeLayer,
  stopLayer,
  cursorMarker,
  fleetPage = 0,
  rawPage = 0,
  cursorIndex = 0,
  playing = null,
  requestNumber = 0,
  fuelModelRequestNumber = 0,
  activeFuelModel = null,
  activeTab = 'trip',
  liveReturnTab = 'trip';
let peers = [],
  fleetRows = [],
  liveAiHistory = [],
  liveAiSending = false,
  liveAiRequestNumber = 0;
let liveMap,
  liveRouteLayer,
  liveSegment,
  liveMarker,
  liveTimer = null,
  liveIndex = 0,
  liveActive = false,
  liveRunning = false,
  liveAccelerationDirection = 0,
  liveAccelerationStreak = 0,
  liveEcoAlertDirection = 0,
  liveEcoAlertRemaining = 0;
const LIVE_ACCELERATION_THRESHOLD = 3;
const MIN_COMPARE_REFERENCE_TRIPS = 5;
const LIVE_AI_SENSOR_FIELDS = [
  'speed',
  'rpm',
  'load',
  'temp',
  'battery',
  'fuel_level',
  'distance',
  'used',
];
const layout = {
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
  font: { family: 'system-ui, Microsoft JhengHei', color: '#56616a', size: 11 },
  margin: { l: 55, r: 20, t: 32, b: 48 },
  hovermode: 'closest',
  xaxis: { gridcolor: '#e6e9eb', zeroline: false },
  yaxis: { gridcolor: '#e6e9eb', zeroline: false },
  showlegend: false,
};
const config = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d'],
  toImageButtonOptions: { format: 'png', scale: 2 },
};
async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw Error(`資料讀取失敗 (${r.status})`);
  return r.json();
}
function appendLiveAiMessage(role, text) {
  const message = document.createElement('p');
  message.className = `live-ai-message ${role}`;
  message.textContent = text;
  $('liveAiMessages').append(message);
  $('liveAiMessages').scrollTop = $('liveAiMessages').scrollHeight;
  return message;
}
function liveAiPoint(point, offsetSec = null) {
  const sample = {};
  if (Number.isFinite(Number(offsetSec))) sample.offset_sec = Number(offsetSec);
  LIVE_AI_SENSOR_FIELDS.forEach((field) => {
    const value = point[field];
    sample[field] =
      value == null || !Number.isFinite(Number(value)) ? null : Number(value);
  });
  return sample;
}
function buildLiveAiSensorContext() {
  const points = current.points;
  const events = [];
  let direction = 0;
  let streak = 0;
  let event = null;
  let maxAcceleration = 0;
  let maxBraking = 0;
  const finishEvent = () => {
    if (event) events.push(event);
    event = null;
  };
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    const previous = points[i - 1];
    const elapsedSec = Number(point.sec) - Number(previous.sec);
    const hasSpeed =
      !point.break_before &&
      point.speed != null &&
      previous.speed != null &&
      Number.isFinite(Number(point.speed)) &&
      Number.isFinite(Number(previous.speed)) &&
      Number.isFinite(elapsedSec) &&
      elapsedSec > 0;
    const acceleration = hasSpeed
      ? (Number(point.speed) - Number(previous.speed)) / elapsedSec
      : null;
    const nextDirection =
      acceleration != null && acceleration > LIVE_ACCELERATION_THRESHOLD
        ? 1
        : acceleration != null && acceleration < -LIVE_ACCELERATION_THRESHOLD
          ? -1
          : 0;
    if (!nextDirection) {
      finishEvent();
      direction = 0;
      streak = 0;
      continue;
    }
    if (nextDirection !== direction) {
      finishEvent();
      direction = nextDirection;
      streak = 0;
    }
    streak++;
    const absoluteAcceleration = Math.abs(acceleration);
    if (direction > 0) maxAcceleration = Math.max(maxAcceleration, absoluteAcceleration);
    else maxBraking = Math.max(maxBraking, absoluteAcceleration);
    if (streak === 3) {
      event = {
        type: direction > 0 ? 'acceleration' : 'braking',
        start_offset_sec: Number(points[i - 2].sec),
        end_offset_sec: Number(point.sec),
        max_kmh_s: absoluteAcceleration,
      };
    } else if (streak > 3 && event) {
      event.end_offset_sec = Number(point.sec);
      event.max_kmh_s = Math.max(event.max_kmh_s, absoluteAcceleration);
    }
  }
  finishEvent();
  const currentPoint = points[Math.min(liveIndex, points.length - 1)];
  const recentPoints = points
    .slice(Math.max(0, liveIndex - 19), liveIndex + 1)
    .map((point) => liveAiPoint(point, point.sec));
  const qualifyingEvents = events.filter((item) => item.type);
  return {
    threshold_kmh_s: LIVE_ACCELERATION_THRESHOLD,
    current: liveAiPoint(currentPoint, currentPoint.sec),
    recent_points: recentPoints,
    trip_summary: {
      observations: points.length,
      acceleration_events: qualifyingEvents.filter((item) => item.type === 'acceleration').length,
      braking_events: qualifyingEvents.filter((item) => item.type === 'braking').length,
      max_acceleration_kmh_s: maxAcceleration,
      max_braking_kmh_s: maxBraking,
      recent_events: qualifyingEvents.slice(-10),
    },
  };
}
async function refreshLiveAiStatus() {
  try {
    const { configured, model } = await get('/api/ai/status');
    $('liveAiStatus').textContent = configured
      ? `Gemini 已設定 · ${model}`
      : `尚未設定 GEMINI_API_KEY · ${model}`;
    $('liveAiStatus').classList.toggle('ready', configured);
  } catch (e) {
    $('liveAiStatus').textContent = `模型狀態讀取失敗：${e.message}`;
  }
}
async function submitLiveAiQuestion(event) {
  event.preventDefault();
  const input = $('liveAiInput');
  const question = input.value.trim();
  if (!question || liveAiSending) return;
  if (!current || !current.points.length) {
    appendLiveAiMessage('assistant', '目前沒有可分析的行程資料，請先選取一趟行程。');
    return;
  }
  appendLiveAiMessage('user', question);
  input.value = '';
  input.disabled = true;
  $('liveAiSend').disabled = true;
  liveAiSending = true;
  const requestNumber = ++liveAiRequestNumber;
  const pending = appendLiveAiMessage('assistant', '正在分析感測器資料…');
  try {
    const response = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: question,
        history: liveAiHistory.slice(-8),
        sensor_context: buildLiveAiSensorContext(),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.error || `AI 請求失敗 (${response.status})`);
    if (requestNumber !== liveAiRequestNumber) return;
    pending.textContent = result.answer;
    liveAiHistory.push(
      { role: 'user', content: question },
      { role: 'assistant', content: result.answer },
    );
    liveAiHistory = liveAiHistory.slice(-12);
  } catch (e) {
    if (requestNumber === liveAiRequestNumber) {
      pending.textContent = `無法取得 AI 回覆：${e.message}`;
      pending.classList.add('error');
    }
  } finally {
    if (requestNumber === liveAiRequestNumber) {
      input.disabled = false;
      $('liveAiSend').disabled = false;
      liveAiSending = false;
      if (document.activeElement !== input) input.focus();
    }
  }
}
function error(e) {
  $('error').textContent = e.message || String(e);
  $('error').hidden = false;
  $('loading').hidden = true;
}
function card(title, value, unit = '', note = '') {
  return `<div class="card"><span>${esc(title)}</span><strong>${value} <small>${esc(unit)}</small></strong><small>${esc(note)}</small></div>`;
}
function clearFuelModel(message, warning = false) {
  activeFuelModel = null;
  $('fuelModelContent').hidden = true;
  $('fuelModelBadge').hidden = true;
  $('fuelModelCards').innerHTML = '';
  $('fuelModelWarning').hidden = true;
  $('fuelModelWarning').textContent = '';
  $('fuelModelFeatureList').innerHTML = '';
  $('fuelModelDetailsBody').innerHTML = '';
  if (window.Plotly) Plotly.purge('fuelModelChart');
  $('fuelModelStatus').className = warning ? 'notice warn' : 'notice';
  $('fuelModelStatus').textContent = message;
  $('fuelModelStatus').hidden = !message;
}
function renderFuelModelDetails(data) {
  const model = data.model,
    comparisons = model.validation_comparison_mae_l_per_100km || {},
    validationMetric = model.validation_metrics.l_per_100km || {},
    methodName =
      model.method === 'history_xgboost'
        ? '車輛歷史特徵 XGBoost'
        : model.method === 'total_fuel_xgboost'
          ? '總耗油目標 XGBoost'
          : model.method;
  const splitName = {
    train: '訓練集',
    validation: '驗證集',
    test: '測試集追蹤評估',
    other: '實驗樣本以外',
  }[data.data_split] || data.data_split;
  const sourceName = {
    saved_trip_features: '已保存的逐趟特徵（依資料分組建構歷史）',
    frozen_training_history: '凍結的訓練集歷史統計',
  }[data.feature_source] || data.feature_source;
  const validationRows = Object.entries(comparisons)
    .map(
      ([name, value]) =>
        `<tr><td>${esc(name)}</td><td>${fmt(value, 3)} L/100 km</td></tr>`,
    )
    .join('');
  const vehicleRows = (validationMetric.per_vehicle || [])
    .map(
      (row) =>
        `<tr><td>${esc(row.vehicle)}</td><td>${fmt(row.sample_count, 0)}</td><td>${fmt(row.mae, 3)} L/100 km</td></tr>`,
    )
    .join('');
  const testMetric = model.test_follow_up_metrics?.l_per_100km;
  const limitations = data.limitations
    .map((item) => `<li>${esc(item)}</li>`)
    .join('');
  $('fuelModelDetailsBody').innerHTML = `
    <dl class="fuel-model-meta">
      <dt>目前展示模型</dt><dd>${esc(methodName)}（實驗模型）</dd>
      <dt>模型版本</dt><dd>XGBoost ${esc(model.xgboost_version)}；使用 ${fmt(model.tree_count, 0)} 棵樹</dd>
      <dt>訓練目標</dt><dd>${model.target === 'fuel_l' ? '整趟耗油（L）' : '平均油耗（L/100 km）'}</dd>
      <dt>目前行程分組</dt><dd>${esc(splitName)}</dd>
      <dt>歷史特徵來源</dt><dd>${esc(sourceName)}</dd>
      <dt>驗證集選擇理由</dt><dd>${esc(model.selection_reason)}</dd>
      <dt>驗證集評估</dt><dd>${fmt(validationMetric.n, 0)} 趟；MAE ${fmt(validationMetric.mae, 3)}、RMSE ${fmt(validationMetric.rmse, 3)}、R² ${fmt(validationMetric.r2, 3)}；各車 MAE 等權平均 ${fmt(validationMetric.equal_weight_vehicle_mae, 3)} L/100 km</dd>
      <dt>驗證絕對誤差</dt><dd>中位數 ${fmt(validationMetric.median_absolute_error, 3)}；P90 ${fmt(validationMetric.p90_absolute_error, 3)} L/100 km</dd>
      <dt>測試追蹤 MAE</dt><dd>${testMetric ? `${fmt(testMetric.mae, 3)} L/100 km（${fmt(testMetric.n, 0)} 趟）` : '未提供'}</dd>
      <dt>SHAP 解釋</dt><dd>${esc(data.explanation.method)}；${esc(data.explanation.background)}</dd>
    </dl>
    <h4>驗證集 MAE 對照</h4>
    <div class="tablewrap"><table><thead><tr><th>方法</th><th>MAE</th></tr></thead><tbody>${validationRows}</tbody></table></div>
    <h4>驗證集各車 MAE</h4>
    <div class="tablewrap"><table><thead><tr><th>車輛</th><th>行程數</th><th>MAE</th></tr></thead><tbody>${vehicleRows}</tbody></table></div>
    <ul>${limitations}</ul>`;
}
async function renderFuelModel(data, tripRequest, modelRequest) {
  if (
    modelRequest !== fuelModelRequestNumber ||
    tripRequest !== requestNumber ||
    Number($('journey').value) !== tripId
  ) {
    return;
  }
  const fuelTarget = data.model.target === 'fuel_l',
    actual = data.actual,
    estimate = data.estimate,
    diff = data.estimated_difference,
    baseline = data.history_baseline.l_per_100km,
    actualL100 = actual.l_per_100km,
    estimateL100 = estimate.l_per_100km,
    differenceL100 =
      actualL100 == null || estimateL100 == null
        ? null
        : actualL100 - estimateL100;
  $('fuelModelStatus').hidden = true;
  $('fuelModelContent').hidden = false;
  $('fuelModelBadge').hidden = false;
  $('fuelModelBadge').textContent = '實驗模型';
  $('fuelModelCards').innerHTML = fuelTarget
    ? card('實際耗油', fmt(actual.fuel_l, 2), 'L', `實際油耗 ${fmt(actualL100, 2)} L/100 km`) +
      card('模型估計', fmt(estimate.fuel_l, 2), 'L', `換算 ${fmt(estimateL100, 2)} L/100 km`) +
      card('估計差異（實際－估計）', fmt(diff.value, 2), 'L', `換算差異 ${fmt(differenceL100, 2)} L/100 km`) +
      card('歷史中位數基準', fmt(baseline, 2), 'L/100 km', '僅為訓練行程車輛油耗中位數')
    : card('實際油耗', fmt(actualL100, 2), 'L/100 km', `本趟總耗油 ${fmt(actual.fuel_l, 2)} L`) +
      card('模型估計', fmt(estimateL100, 2), 'L/100 km', `換算本趟耗油 ${fmt(estimate.fuel_l, 2)} L`) +
      card('估計差異（實際－估計）', fmt(diff.value, 2), 'L/100 km', `本趟耗油差異 ${fmt(actual.fuel_l == null || estimate.fuel_l == null ? null : actual.fuel_l - estimate.fuel_l, 2)} L`) +
      card('歷史中位數基準', fmt(baseline, 2), 'L/100 km', '僅為訓練行程車輛油耗中位數');

  const model = data.model,
    comparison = model.validation_comparison_mae_l_per_100km || {},
    baselineMae = comparison.vehicle_median_baseline,
    selectedMae = model.validation_metrics.l_per_100km?.mae,
    warnings = [...data.warnings];
  if (selectedMae != null && baselineMae != null && selectedMae >= baselineMae) {
    warnings.unshift(
      `目前機器學習模型尚未優於歷史基準：驗證 MAE ${fmt(selectedMae, 3)}，車輛中位數基準 ${fmt(baselineMae, 3)} L/100 km。`,
    );
  } else {
    warnings.unshift(
      `本次驗證集 MAE 為 ${fmt(selectedMae, 3)}，車輛中位數基準為 ${fmt(baselineMae, 3)} L/100 km；仍屬實驗展示。`,
    );
  }
  $('fuelModelWarning').hidden = false;
  $('fuelModelWarning').innerHTML = warnings.map(esc).join('<br>');
  $('fuelModelChartSummary').textContent =
    `模型解釋起點 ${fmt(data.explanation.base_value, 3)} ${data.explanation.target_unit}；` +
    `最終估計 ${fmt(data.explanation.prediction, 3)} ${data.explanation.target_unit}。` +
    '正貢獻推高模型估計，負貢獻降低模型估計。';
  await renderFuelModelChart(data);
  if (
    modelRequest !== fuelModelRequestNumber ||
    tripRequest !== requestNumber
  ) {
    if (
      activeFuelModel &&
      activeFuelModel.modelRequest === fuelModelRequestNumber
    ) {
      await renderFuelModelChart(activeFuelModel.data);
    }
    return;
  }
  const items = data.explanation.features;
  $('fuelModelFeatureList').innerHTML =
    `<table><thead><tr><th>特徵</th><th>本趟特徵值</th><th>SHAP 貢獻</th><th>模型方向</th></tr></thead><tbody>` +
    items
      .map((item) => {
        const value =
          item.value == null
            ? '合併其他特徵'
            : `${fmt(item.value, 3)} ${esc(item.unit)}`;
        return `<tr><td>${esc(item.name)}</td><td>${value}</td><td>${fmt(item.shap_value, 3)} ${esc(data.explanation.target_unit)}</td><td>${item.shap_value > 0 ? '推高估計' : item.shap_value < 0 ? '降低估計' : '無明顯改變'}</td></tr>`;
      })
      .join('') +
    '</tbody></table>';
  renderFuelModelDetails(data);
}
async function renderFuelModelChart(data) {
  const items = data.explanation.features;
  const labels = [
    '模型解釋起點',
    ...items.map((item) => item.name),
    '模型最終估計',
  ];
  const measures = [
    'absolute',
    ...items.map(() => 'relative'),
    'total',
  ];
  const values = [
    data.explanation.base_value,
    ...items.map((item) => item.shap_value),
    data.explanation.prediction,
  ];
  await drawPlot(
    'fuelModelChart',
    [
      {
        type: 'waterfall',
        orientation: 'h',
        y: labels,
        x: values,
        measure: measures,
        increasing: { marker: { color: '#b96e49' } },
        decreasing: { marker: { color: '#557a91' } },
        totals: { marker: { color: '#53616a' } },
        connector: { line: { color: '#aab2b8', width: 1 } },
        hovertemplate: '%{y}<br>貢獻／估計：%{x:.3f} ' +
          `${esc(data.explanation.target_unit)}<extra></extra>`,
      },
    ],
    {
      margin: { l: 185, r: 28, t: 20, b: 55 },
      xaxis: { title: `模型輸出與特徵貢獻（${data.explanation.target_unit}）` },
      yaxis: { automargin: true },
      height: 360,
    },
  );
}
async function loadFuelModel(id, tripRequest, modelRequest) {
  try {
    const response = await fetch(`/api/fuel-model?id=${id}`);
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      if (response.status === 404) {
        throw Error('目前執行的伺服器版本尚未載入油耗模型 API，請重新啟動 FuelWise 後再試。');
      }
      throw Error(`模型 API 回傳非 JSON 回應 (${response.status})，請重新啟動 FuelWise 後再試。`);
    }
    const data = await response.json();
    if (
      modelRequest !== fuelModelRequestNumber ||
      tripRequest !== requestNumber ||
      Number($('journey').value) !== id
    ) {
      return;
    }
    if (!response.ok || !data.available) {
      throw Error(data.reason || `模型估計讀取失敗 (${response.status})`);
    }
    activeFuelModel = { data, modelRequest };
    await renderFuelModel(data, tripRequest, modelRequest);
  } catch (e) {
    if (
      modelRequest === fuelModelRequestNumber &&
      tripRequest === requestNumber &&
      Number($('journey').value) === id
    ) {
      clearFuelModel(e.message || '模型估計暫時無法使用。', true);
    }
  }
}
function table(id, headers, rows) {
  $(id).innerHTML =
    `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('') || '<tr><td>此範圍沒有資料</td></tr>'}</tbody></table>`;
}
function withinDate(t) {
  return (
    (!$('dateFrom').value || t.start.slice(0, 10) >= $('dateFrom').value) &&
    (!$('dateTo').value || t.start.slice(0, 10) <= $('dateTo').value)
  );
}
function setTab(name) {
  activeTab = name;
  document.body.classList.toggle('live-mode', name === 'live');
  $('cards').hidden = name === 'live';
  document
    .querySelectorAll('.tab')
    .forEach((e) => (e.hidden = e.id !== name || (name !== 'fleet' && !current)));
  document
    .querySelectorAll('nav button')
    .forEach((e) => e.classList.toggle('active', e.dataset.tab === name));
  $('title').textContent = document
    .querySelector(`nav [data-tab="${name}"]`)
    .textContent
    .trim();
  setTimeout(() => {
    if (map) map.invalidateSize();
    if (liveMap) liveMap.invalidateSize();
    document
      .querySelectorAll('.tab:not([hidden]) .js-plotly-plot')
      .forEach((e) => Plotly.Plots.resize(e));
  }, 80);
  if (name === 'quality' && current) loadRaw();
}
function drawPlot(id, data, extra = {}) {
  if (!window.Plotly) {
    $(id).innerHTML =
      '<p class="notice warn">圖表函式庫載入失敗，請確認能連線 cdn.plot.ly 後重新整理。</p>';
    return Promise.resolve();
  }
  return Plotly.react(id, data, { ...layout, ...extra }, config);
}
function fillJourneys() {
  const rows = catalog.trips
    .filter((t) => t.vehicle === $('vehicle').value && withinDate(t))
    .sort((a, b) => b.start.localeCompare(a.start));
  $('journey').innerHTML = rows
    .map((t) => `<option value="${t.id}">${esc(t.start)} ｜ ${esc(t.journey)}</option>`)
    .join('');
  return rows;
}
async function loadSelected() {
  stopLiveMission();
  const modelRequest = ++fuelModelRequestNumber;
  clearFuelModel('等待選取行程模型估計…');
  liveAiRequestNumber++;
  liveAiSending = false;
  $('liveAiInput').disabled = false;
  $('liveAiSend').disabled = false;
  liveAiHistory = [];
  $('liveAiMessages').innerHTML =
    '<p class="live-ai-message assistant">你好！我可以根據目前行程的感測器資料，協助分析急加速、急煞車與行駛狀況。</p>';
  const id = Number($('journey').value);
  if (!id) {
    ++requestNumber;
    clearFuelModel('目前沒有可推論的行程。', true);
    stopPlay();
    current = null;
    tripId = null;
    $('cards').innerHTML = '';
    $('subtitle').textContent = '所選範圍沒有行程';
    $('download').removeAttribute('href');
    $('loading').hidden = true;
    setTab(activeTab);
    $('error').textContent = '這台車在所選日期範圍沒有行程，請更改篩選。';
    $('error').hidden = false;
    return;
  }
  const serial = ++requestNumber;
  tripId = id;
  clearFuelModel('正在計算本趟行程油耗估計與特徵解釋…');
  loadFuelModel(id, serial, modelRequest);
  $('loading').hidden = false;
  $('error').hidden = true;
  stopPlay();
  try {
    const data = await get(`/api/trip?id=${id}`);
    if (serial !== requestNumber) return;
    current = data;
    tripId = id;
    setTab(activeTab);
    rawPage = 0;
    cursorIndex = 0;
    const t = catalog.trips.find((t) => t.id === id),
      s = data.summary;
    $('subtitle').textContent = `${t.plate} · ${t.vehicle} · ${t.journey} ｜ ${s.start} → ${s.end}`;
    $('download').href = `/api/export?id=${id}`;
    $('cards').innerHTML =
      card('本趟行駛距離', fmt(s.distance_km, 2), 'km') +
      card('本趟耗油', fmt(s.fuel_l, 2), 'L') +
      card('百公里油耗', fmt(s.l100, 2), 'L/100km') +
      card('行程歷時', fmt(s.duration_sec / 60, 1), 'min') +
      card('停車引擎運轉', fmt(s.idle_sec / 60, 1), 'min', `占可判讀時間 ${fmt(s.idle_pct)}%`);
    $('insights').innerHTML =
      data.notes.map((n) => `<p>${esc(n)}</p>`).join('') +
      (s.quality_flags
        ? `<p class="badge">${s.quality_flags} 類品質問題，請查看資料品質頁。</p>`
        : '');
    $('cursor').max = Math.max(0, data.points.length - 1);
    $('cursor').value = 0;
    $('windowStart').value = 0;
    $('windowEnd').value = (s.duration_sec / 60).toFixed(2);
    renderMap();
    renderTimeline();
    renderSignals();
    renderEvents();
    renderQuality();
    renderCompare();
    updateCursor(0);
    if (activeTab === 'quality') loadRaw();
    $('loading').hidden = true;
  } catch (e) {
    if (serial === requestNumber) error(e);
  }
}
function validPos(p) {
  return Number.isFinite(p.lat) && Number.isFinite(p.lon);
}
function color(v, max) {
  if (v == null) return '#99a9b0';
  const q = Math.max(0, Math.min(1, v / max));
  return `hsl(${215 - 210 * q},48%,47%)`;
}
function renderMap() {
  if (!window.L) {
    $('map').innerHTML =
      '<p class="notice warn">地圖函式庫載入失敗，請確認能連線 unpkg.com。數據仍可使用。</p>';
    return;
  }
  if (!map) {
    map = L.map('map', { preferCanvas: true }).setView([23.6, 121], 7);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    routeLayer = L.layerGroup().addTo(map);
    stopLayer = L.layerGroup().addTo(map);
  }
  routeLayer.clearLayers();
  stopLayer.clearLayers();
  if (cursorMarker) {
    map.removeLayer(cursorMarker);
    cursorMarker = null;
  }
  const p = current.points,
    metric = $('mapMetric').value,
    max = { speed: 110, rpm: 2500, load: 100 }[metric];
  $('mapLegend').textContent =
    `藍 0 → 紅 ≥${max} ${metric === 'speed' ? 'km/h' : metric === 'rpm' ? 'rpm' : '%'}`;
  for (let i = 1; i < p.length; i++) {
    if (!validPos(p[i]) || !validPos(p[i - 1]) || p[i].break_before) continue;
    L.polyline(
      [
        [p[i - 1].lat, p[i - 1].lon],
        [p[i].lat, p[i].lon],
      ],
      { color: color(p[i][metric], max), weight: 5, opacity: 0.85 },
    )
      .addTo(routeLayer)
      .on('click', () => updateCursor(i));
  }
  const good = p.filter(validPos);
  if (good.length) {
    const ends = [
      [good[0], '#718391', '第一個有效GPS點'],
      [good.at(-1), '#e2534b', '最後一個有效GPS點'],
    ];
    ends.forEach(([v, c, t]) =>
      L.circleMarker([v.lat, v.lon], {
        radius: 8,
        color: 'white',
        weight: 2,
        fillColor: c,
        fillOpacity: 1,
      })
        .addTo(routeLayer)
        .bindTooltip(t),
    );
    map.fitBounds(L.latLngBounds(good.map((v) => [v.lat, v.lon])), {
      padding: [25, 25],
      maxZoom: 15,
    });
  }
  current.stops.forEach((s) => {
    if (validPos(s))
      L.circleMarker([s.lat, s.lon], {
        radius: Math.min(18, 6 + s.seconds / 500),
        color: '#d78a16',
        fillColor: '#efb652',
        fillOpacity: 0.75,
      })
        .addTo(stopLayer)
        .bindTooltip(`${esc(s.start)} · ${fmt(s.seconds / 60)} 分鐘`)
        .on('click', () => updateCursor(s.index));
  });
}
function ensureLiveMap() {
  if (!window.L) {
    error(Error('地圖函式庫載入失敗，請確認能連線 unpkg.com。'));
    return false;
  }
  if (!liveMap) {
    liveMap = L.map('liveMap', { preferCanvas: true }).setView([23.6, 121], 7);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(liveMap);
    liveRouteLayer = L.layerGroup().addTo(liveMap);
  }
  liveRouteLayer.clearLayers();
  liveSegment = null;
  liveMarker = null;
  const positions = current.points.filter(validPos).map((p) => [p.lat, p.lon]);
  if (positions.length) {
    liveMap.fitBounds(L.latLngBounds(positions), { padding: [30, 30], maxZoom: 15 });
  }
  return true;
}
function resetLiveEcoAlert() {
  liveAccelerationDirection = 0;
  liveAccelerationStreak = 0;
  liveEcoAlertDirection = 0;
  liveEcoAlertRemaining = 0;
  $('liveEcoAlert').hidden = true;
  $('liveEcoAlert').textContent = '';
}
function updateLiveEcoAlert(point, previous) {
  const alert = $('liveEcoAlert');
  if (liveEcoAlertRemaining > 0) {
    liveEcoAlertRemaining--;
    if (liveEcoAlertRemaining === 0) {
      alert.hidden = true;
      liveEcoAlertDirection = 0;
    }
  }
  const elapsedSec = previous ? point.sec - previous.sec : 0;
  const validSpeed =
    previous &&
    !point.break_before &&
    point.speed != null &&
    previous.speed != null &&
    Number.isFinite(Number(point.speed)) &&
    Number.isFinite(Number(previous.speed)) &&
    Number.isFinite(elapsedSec) &&
    elapsedSec > 0;
  const acceleration = validSpeed
    ? (Number(point.speed) - Number(previous.speed)) / elapsedSec
    : null;
  const direction =
    acceleration == null
      ? 0
      : acceleration > LIVE_ACCELERATION_THRESHOLD
        ? 1
        : acceleration < -LIVE_ACCELERATION_THRESHOLD
          ? -1
          : 0;
  if (!direction) {
    liveAccelerationDirection = 0;
    liveAccelerationStreak = 0;
    return;
  }
  if (direction === liveAccelerationDirection) {
    liveAccelerationStreak++;
  } else {
    liveAccelerationDirection = direction;
    liveAccelerationStreak = 1;
    liveEcoAlertDirection = 0;
  }
  if (liveAccelerationStreak < 3 || liveEcoAlertDirection === direction) return;
  alert.textContent =
    direction > 0
      ? '省油提醒：請平順踩油門，避免急加速。'
      : '省油提醒：請平順踩煞車，避免急煞車。';
  alert.hidden = false;
  liveEcoAlertDirection = direction;
  liveEcoAlertRemaining = 6;
}
function renderLivePoint(index) {
  const p = current.points[index],
    previous = index > 0 ? current.points[index - 1] : null;
  updateLiveEcoAlert(p, previous);
  const values = [
    ['車速', p.speed, 'km/h', 1, 'speed'],
    ['引擎轉速', p.rpm, 'rpm', 0, 'rpm'],
    ['引擎負載', p.load, '%', 1, 'load'],
    ['冷卻水溫', p.temp, '°C', 1, 'temp'],
    ['電瓶電壓', p.battery, 'V', 2, 'battery'],
    ['油箱油量', p.fuel_level, '%', 1, 'fuel_level'],
    ['累積里程', p.distance, 'km', 2, 'distance'],
    ['累積耗油', p.used, 'L', 2, 'used'],
  ];
  $('liveSensors').innerHTML = values
    .map(
      ([label, value, unit, decimals, key]) => {
        const previousValue = previous ? previous[key] : null;
        const elapsedSec = previous ? p.sec - previous.sec : 0;
        const comparable =
          value != null &&
          previousValue != null &&
          Number.isFinite(Number(value)) &&
          Number.isFinite(Number(previousValue)) &&
          Number.isFinite(elapsedSec) &&
          elapsedSec > 0;
        const delta = comparable ? Number(value) - Number(previousValue) : null;
        const percentChange =
          delta == null || Number(previousValue) === 0
            ? null
            : (delta / Math.abs(Number(previousValue))) * 100;
        const trendClass =
          percentChange == null ? '' : percentChange > 0 ? 'increase' : percentChange < 0 ? 'decrease' : 'steady';
        const arrow =
          percentChange == null ? '' : percentChange > 0 ? '▲' : percentChange < 0 ? '▼' : '→';
        const rate =
          percentChange == null
            ? '—'
            : `<span class="live-change-arrow" aria-hidden="true">${arrow}</span> ${fmt(Math.abs(percentChange), 1)}%`;
        return `<div class="live-sensor"><span>${esc(label)}</span><strong>${fmt(value, decimals)} <small>${esc(unit)}</small></strong><div class="live-sensor-change"><small class="live-change-value ${trendClass}">${rate}</small></div></div>`;
      },
    )
    .join('');
  $('liveTimestamp').textContent = `${p.time} · 觀測點 ${index + 1}`;
  $('liveProgressBar').style.width = `${((index + 1) / current.points.length) * 100}%`;
  if (!validPos(p)) {
    liveSegment = null;
    return;
  }
  if (!liveSegment || p.break_before || (previous && !validPos(previous))) {
    liveSegment = L.polyline([], { color: '#297a72', weight: 5, opacity: 0.9 }).addTo(
      liveRouteLayer,
    );
  }
  liveSegment.addLatLng([p.lat, p.lon]);
  if (!liveMarker) {
    liveMarker = L.circleMarker([p.lat, p.lon], {
      radius: 9,
      color: '#102e37',
      weight: 3,
      fillColor: '#ffffff',
      fillOpacity: 1,
    }).addTo(liveMap);
  } else {
    liveMarker.setLatLng([p.lat, p.lon]);
  }
  liveMap.panTo([p.lat, p.lon], { animate: false });
}
function setLiveRunning(running, status) {
  liveRunning = running;
  $('liveStatus').textContent = status;
  $('liveStatus').classList.toggle('running', running);
  $('liveToggle').textContent = running
    ? '❚❚ 暫停任務'
    : liveActive
      ? '▶ 繼續任務'
      : '↻ 重新開始';
  $('liveStop').disabled = !liveActive;
  document.querySelector('.sensor-pulse').classList.toggle('running', running);
}
function startLiveMission() {
  if (!current || !current.points.length) {
    const message = '目前沒有可重播的行程資料，請先選取一趟行程。';
    $('liveStatus').textContent = '無行程資料';
    error(Error(message));
    return;
  }
  if (liveActive) {
    if (!liveRunning) {
      setLiveRunning(true, '模擬重播中');
      liveTimer = setInterval(advanceLiveMission, 500);
    }
    return;
  }
  $('error').hidden = true;
  try {
    resetLiveEcoAlert();
    if (!ensureLiveMap()) {
      $('liveStatus').textContent = '地圖無法啟動';
      return;
    }
    liveActive = true;
    liveIndex = 0;
    $('liveTrip').textContent = `${current.vehicle} · ${current.journey}`;
    renderLivePoint(liveIndex);
    setLiveRunning(true, '模擬重播中');
    liveTimer = setInterval(advanceLiveMission, 500);
  } catch (e) {
    stopLiveMission();
    $('liveStatus').textContent = '啟動失敗';
    error(e);
  }
}
function advanceLiveMission() {
  if (liveIndex >= current.points.length - 1) {
    clearInterval(liveTimer);
    liveTimer = null;
    liveActive = false;
    setLiveRunning(false, '重播完成');
    return;
  }
  liveIndex++;
  renderLivePoint(liveIndex);
}
function stopLiveMission() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  resetLiveEcoAlert();
  if (!liveActive && !liveRunning) return;
  liveActive = false;
  setLiveRunning(false, '任務已結束');
}
function updateCursor(i) {
  if (!current) return;
  i = Math.min(current.points.length - 1, Math.max(0, Number(i)));
  cursorIndex = i;
  const p = current.points[i];
  $('cursor').value = i;
  $('cursorTime').textContent = `${p.time} · ${fmt(p.sec / 60)} 分`;
  const values = [
    ['CAN車速', p.speed, 'km/h'],
    ['引擎轉速', p.rpm, 'rpm'],
    ['引擎負載', p.load, '%'],
    ['累積耗油', p.used, 'L'],
    ['累積距離', p.distance, 'km'],
    ['冷卻水溫', p.temp, '°C'],
  ];
  $('pointInfo').innerHTML = values
    .map(([t, v, u]) => `<p>${t}<b>${fmt(v)} <small>${u}</small></b></p>`)
    .join('');
  if (map && validPos(p)) {
    if (!cursorMarker)
      cursorMarker = L.circleMarker([p.lat, p.lon], {
        radius: 9,
        color: '#102e37',
        weight: 3,
        fillColor: 'white',
        fillOpacity: 1,
      }).addTo(map);
    else cursorMarker.setLatLng([p.lat, p.lon]);
    cursorMarker.setStyle({ opacity: 1, fillOpacity: 1 });
  } else if (cursorMarker) cursorMarker.setStyle({ opacity: 0, fillOpacity: 0 });
  if (window.Plotly && $('timeline').data) {
    Plotly.relayout('timeline', {
      shapes: [
        {
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: p.sec / 60,
          x1: p.sec / 60,
          y0: 0,
          y1: 1,
          line: { color: '#d58d29', width: 1, dash: 'dot' },
        },
      ],
    });
  }
}
function hookChart(id) {
  const el = $(id);
  if (!el.on) return;
  el.removeAllListeners('plotly_click');
  el.on('plotly_click', (ev) => {
    const i = ev.points?.[0]?.customdata;
    if (Number.isInteger(i)) updateCursor(i);
  });
}
function chartArrays(key, xkey = 'sec') {
  const p = current.points;
  let x = [],
    y = [],
    custom = [];
  for (let i = 0; i < p.length; i++) {
    if (i && p[i].sec - p[i - 1].sec > 120) {
      x.push(null);
      y.push(null);
      custom.push(null);
    }
    x.push(xkey === 'sec' ? p[i].sec / 60 : p[i][xkey]);
    y.push(p[i][key]);
    custom.push(i);
  }
  return { x, y, customdata: custom };
}
async function renderTimeline() {
  const fuel = {
    ...chartArrays('used'),
    name: '累積耗油 L',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#a98550', shape: 'hv' },
    yaxis: 'y2',
  };
  const speed = {
    ...chartArrays('speed'),
    name: 'CAN速度 km/h',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#718391', width: 2 },
  };
  await drawPlot('timeline', [speed, fuel], {
    showlegend: true,
    legend: { orientation: 'h', y: 1.13 },
    xaxis: { title: '時間（分鐘）', gridcolor: '#e6e9eb' },
    yaxis: { title: 'CAN車速 km/h', gridcolor: '#e6e9eb', rangemode: 'tozero' },
    yaxis2: {
      title: '累積耗油 L',
      overlaying: 'y',
      side: 'right',
      rangemode: 'tozero',
      showgrid: false,
    },
    margin: { l: 55, r: 55, t: 35, b: 45 },
  });
  hookChart('timeline');
  $('windowSummary').textContent = '顯示全程。可輸入區間檢查燃油與里程增量。';
}
function windowApply() {
  if (!current) return;
  const lo = Number($('windowStart').value) * 60,
    hi = Number($('windowEnd').value) * 60;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < 0 || hi <= lo) {
    alert('請輸入有效起終時間，終點須大於起點。');
    return;
  }
  const pts = current.points.filter((p) => p.sec >= lo && p.sec <= hi);
  if (pts.length < 2) {
    $('windowSummary').textContent = '區間內不足兩筆觀測，無法計算增量。';
    return;
  }
  const first = pts[0],
    last = pts.at(-1);
  const fuel = first.used != null && last.used != null ? last.used - first.used : null,
    dist = first.distance != null && last.distance != null ? last.distance - first.distance : null;
  let idle = 0;
  for (let i = 0; i < pts.length - 1; i++) if (pts[i].idle) idle += pts[i].dt;
  $('windowSummary').textContent =
    `實際觀測區間 ${fmt(first.sec / 60)}–${fmt(last.sec / 60)} 分｜${pts.length} 筆｜里程增量 ${fmt(dist, 2)} km｜燃油增量 ${fmt(fuel, 2)} L｜停車引擎運轉 ${fmt(idle / 60)} 分。區間燃油受計數器刻度限制。`;
  if (window.Plotly) Plotly.relayout('timeline', { 'xaxis.range': [lo / 60, hi / 60] });
  updateCursor(first.index);
}
async function renderSignals() {
  const metrics = [
    ['used', '累積耗油', 'L', '#a98550'],
    ['speed', 'CAN 車速', 'km/h', '#718391'],
    ['rpm', '引擎轉速', 'rpm', '#777f8d'],
    ['load', '當前速度引擎負載', '%', '#71858d'],
    ['temp', '冷卻液溫度', '°C', '#ad7c6d'],
    ['battery', '電瓶電壓', 'V', '#688296'],
    ['fuel_level', '油量百分比', '%', '#897e91'],
    ['accel', '區間平均加速度（≤30秒）', 'm/s²', '#687d86'],
  ];
  const xkey = $('xAxis').value;
  $('signalCharts').innerHTML = metrics
    .map(
      (m, i) =>
        `<div><h3>${m[1]} <small class="muted">${m[2]}</small></h3><div id="sig${i}" class="chart"></div></div>`,
    )
    .join('');
  for (let i = 0; i < metrics.length; i++) {
    const [key, title, unit, col] = metrics[i],
      available = current.points.some((p) => p[key] != null);
    if (!available) {
      $('sig' + i).innerHTML = '<p class="notice">本趟無可用資料。</p>';
      continue;
    }
    await drawPlot(
      'sig' + i,
      [
        {
          ...chartArrays(key, xkey),
          type: 'scatter',
          mode: xkey === 'distance' && key !== 'used' ? 'markers' : 'lines',
          marker: { color: col, size: 4, opacity: 0.65 },
          line: {
            color: col,
            width: 1.5,
            shape: key === 'used' && xkey === 'sec' ? 'hv' : 'linear',
          },
          name: title,
        },
      ],
      {
        xaxis: { title: xkey === 'sec' ? '時間（分鐘）' : '累積里程（km）', gridcolor: '#e6e9eb' },
        yaxis: { title: unit, gridcolor: '#e6e9eb' },
        margin: { l: 55, r: 12, t: 10, b: 48 },
      },
    );
    hookChart('sig' + i);
  }
  await drawPlot(
    'speedBands',
    [
      {
        type: 'bar',
        x: current.bands.map((x) => x.band),
        y: current.bands.map((x) => x.pct),
        marker: { color: '#718391' },
      },
    ],
    { yaxis: { title: '可判讀時間占比 %' }, xaxis: { title: '速度 km/h' } },
  );
  await drawPlot(
    'engineScatter',
    [
      {
        type: 'scatter',
        mode: 'markers',
        x: current.points.map((p) => p.rpm),
        y: current.points.map((p) => p.load),
        customdata: current.points.map((p) => p.index),
        marker: {
          size: 5,
          color: current.points.map((p) => p.speed ?? 0),
          colorscale: 'Viridis',
          showscale: true,
          colorbar: { title: 'km/h' },
          opacity: 0.65,
        },
      },
    ],
    {
      xaxis: { title: '轉速 rpm' },
      yaxis: { title: '負載 %' },
      margin: { l: 50, r: 60, t: 15, b: 45 },
    },
  );
  hookChart('engineScatter');
}
function renderEvents() {
  const source = $('eventSource').value;
  let events = [];
  if (source !== 'derived') events.push(...current.events);
  if (source !== 'raw') events.push(...current.stops);
  events.sort((a, b) => String(a.start).localeCompare(String(b.start)));
  table(
    'eventTable',
    ['時間', '事件', '來源', '位置對時差', '詳情'],
    events.map(
      (e) =>
        `<tr data-point="${e.index}"><td>${esc(e.start)}</td><td>${esc(e.type)}</td><td>${esc(e.source)}</td><td>${e.source === 'CAN推估' ? '觀測區間' : fmt(e.location_offset_sec, 0) + ' 秒'}</td><td>${esc(e.info ? JSON.stringify(e.info) : `${fmt(e.seconds / 60)} 分鐘`)}</td></tr>`,
    ),
  );
  table(
    'stopTable',
    ['開始', '結束', '分鐘', '緯度', '經度'],
    current.stops.map(
      (s) =>
        `<tr data-point="${s.index}"><td>${esc(s.start)}</td><td>${esc(s.end)}</td><td>${fmt(s.seconds / 60)}</td><td>${fmt(s.lat, 6)}</td><td>${fmt(s.lon, 6)}</td></tr>`,
    ),
  );
}
function renderQuality() {
  const q = current.quality,
    s = current.summary;
  const items = [
    ['無效時間', q.invalid_time_rows + ' 筆'],
    ['重複時間', q.duplicate_timestamps + ' 筆'],
    ['GPS缺失／無效', q.missing_gps_rows + ' 筆'],
    ['GPS跳點', q.gps_jump_segments + ' 段'],
    ['CAN異常／未知', q.abnormal_can_rows + ' 筆'],
    ['最長取樣間隔', fmt(q.max_gap_sec) + ' 秒'],
    ['排除長空檔', fmt(q.gap_seconds / 60) + ' 分'],
    ['里程倒退', q.odo_reset ? '是／不計里程' : '未發現'],
    ['燃油倒退', q.fuel_reset ? '是／不計油耗' : '未發現'],
    ['最小正燃油增量', fmt(q.fuel_min_positive_step_l, 3) + ' L'],
    ['最小正里程增量', fmt(q.odo_min_positive_step_km, 3) + ' km'],
    ['車速可判讀時間', fmt(s.speed_known_sec / 60) + ' 分'],
    ['限速可比較時間', fmt(s.speed_limit_known_sec / 60) + ' 分'],
    ['觀測超限時間', s.speed_limit_known_sec ? fmt(s.overspeed_sec / 60) + ' 分' : '無可用限速'],
    ['CAN狀態欄位', q.can_status_available ? '有，僅採0' : '缺少，品質未驗證'],
  ];
  $('qualityInfo').innerHTML = items
    .map(([k, v]) => `<p><span class="muted">${esc(k)}</span><br><b>${esc(v)}</b></p>`)
    .join('');
}
async function loadRaw() {
  const id = tripId,
    page = rawPage;
  try {
    const d = await get(`/api/raw?id=${id}&page=${page}`);
    if (id !== tripId || page !== rawPage) return;
    const cols = catalog.meta.columns
      .concat(['source_row'])
      .filter((v, i, a) => a.indexOf(v) === i);
    table(
      'rawTable',
      cols,
      d.rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c])}</td>`).join('')}</tr>`),
    );
    $('rawPage').textContent =
      `第 ${page + 1} / ${Math.max(1, Math.ceil(d.total / 100))} 頁 · 共 ${d.total} 筆`;
    $('rawPrev').disabled = page === 0;
    $('rawNext').disabled = (page + 1) * 100 >= d.total;
  } catch (e) {
    error(e);
  }
}
function hav(a, b, c, d) {
  if ([a, b, c, d].some((v) => !Number.isFinite(v))) return Infinity;
  const rad = Math.PI / 180,
    p = (c - a) * rad,
    l = (d - b) * rad;
  return (
    12742000 *
    Math.asin(
      Math.sqrt(
        Math.min(
          1,
          Math.sin(p / 2) ** 2 + Math.cos(a * rad) * Math.cos(c * rad) * Math.sin(l / 2) ** 2,
        ),
      ),
    )
  );
}
function comparisonTimestamp(value) {
  if (typeof value !== 'string') return null;
  const match = value.trim().match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})?$/i,
  );
  if (!match) return null;
  const [, year, month, day, hour, minute, second, fraction = '', zone = ''] = match;
  const parts = [year, month, day, hour, minute, second].map(Number);
  const localTimestamp =
    Date.UTC(parts[0], parts[1] - 1, parts[2], parts[3], parts[4], parts[5]) +
    (fraction ? Number(`0.${fraction}`) * 1000 : 0);
  const check = new Date(localTimestamp);
  if (
    check.getUTCFullYear() !== parts[0] ||
    check.getUTCMonth() !== parts[1] - 1 ||
    check.getUTCDate() !== parts[2] ||
    check.getUTCHours() !== parts[3] ||
    check.getUTCMinutes() !== parts[4] ||
    check.getUTCSeconds() !== parts[5]
  )
    return null;
  if (!zone || zone.toUpperCase() === 'Z') return localTimestamp;
  const offsetMatch = zone.match(/^([+-])(\d{2}):(\d{2})$/);
  if (!offsetMatch || Number(offsetMatch[2]) > 23 || Number(offsetMatch[3]) > 59) return null;
  const offsetMinutes = Number(offsetMatch[2]) * 60 + Number(offsetMatch[3]);
  return localTimestamp - (offsetMatch[1] === '+' ? 1 : -1) * offsetMinutes * 60_000;
}
function hasValidTripEconomy(trip) {
  return (
    Number.isFinite(trip.distance_km) &&
    trip.distance_km > 0 &&
    Number.isFinite(trip.fuel_l) &&
    trip.fuel_l >= 0 &&
    Number.isFinite(trip.l100) &&
    trip.l100 >= 0
  );
}
function isHistoricalReference(trip, selectedTripId, selectedStart) {
  if (trip.id === selectedTripId) return false;
  const tripStart = comparisonTimestamp(trip.start);
  const tripEnd = comparisonTimestamp(trip.end);
  return (
    selectedStart != null &&
    tripStart != null &&
    tripEnd != null &&
    tripStart <= tripEnd &&
    tripEnd < selectedStart &&
    hasValidTripEconomy(trip)
  );
}
function percentileLinear(values, probability) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  // Use linear interpolation between adjacent sorted values, consistent for Q1, median, and Q3.
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
function summarizeComparison(values, currentL100) {
  const n = values.length;
  const median = percentileLinear(values, 0.5);
  const rawDifference =
    n >= MIN_COMPARE_REFERENCE_TRIPS &&
    Number.isFinite(currentL100) &&
    median > 0
      ? (currentL100 / median - 1) * 100
      : null;
  return {
    n,
    median,
    q1: percentileLinear(values, 0.25),
    q3: percentileLinear(values, 0.75),
    difference: Number.isFinite(rawDifference) ? rawDifference : null,
  };
}
function comparisonStatusMessage(currentValid, n, median, difference, reason) {
  if (!currentValid) return `本趟資料不足，無法比較：${reason}`;
  if (n < MIN_COMPARE_REFERENCE_TRIPS)
    return `參考資料不足：目前有 ${n} 趟有效歷史行程；少於 ${MIN_COMPARE_REFERENCE_TRIPS} 趟，不顯示相對差異。`;
  if (median === 0) return '歷史中位數為 0 L/100 km，無法計算本趟相對差異。';
  if (!Number.isFinite(difference)) return '無法計算本趟相對差異。';
  if (difference === 0) return '本趟與歷史中位數相同（0.0%）。';
  return `本趟${difference > 0 ? '高於' : '低於'}歷史中位數 ${fmt(Math.abs(difference), 1)}%。`;
}
function comparisonFailureReason() {
  const summary = current.summary;
  const quality = current.quality;
  const points = current.points;
  const reasons = [];
  if (quality.fuel_reset) reasons.push('累積耗油計數器倒退');
  else if (
    !Number.isFinite(summary.fuel_l) &&
    (!Number.isFinite(points[0]?.fuel) || !Number.isFinite(points.at(-1)?.fuel))
  )
    reasons.push('累積耗油首末值缺失或未通過 CAN 品質判斷');
  if (quality.odo_reset) reasons.push('累積里程計數器倒退');
  else if (
    !Number.isFinite(summary.distance_km) &&
    (!Number.isFinite(points[0]?.odo) || !Number.isFinite(points.at(-1)?.odo))
  )
    reasons.push('累積里程首末值缺失或未通過 CAN 品質判斷');
  if (Number.isFinite(summary.distance_km) && summary.distance_km <= 0)
    reasons.push('有效行駛距離未大於 0');
  if (Number.isFinite(summary.fuel_l) && summary.fuel_l < 0)
    reasons.push('有效累積耗油差小於 0');
  if (!reasons.length) reasons.push('既有油耗或里程品質判斷未提供有效指標');
  return reasons.join('；');
}
function compareCandidateStatus(trip, selectedStart) {
  if (trip.id === tripId) return '本趟，不納入基準';
  const tripStart = comparisonTimestamp(trip.start);
  const tripEnd = comparisonTimestamp(trip.end);
  if (selectedStart == null || tripStart == null || tripEnd == null || tripStart > tripEnd)
    return '時間缺失或無法比較';
  if (tripEnd >= selectedStart) return '非本趟之前的歷史行程';
  if (!hasValidTripEconomy(trip)) return '油耗／里程指標無效';
  return '納入歷史基準';
}
function renderCompare() {
  const s = current.summary;
  const radius = Number($('odRadius').value);
  const selectedStart = comparisonTimestamp(s.start);
  peers = catalog.trips
    .filter(
      (t) =>
        t.vehicle === current.vehicle &&
        typeof t.start === 'string' &&
        withinDate(t) &&
        hav(s.start_lat, s.start_lon, t.start_lat, t.start_lon) <= radius &&
        hav(s.end_lat, s.end_lon, t.end_lat, t.end_lon) <= radius,
    )
    .sort((a, b) => String(b.start).localeCompare(String(a.start)));
  const references = peers.filter((t) => isHistoricalReference(t, tripId, selectedStart));
  const values = references.map((t) => t.l100);
  const currentValid = hasValidTripEconomy(s);
  const { n, median, q1, q3, difference } = summarizeComparison(
    values,
    currentValid ? s.l100 : null,
  );
  const dateFrom = $('dateFrom').value || '不限';
  const dateTo = $('dateTo').value || '不限';
  const historyTimeNote =
    selectedStart == null
      ? '本趟開始時間無法可靠比較，不建立歷史基準。'
      : '只使用結束時間早於本趟開始時間的歷史行程。';
  $('peerNote').textContent =
    `同一車輛：${current.vehicle} · 起終點篩選半徑：${radius.toLocaleString('zh-TW')} m · 日期範圍：${dateFrom} 至 ${dateTo} · ${historyTimeNote}`;
  $('compareSummary').innerHTML =
    card('本趟油耗', fmt(currentValid ? s.l100 : null, 2), 'L/100 km') +
    card('有效歷史參考行程', n, '趟', `介面參考門檻 ${MIN_COMPARE_REFERENCE_TRIPS} 趟`) +
    card('歷史中位數', fmt(median, 2), 'L/100 km') +
    card('歷史四分位範圍', n ? `${fmt(q1, 2)}–${fmt(q3, 2)}` : '—', 'L/100 km');
  $('compareStatus').textContent = comparisonStatusMessage(
    currentValid,
    n,
    median,
    difference,
    currentValid ? '' : comparisonFailureReason(),
  );
  const currentPoint = currentValid ? [s] : [];
  const chartData = [
    {
      type: 'scatter',
      mode: 'markers',
      name: '有效歷史參考行程',
      x: references.map((t) => (Number.isFinite(t.idle_sec) ? t.idle_sec / 60 : null)),
      y: references.map((t) => t.l100),
      text: references.map((t) => `${t.journey} · 納入歷史基準`),
      customdata: references.map((t) => t.id),
      marker: { color: '#718391', size: 8 },
      hovertemplate:
        '%{text}<br>停車引擎運轉 %{x:.1f} 分鐘<br>油耗 %{y:.2f} L/100 km<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'markers',
      name: '本趟（不納入基準）',
      x: currentPoint.map((t) => (Number.isFinite(t.idle_sec) ? t.idle_sec / 60 : null)),
      y: currentPoint.map((t) => t.l100),
      text: currentPoint.map(() => `${current.journey} · 本趟，不納入基準`),
      customdata: currentPoint.map(() => tripId),
      marker: { color: '#b9853d', size: 13, symbol: 'diamond' },
      hovertemplate:
        '%{text}<br>停車引擎運轉 %{x:.1f} 分鐘<br>油耗 %{y:.2f} L/100 km<extra></extra>',
    },
  ];
  drawPlot(
    'compareChart',
    chartData,
    {
      showlegend: true,
      legend: { orientation: 'h', y: 1.13 },
      xaxis: { title: '停車引擎運轉時間（分鐘）' },
      yaxis: { title: '每百公里耗油（L/100 km）' },
      shapes:
        median == null
          ? []
          : [
              {
                type: 'line',
                xref: 'paper',
                x0: 0,
                x1: 1,
                yref: 'y',
                y0: median,
                y1: median,
                line: { color: '#b9853d', width: 2, dash: 'dash' },
              },
            ],
      annotations:
        median == null
          ? []
          : [
              {
                xref: 'paper',
                x: 0.99,
                yref: 'y',
                y: median,
                text: `歷史中位數 ${fmt(median, 2)}（${n} 趟）`,
                showarrow: false,
                xanchor: 'right',
                yshift: 10,
                bgcolor: 'rgba(255,255,255,0.8)',
              },
            ],
    },
  ).then(() => {
    const el = $('compareChart');
    if (el.on) {
      el.removeAllListeners('plotly_click');
      el.on('plotly_click', (ev) => openTrip(ev.points[0].customdata));
    }
  });
  table(
    'peerTable',
    ['行程', '開始', '距離 km', '耗油 L', 'L/100 km', '基準狀態', '停車引擎運轉 min'],
    peers.map((t) => tripRow(t, compareCandidateStatus(t, selectedStart))),
  );
}
function tripRow(t, comparisonStatus) {
  return `<tr data-id="${t.id}"><td>${esc(t.journey)}</td><td>${esc(t.start)}</td><td>${fmt(t.distance_km, 2)}</td><td>${fmt(t.fuel_l, 2)}</td><td>${fmt(t.l100, 2)}</td><td>${esc(comparisonStatus)}</td><td>${fmt(Number.isFinite(t.idle_sec) ? t.idle_sec / 60 : null)}</td></tr>`;
}
function renderFleet() {
  const all = catalog.trips.filter(withinDate),
    usable = all.filter((t) => t.distance_km > 0 && t.fuel_l != null);
  const dist = sum(usable, 'distance_km'),
    fuel = sum(usable, 'fuel_l');
  $('fleetCards').innerHTML =
    card('車輛數', new Set(all.map((t) => t.vehicle)).size) +
    card('行程數', all.length) +
    card('有效配對里程', fmt(dist, 0), 'km') +
    card('有效配對耗油', fmt(fuel, 1), 'L') +
    card(
      '整體油耗',
      fmt(dist ? (fuel / dist) * 100 : null, 2),
      'L/100km',
      `耗油與里程皆有效 ${usable.length} 趟`,
    );
  const by = {};
  all.forEach((t) => {
    if (!by[t.vehicle]) by[t.vehicle] = { plate: t.plate, trips: 0, fuel: 0, valid: 0 };
    by[t.vehicle].trips++;
    if (t.fuel_l != null) {
      by[t.vehicle].fuel += t.fuel_l;
      by[t.vehicle].valid++;
    }
  });
  const v = Object.values(by);
  drawPlot(
    'fleetChart',
    [
      {
        type: 'bar',
        x: v.map((t) => t.plate),
        y: v.map((t) => (t.valid ? t.fuel : null)),
        text: v.map((t) => `${t.trips} 趟 / 有效耗油 ${t.valid} 趟`),
        marker: { color: '#718391' },
      },
    ],
    {
      yaxis: { title: '有有效量測的累積耗油 L' },
      xaxis: { title: '車輛' },
      margin: { l: 60, r: 10, t: 15, b: 80 },
    },
  );
  const search = $('fleetSearch').value.trim().toLowerCase();
  fleetRows = all
    .filter((t) => `${t.plate} ${t.vehicle} ${t.journey}`.toLowerCase().includes(search))
    .sort((a, b) => b.start.localeCompare(a.start));
  fleetPage = Math.min(fleetPage, Math.max(0, Math.ceil(fleetRows.length / 50) - 1));
  renderFleetTable();
}
function renderFleetTable() {
  const rows = fleetRows.slice(fleetPage * 50, (fleetPage + 1) * 50);
  table(
    'fleetTable',
    ['車輛', '行程', '開始', '距離 km', '耗油 L', 'L/100km', '品質問題類數'],
    rows.map(
      (t) =>
        `<tr data-id="${t.id}"><td>${esc(t.plate)}</td><td>${esc(t.journey)}</td><td>${esc(t.start)}</td><td>${fmt(t.distance_km, 2)}</td><td>${fmt(t.fuel_l, 2)}</td><td>${fmt(t.l100, 2)}</td><td>${t.quality_flags}</td></tr>`,
    ),
  );
  $('fleetCount').textContent = `${fleetRows.length} 趟`;
  $('fleetPage').textContent =
    `${fleetPage + 1} / ${Math.max(1, Math.ceil(fleetRows.length / 50))}`;
  $('fleetPrev').disabled = !fleetPage;
  $('fleetNext').disabled = (fleetPage + 1) * 50 >= fleetRows.length;
}
function openTrip(id) {
  const t = catalog.trips.find((t) => t.id === Number(id));
  if (!t) return;
  $('vehicle').value = t.vehicle;
  fillJourneys();
  $('journey').value = t.id;
  setTab('trip');
  loadSelected();
}
function stopPlay() {
  if (playing) clearInterval(playing);
  playing = null;
  $('play').textContent = '▶ 逐筆播放';
}
async function init() {
  refreshLiveAiStatus();
  try {
    catalog = await get('/api/catalog');
    const v = new Map();
    catalog.trips.forEach((t) => v.set(t.vehicle, t.plate));
    $('vehicle').innerHTML = [...v]
      .sort((a, b) => a[1].localeCompare(b[1]))
      .map(([k, p]) => `<option value="${esc(k)}">${esc(p)} · ${esc(k)}</option>`)
      .join('');
    $('source').textContent =
      `${catalog.meta.source} · ${fmt(catalog.meta.rows, 0)} 筆 · ${catalog.meta.columns.length} 欄`;
    const dates = catalog.trips.map((t) => t.start.slice(0, 10)).sort();
    if (!dates.length) throw Error('沒有可分析行程，請查看匯入紀錄。');
    $('dateFrom').value = dates[0];
    $('dateTo').value = dates.at(-1);
    // Prefer the previously studied vehicle when present; otherwise first vehicle.
    if (v.has('AHMPUL0C13')) $('vehicle').value = 'AHMPUL0C13';
    fillJourneys();
    renderFleet();
    await loadSelected();
  } catch (e) {
    error(e);
  }
}
$('vehicle').onchange = () => {
  fillJourneys();
  loadSelected();
};
$('journey').onchange = loadSelected;
$('refresh').onclick = () => {
  if ($('dateFrom').value > $('dateTo').value) {
    alert('起始日期不可晚於結束日期');
    return;
  }
  fillJourneys();
  fleetPage = 0;
  renderFleet();
  loadSelected();
};
document.querySelectorAll('nav button').forEach((b) => {
  b.onclick = () => {
    if (b.dataset.tab === 'live' && activeTab !== 'live') liveReturnTab = activeTab;
    setTab(b.dataset.tab);
    if (b.dataset.tab === 'live') startLiveMission();
  };
});
$('liveBack').onclick = () => setTab(liveReturnTab);
$('liveToggle').onclick = () => {
  if (liveRunning) {
    clearInterval(liveTimer);
    liveTimer = null;
    setLiveRunning(false, '重播已暫停');
  } else {
    startLiveMission();
  }
};
$('liveStop').onclick = stopLiveMission;
$('liveAiForm').addEventListener('submit', submitLiveAiQuestion);
$('liveAiInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('liveAiForm').requestSubmit();
  }
});
$('cursor').oninput = (e) => {
  stopPlay();
  updateCursor(e.target.value);
};
$('play').onclick = () => {
  if (playing) {
    stopPlay();
    return;
  }
  if (!current) return;
  if (cursorIndex >= current.points.length - 1) updateCursor(0);
  $('play').textContent = '❚❚ 暫停';
  playing = setInterval(() => {
    if (cursorIndex >= current.points.length - 1) {
      stopPlay();
      return;
    }
    updateCursor(cursorIndex + 1);
  }, 350);
};
$('applyWindow').onclick = windowApply;
$('resetWindow').onclick = () => {
  $('windowStart').value = 0;
  $('windowEnd').value = (current.summary.duration_sec / 60).toFixed(2);
  renderTimeline();
};
$('mapMetric').onchange = () => {
  renderMap();
  updateCursor(cursorIndex);
};
$('xAxis').onchange = renderSignals;
$('eventSource').onchange = renderEvents;
$('odRadius').onchange = renderCompare;
$('rawPrev').onclick = () => {
  rawPage = Math.max(0, rawPage - 1);
  loadRaw();
};
$('rawNext').onclick = () => {
  rawPage++;
  loadRaw();
};
$('fleetSearch').oninput = () => {
  fleetPage = 0;
  renderFleet();
};
$('fleetPrev').onclick = () => {
  fleetPage--;
  renderFleetTable();
};
$('fleetNext').onclick = () => {
  fleetPage++;
  renderFleetTable();
};
document.addEventListener('click', (e) => {
  const row = e.target.closest('tr[data-id],tr[data-point]');
  if (!row) return;
  if (row.dataset.id) openTrip(row.dataset.id);
  else {
    setTab('trip');
    updateCursor(Number(row.dataset.point));
    if (map && validPos(current.points[Number(row.dataset.point)]))
      map.panTo([
        current.points[Number(row.dataset.point)].lat,
        current.points[Number(row.dataset.point)].lon,
      ]);
  }
});
init();
