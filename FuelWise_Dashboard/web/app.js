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
  activeTab = 'trip';
let peers = [],
  fleetRows = [];
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
function error(e) {
  $('error').textContent = e.message || String(e);
  $('error').hidden = false;
  $('loading').hidden = true;
}
function card(title, value, unit = '', note = '') {
  return `<div class="card"><span>${esc(title)}</span><strong>${value} <small>${esc(unit)}</small></strong><small>${esc(note)}</small></div>`;
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
  document
    .querySelectorAll('.tab')
    .forEach((e) => (e.hidden = e.id !== name || (name !== 'fleet' && !current)));
  document
    .querySelectorAll('nav button')
    .forEach((e) => e.classList.toggle('active', e.dataset.tab === name));
  $('title').textContent = document
    .querySelector(`nav [data-tab="${name}"]`)
    .textContent.slice(3)
    .trim();
  setTimeout(() => {
    if (map) map.invalidateSize();
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
  const id = Number($('journey').value);
  if (!id) {
    ++requestNumber;
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
      card('本趟行駛距離', fmt(s.distance_km, 2), 'km', '累積里程差') +
      card('本趟耗油', fmt(s.fuel_l, 2), 'L', '累積燃油差') +
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
  if (window.Plotly && $('timeline').data)
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
function renderCompare() {
  const s = current.summary,
    r = Number($('odRadius').value);
  peers = catalog.trips.filter(
    (t) =>
      t.vehicle === current.vehicle &&
      withinDate(t) &&
      hav(s.start_lat, s.start_lon, t.start_lat, t.start_lon) <= r &&
      hav(s.end_lat, s.end_lon, t.end_lat, t.end_lon) <= r,
  );
  const ref = peers.filter((t) => t.id !== tripId && t.fuel_l != null);
  const vals = ref.map((t) => t.fuel_l).sort((a, b) => a - b),
    n = vals.length,
    median = n ? (vals[Math.floor((n - 1) / 2)] + vals[Math.floor(n / 2)]) / 2 : null;
  $('peerNote').textContent =
    `符合OD ${peers.length} 趟；排除本趟後有效耗油樣本 ${n} 趟，中位耗油 ${fmt(median, 2)} L。比較僅描述歷史差異；載重、工作目的與中途路線未控制，不代表節油效果。`;
  drawPlot(
    'compareChart',
    [
      {
        type: 'scatter',
        mode: 'markers',
        x: peers.map((t) => t.idle_sec / 60),
        y: peers.map((t) => t.fuel_l),
        text: peers.map((t) => t.journey),
        customdata: peers.map((t) => t.id),
        marker: {
          color: peers.map((t) => (t.id === tripId ? '#b9853d' : '#718391')),
          size: peers.map((t) => (t.id === tripId ? 13 : 8)),
        },
        hovertemplate: '%{text}<br>停車引擎運轉 %{x:.1f} 分<br>耗油 %{y:.2f} L<extra></extra>',
      },
    ],
    { xaxis: { title: '停車引擎運轉（分鐘）' }, yaxis: { title: '本趟耗油（L）' } },
  ).then(() => {
    const el = $('compareChart');
    if (el.on) {
      el.removeAllListeners('plotly_click');
      el.on('plotly_click', (ev) => openTrip(ev.points[0].customdata));
    }
  });
  table(
    'peerTable',
    ['行程', '開始', '距離 km', '耗油 L', 'L/100km', '停車引擎運轉 min'],
    peers.map((t) => tripRow(t)),
  );
}
function tripRow(t) {
  return `<tr data-id="${t.id}"><td>${esc(t.journey)}</td><td>${esc(t.start)}</td><td>${fmt(t.distance_km, 2)}</td><td>${fmt(t.fuel_l, 2)}</td><td>${fmt(t.l100, 2)}</td><td>${fmt(t.idle_sec / 60)}</td></tr>`;
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
document.querySelectorAll('nav button').forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
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
