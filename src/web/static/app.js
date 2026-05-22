const statusEl = document.getElementById('status');
const map = L.map('map').setView([40.9067, 29.1550], 17);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

const gtMarker = L.circleMarker([0, 0], {
  radius: 7,
  color: '#5ad37a',
  fillColor: '#5ad37a',
  fillOpacity: 0.9
}).addTo(map);
gtMarker.bindTooltip('Lat: --\nLon: --', {
  direction: 'top',
  offset: [0, -8],
  opacity: 0.95,
  className: 'gt-tooltip'
});

const estMarker = L.marker([0, 0], {
  icon: L.divIcon({
    className: 'est-icon',
    html: '<div class="est-x"></div>',
    iconSize: [14, 14]
  })
}).addTo(map);

const estHistoryLayer = L.layerGroup().addTo(map);
const estHistory = [];

const gtTrail = L.polyline([], { color: '#5ad37a', weight: 2, opacity: 0.6 }).addTo(map);

const infoEst = document.getElementById('info-est');
const infoGt = document.getElementById('info-gt');
const infoError = document.getElementById('info-error');
const historyList = document.getElementById('history-list');
const pulseRing = document.getElementById('pulse-ring');
const toastStack = document.getElementById('toast-stack');

let lastGt = null;
let mapInitialized = false;

function updateGt(lat, lon) {
  lastGt = [lat, lon];
  gtMarker.setLatLng([lat, lon]);
  gtMarker.setTooltipContent(`Lat: ${lat.toFixed(6)}<br>Lon: ${lon.toFixed(6)}`);
  gtTrail.addLatLng([lat, lon]);
  if (!mapInitialized) {
    map.setView([lat, lon], 17);
    mapInitialized = true;
  }
}

function updateEst(lat, lon) {
  estMarker.setLatLng([lat, lon]);
}

function addEstimateHistory(lat, lon) {
  const marker = L.marker([lat, lon], {
    icon: L.divIcon({
      className: 'est-icon',
      html: '<div class="est-x"></div>',
      iconSize: [12, 12]
    })
  });
  marker.addTo(estHistoryLayer);
  estHistory.push(marker);
  if (estHistory.length > 5) {
    const oldest = estHistory.shift();
    if (oldest) {
      estHistoryLayer.removeLayer(oldest);
    }
  }
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const toRad = (value) => (value * Math.PI) / 180;
  const r = 6371000;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return r * c;
}

function updateLocalizationInfo(payload) {
  const estLat = payload.est_lat;
  const estLon = payload.est_lon;
  const gtLat = payload.gt_lat;
  const gtLon = payload.gt_lon;
  if (infoEst) {
    infoEst.textContent = `${estLat.toFixed(6)}, ${estLon.toFixed(6)}`;
  }
  if (infoGt) {
    infoGt.textContent = `${gtLat.toFixed(6)}, ${gtLon.toFixed(6)}`;
  }
  if (infoError) {
    const errorMeters = haversineMeters(estLat, estLon, gtLat, gtLon);
    infoError.textContent = `${errorMeters.toFixed(4)} m`;
  }

  if (historyList) {
    const errorMeters = haversineMeters(estLat, estLon, gtLat, gtLon);
    const row = document.createElement('div');
    row.className = 'history-item';
    row.innerHTML = `<strong>${errorMeters.toFixed(4)} m</strong><span>${estLat.toFixed(5)}, ${estLon.toFixed(5)}</span>`;
    historyList.prepend(row);
    while (historyList.children.length > 5) {
      historyList.removeChild(historyList.lastChild);
    }
  }
}

function setStatus(text, good) {
  statusEl.textContent = text;
  statusEl.style.borderColor = good ? '#2f5244' : '#3a2430';
  statusEl.style.background = good ? 'rgba(44, 120, 80, 0.25)' : 'rgba(120, 40, 60, 0.25)';
}

function showToast(message, tone = 'info') {
  if (!toastStack) {
    return;
  }
  const toast = document.createElement('div');
  toast.className = `toast ${tone}`;
  toast.textContent = message;
  toastStack.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 2600);
}

function pulseMap() {
  if (!pulseRing) {
    return;
  }
  pulseRing.classList.remove('active');
  void pulseRing.offsetWidth;
  pulseRing.classList.add('active');
  setTimeout(() => pulseRing.classList.remove('active'), 900);
}

async function postJson(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.visual_localization_running) {
      const carlaNote = data.carla_sim_running ? ' + CARLA' : '';
      setStatus(`Running${carlaNote}`, true);
    } else {
      const carlaNote = data.carla_sim_running ? 'CARLA only' : 'Stopped';
      setStatus(carlaNote, !!data.carla_sim_running);
    }
  } catch (err) {
    setStatus('Disconnected', false);
  }
}

function connectWs() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus('Connected', true);
  };

  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      if (payload.type === 'gt') {
        updateGt(payload.lat, payload.lon);
      }
      if (payload.type === 'localization') {
        updateGt(payload.gt_lat, payload.gt_lon);
        updateEst(payload.est_lat, payload.est_lon);
        addEstimateHistory(payload.est_lat, payload.est_lon);
        updateLocalizationInfo(payload);
        showToast('Localization complete', 'success');
        pulseMap();
      }
      if (payload.type === 'localization_started') {
        showToast('Localization started', 'info');
      }
    } catch (err) {
      console.warn('Bad payload', err);
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected', false);
    setTimeout(connectWs, 1200);
  };
}

function bindControls() {
  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const carlaBtn = document.getElementById('carla-btn');


  startBtn.addEventListener('click', async () => {
    const payload = {
      start_visual: true,
      start_localization_server: false,
      carla_host: document.getElementById('carla-host').value,
      carla_port: Number(document.getElementById('carla-port').value || 2000),
      loc_host: document.getElementById('loc-host').value,
      loc_port: Number(document.getElementById('loc-port').value || 5555),
      web_host: location.hostname,
      web_port: Number(location.port || 80),
      web_rate: Number(document.getElementById('web-rate').value || 6),
      no_preview_window: document.getElementById('no-preview').checked
    };
    await postJson('/api/start', payload);
    await refreshStatus();
  });

  stopBtn.addEventListener('click', async () => {
    await postJson('/api/stop', {});
    await refreshStatus();
  });

  carlaBtn.addEventListener('click', async () => {
    const payload = {
      carla_exe: document.getElementById('carla-exe').value,
      carla_args: document.getElementById('carla-args').value
    };
    const res = await postJson('/api/launch-carla', payload);
    if (!res.ok) {
      setStatus(`CARLA error: ${res.error || 'unknown'}`, false);
      return;
    }
    await refreshStatus();
  });
}

function init() {
  initStarfield();
  const style = document.createElement('style');
  style.textContent = `
    .est-icon .est-x {
      width: 14px;
      height: 14px;
      position: relative;
    }
    .est-icon .est-x::before,
    .est-icon .est-x::after {
      content: '';
      position: absolute;
      left: 6px;
      top: 0;
      width: 2px;
      height: 14px;
      background: #ff5d5d;
    }
    .est-icon .est-x::before { transform: rotate(45deg); }
    .est-icon .est-x::after { transform: rotate(-45deg); }
  `;
  document.head.appendChild(style);

  bindControls();
  refreshStatus();
  connectWs();
}

init();

function initStarfield() {
  const canvas = document.getElementById('starfield');
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  let width = 0;
  let height = 0;
  let animationFrame = null;
  const density = 0.00008;
  const maxLinkDistance = 140;
  const mouse = { x: -9999, y: -9999 };
  let points = [];

  function resize() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.floor(width * window.devicePixelRatio);
    canvas.height = Math.floor(height * window.devicePixelRatio);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const count = Math.max(50, Math.floor(width * height * density));
    points = Array.from({ length: count }).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      radius: 1 + Math.random() * 1.5
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(200, 230, 255, 0.75)';

    for (const point of points) {
      point.x += point.vx;
      point.y += point.vy;
      if (point.x < -50) point.x = width + 50;
      if (point.x > width + 50) point.x = -50;
      if (point.y < -50) point.y = height + 50;
      if (point.y > height + 50) point.y = -50;
      ctx.beginPath();
      ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.strokeStyle = 'rgba(120, 200, 255, 0.28)';
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      for (let j = i + 1; j < points.length; j++) {
        const q = points[j];
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const dist = Math.hypot(dx, dy);
        if (dist < maxLinkDistance) {
          const alpha = 1 - dist / maxLinkDistance;
          ctx.strokeStyle = `rgba(110, 190, 255, ${alpha * 0.35})`;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.stroke();
        }
      }
    }

    if (mouse.x > -1) {
      for (const p of points) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.hypot(dx, dy);
        if (dist < maxLinkDistance) {
          const alpha = 1 - dist / maxLinkDistance;
          ctx.strokeStyle = `rgba(90, 211, 122, ${alpha * 0.5})`;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
        }
      }
    }

    animationFrame = requestAnimationFrame(draw);
  }

  function onMouseMove(event) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = event.clientX - rect.left;
    mouse.y = event.clientY - rect.top;
  }

  function onMouseLeave() {
    mouse.x = -9999;
    mouse.y = -9999;
  }

  function cleanup() {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
    window.removeEventListener('resize', resize);
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseleave', onMouseLeave);
  }

  resize();
  draw();
  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseleave', onMouseLeave);
  window.addEventListener('beforeunload', cleanup);
}
