// ============================================================
//  Chart Eye — Side Panel Logic
//  Iterative screenshot → region detection → crop → analysis
// ============================================================

const API_BASE  = 'http://127.0.0.1:8000';
const MAX_ITERS = 3;

// ── DOM refs ─────────────────────────────────────────────────
const chatArea    = document.getElementById('chat-area');
const visionStrip = document.getElementById('vision-strip');
const userInput   = document.getElementById('user-input');
const analyzeBtn  = document.getElementById('analyze-btn');
const statusText  = document.getElementById('status-text');

// ── Guard against double-send ────────────────────────────────
let isAnalyzing = false;

// ============================================================
//  UI STATE MACHINE
// ============================================================
function setAppState(state) {
  // state: 'idle' | 'scanning' | 'analyzing'
  document.body.className = state;
  const labels = { idle: 'READY', scanning: 'SCANNING', analyzing: 'ANALYZING' };
  statusText.textContent = labels[state] ?? 'READY';
}

function setInputEnabled(enabled) {
  userInput.disabled  = !enabled;
  analyzeBtn.disabled = !enabled;
  isAnalyzing         = !enabled;
}

// ============================================================
//  VISION STRIP
// ============================================================
function addThumb(dataUrl, label) {
  // Deactivate all previous thumbs
  visionStrip.querySelectorAll('.vision-thumb-wrap').forEach(w => w.classList.remove('active'));

  // Arrow connector between entries
  if (visionStrip.children.length > 0) {
    const arrow = document.createElement('div');
    arrow.className = 'vision-arrow';
    arrow.textContent = '›';
    visionStrip.appendChild(arrow);
  }

  const wrap = document.createElement('div');
  wrap.className = 'vision-thumb-wrap active';

  const img = document.createElement('img');
  img.className = 'vision-thumb';
  img.src       = dataUrl;
  img.title     = label;

  const lbl = document.createElement('div');
  lbl.className   = 'vision-label';
  lbl.textContent = label.toUpperCase();

  wrap.appendChild(img);
  wrap.appendChild(lbl);
  visionStrip.appendChild(wrap);
  wrap.scrollIntoView({ behavior: 'smooth', inline: 'end' });
}

function clearVisionStrip() {
  visionStrip.innerHTML = '';
}

// ============================================================
//  CHAT MESSAGES
// ============================================================
function addMessage(role, text) {
  // role: 'user' | 'ai' | 'system'
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${role}`;

  if (role !== 'system') {
    const roleLabel = document.createElement('div');
    roleLabel.className   = 'msg-role';
    roleLabel.textContent = role === 'user' ? 'YOU' : 'CHART EYE';
    wrapper.appendChild(roleLabel);
  }

  const bubble = document.createElement('div');
  bubble.className   = 'msg-bubble';
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  chatArea.appendChild(wrapper);
  chatArea.scrollTop = chatArea.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg ai';
  wrapper.id        = 'typing-indicator';

  const roleLabel = document.createElement('div');
  roleLabel.className   = 'msg-role';
  roleLabel.textContent = 'CHART EYE';
  wrapper.appendChild(roleLabel);

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble typing-dots';
  bubble.innerHTML = '<span></span><span></span><span></span>';
  wrapper.appendChild(bubble);

  chatArea.appendChild(wrapper);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTypingIndicator() {
  document.getElementById('typing-indicator')?.remove();
}

// ============================================================
//  CANVAS CROP HELPER
// ============================================================
function cropImage(dataUrl, region) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width  = region.w;
      canvas.height = region.h;
      canvas.getContext('2d').drawImage(
        img,
        region.x, region.y, region.w, region.h,  // source
        0, 0, region.w, region.h                  // destination
      );
      resolve(canvas.toDataURL('image/png'));
    };
    img.src = dataUrl;
  });
}

// ============================================================
//  BACKEND CALLS
// ============================================================
async function callAnalyze(base64Image, query, iteration, context) {
  let resp;
  try {
    resp = await fetch(`${API_BASE}/analyze`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: base64Image, query, iteration, context })
    });
  } catch (networkErr) {
    throw new Error('NETWORK — backend unreachable. Is app.py running?');
  }

  if (!resp.ok) {
    let detail = '';
    try { detail = (await resp.json()).detail ?? ''; } catch (_) {}
    throw new Error(`SERVER ${resp.status}: ${detail || resp.statusText}`);
  }

  return resp.json();
}

// ============================================================
//  MAIN ANALYSIS FLOW
// ============================================================
async function runAnalysis(query) {
  // ── 1. Capture screenshot ──────────────────────────────────
  setAppState('scanning');
  addMessage('system', '▸ CAPTURING SCREEN');

  let dataUrl;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png' });
  } catch (e) {
    addMessage('system', '✕ CAPTURE FAILED — CHECK EXTENSION PERMISSIONS');
    return;
  }

  if (!dataUrl) {
    addMessage('system', '✕ CAPTURE RETURNED EMPTY DATA');
    return;
  }

  clearVisionStrip();
  addThumb(dataUrl, 'Full screen');
  setAppState('analyzing');

  // ── 2. Iterative crop loop ────────────────────────────────
  let currentImage = dataUrl;
  let context      = '';

  for (let iteration = 0; iteration < MAX_ITERS; iteration++) {
    const base64 = currentImage.split(',')[1];

    if (iteration === 0) {
      addMessage('system', '▸ DETECTING CHART REGION');
    } else {
      addMessage('system', `▸ ZOOMING INTO ${context}`);
    }

    addTypingIndicator();

    let data;
    try {
      data = await callAnalyze(base64, query, iteration, context);
    } catch (e) {
      removeTypingIndicator();
      addMessage('system', `✕ ${e.message}`);
      return;
    }

    removeTypingIndicator();

    if (data.action === 'crop') {
      // Crop and continue loop
      context      = data.context || 'REGION';
      currentImage = await cropImage(currentImage, data.region);
      addThumb(currentImage, context);
      // next iteration will analyze the cropped image

    } else if (data.action === 'answer') {
      if (data.coaching) {
        addCoachingMessage(data.coaching);
      } else {
        addMessage('ai', data.prediction);
      }
      return;
    }
  }

  // ── 3. Max iterations hit → force final answer ────────────
  addMessage('system', '▸ FINALIZING ANALYSIS');
  addTypingIndicator();

  try {
    const base64 = currentImage.split(',')[1];
    const data   = await callAnalyze(base64, query, MAX_ITERS, context);
    removeTypingIndicator();
    if (data.coaching) {
      addCoachingMessage(data.coaching);
    } else {
      addMessage('ai', data.prediction ?? 'No response from model.');
    }
  } catch (e) {
    removeTypingIndicator();
    addMessage('system', `✕ ${e.message}`);
  }
}

// ============================================================
//  COACHING CARD RENDERER
// ============================================================
function addCoachingMessage(c) {
  // c = TradingCoachOutput from BAML

  const wrapper = document.createElement('div');
  wrapper.className = 'msg ai';

  const roleLabel = document.createElement('div');
  roleLabel.className   = 'msg-role';
  roleLabel.textContent = 'CHART EYE';
  wrapper.appendChild(roleLabel);

  const card = document.createElement('div');
  card.className = 'coaching-card';

  // ── Zoom warning (if not optimal) ────────────────────────
  const zoom = c.zoom_assessment;
  if (zoom && zoom.state !== 'Optimal') {
    const warn = document.createElement('div');
    warn.className = `zoom-warning ${zoom.state === 'TooZoomedIn' ? 'too-in' : 'too-out'}`;
    warn.innerHTML = `
      <span class="zoom-warning-icon">${zoom.state === 'TooZoomedIn' ? '🔍' : '🗺'}</span>
      <span class="zoom-warning-body">
        <span class="zoom-warning-msg">${zoom.state === 'TooZoomedIn' ? 'TOO ZOOMED IN' : 'TOO ZOOMED OUT'} (~${zoom.estimated_candles} candles)</span>
        <span class="zoom-warning-cta">${zoom.call_to_action}</span>
      </span>`;
    card.appendChild(warn);
  }

  // ── Body ─────────────────────────────────────────────────
  const body = document.createElement('div');
  body.className = 'coaching-body';

  // Badge row
  const biasClass = { Bullish: 'badge-bullish', Bearish: 'badge-bearish', Neutral: 'badge-neutral' };
  const confClass = { High: 'badge-high', Medium: 'badge-medium', Low: 'badge-low' };
  const biasArrow = { Bullish: ' ↑', Bearish: ' ↓', Neutral: ' →' };

  const badgeRow = document.createElement('div');
  badgeRow.className = 'badge-row';
  badgeRow.innerHTML = `
    <span class="badge badge-phase">${c.market_phase ?? ''}</span>
    <span class="badge ${biasClass[c.bias] ?? ''}">${c.bias ?? ''}${biasArrow[c.bias] ?? ''}</span>
    <span class="badge ${confClass[c.confidence] ?? ''}">${c.confidence ?? ''} CONF</span>`;
  body.appendChild(badgeRow);

  // Coach commentary
  const commentary = document.createElement('div');
  commentary.className   = 'coach-commentary';
  commentary.textContent = c.coach_commentary ?? '';
  body.appendChild(commentary);

  // Phase explanation (secondary)
  if (c.phase_explanation) {
    const phaseExp = document.createElement('div');
    phaseExp.className   = 'phase-explanation';
    phaseExp.textContent = c.phase_explanation;
    body.appendChild(phaseExp);
  }

  // Candle patterns
  const patterns = (c.candle_patterns ?? []);
  if (patterns.length > 0) {
    const title = document.createElement('div');
    title.className   = 'coaching-section-title';
    title.textContent = 'CANDLE PATTERNS';
    body.appendChild(title);

    const list = document.createElement('div');
    list.className = 'pattern-list';
    patterns.forEach(p => {
      const entry = document.createElement('div');
      entry.className = `pattern-entry ${p.is_valid ? 'valid' : 'invalid'}`;
      entry.innerHTML = `
        <span class="pattern-icon">${p.is_valid ? '✓' : '✗'}</span>
        <span class="pattern-text">
          <strong>${p.type ?? ''}</strong> — ${p.location ?? ''}
          <span class="pattern-ctx">${p.context ?? ''}</span>
        </span>`;
      list.appendChild(entry);
    });
    body.appendChild(list);
  }

  // Fair Value Gaps
  const fvgs = (c.fair_value_gaps ?? []);
  if (fvgs.length > 0) {
    const title = document.createElement('div');
    title.className   = 'coaching-section-title';
    title.textContent = 'FAIR VALUE GAPS';
    body.appendChild(title);

    const list = document.createElement('div');
    list.className = 'pattern-list';
    fvgs.forEach(fvg => {
      const entry = document.createElement('div');
      entry.className = `fvg-entry ${(fvg.direction ?? '').toLowerCase()}`;
      entry.innerHTML = `
        <span class="pattern-icon">◈</span>
        <span><strong>FVG ${fvg.direction ?? ''}</strong> — ${fvg.description ?? ''}</span>`;
      list.appendChild(entry);
    });
    body.appendChild(list);
  }

  // Chart shapes
  const shapes = (c.chart_shapes ?? []);
  if (shapes.length > 0) {
    const title = document.createElement('div');
    title.className   = 'coaching-section-title';
    title.textContent = 'CHART GEOMETRY';
    body.appendChild(title);

    const list = document.createElement('div');
    list.className = 'pattern-list';
    shapes.forEach(s => {
      const entry = document.createElement('div');
      entry.className = 'pattern-entry valid';
      entry.innerHTML = `
        <span class="pattern-icon">△</span>
        <span class="pattern-text">
          <strong>${s.pattern ?? ''}</strong> [${s.status ?? ''}]
          <span class="pattern-ctx">${s.implication ?? ''}</span>
        </span>`;
      list.appendChild(entry);
    });
    body.appendChild(list);
  }

  // Call to action block
  if (c.call_to_action) {
    const cta = document.createElement('div');
    cta.className = 'cta-block';
    cta.innerHTML = `
      <div class="cta-title">⚡ CALL TO ACTION</div>
      <div class="cta-text">${c.call_to_action}</div>`;
    body.appendChild(cta);
  }

  card.appendChild(body);
  wrapper.appendChild(card);
  chatArea.appendChild(wrapper);
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ============================================================
//  ENTRY POINT
// ============================================================
async function handleAnalyze() {
  const query = userInput.value.trim();
  if (!query || isAnalyzing) return;

  addMessage('user', query);
  userInput.value = '';
  setInputEnabled(false);

  try {
    await runAnalysis(query);
  } finally {
    setAppState('idle');
    setInputEnabled(true);
    userInput.focus();
  }
}

// ── Event listeners ──────────────────────────────────────────
analyzeBtn.addEventListener('click', handleAnalyze);

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') handleAnalyze();
});
