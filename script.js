/**
 * script.js — Real-Time Speech Diarization Client
 * ================================================
 *
 * This file implements the browser-side audio pipeline:
 *
 *  1. Request microphone access (Web Audio API)
 *  2. Connect to backend via WebSocket
 *  3. Capture audio in 16kHz float32 format
 *  4. Chunk audio every CHUNK_MS milliseconds
 *  5. Send raw PCM bytes to backend
 *  6. Receive JSON subtitle results
 *  7. Render subtitles with speaker color-coding
 *
 * ALL IMPLEMENTED FROM SCRATCH:
 *  - Audio resampling (downsample to 16kHz)
 *  - Float32 → bytes conversion
 *  - Energy-based VAD indicator (client side)
 *  - Audio level meter
 *  - Speaker registry UI
 *  - Subtitle rendering engine
 */

'use strict';

// ── Configuration ─────────────────────────────────────────
const DEFAULT_SAMPLE_RATE = 16000;
const DEFAULT_CHUNK_MS    = 1500;
const WS_RECONNECT_DELAY  = 2000;   // ms

// ── State ─────────────────────────────────────────────────
let audioContext    = null;
let mediaStream     = null;
let scriptProcessor = null;
let websocket       = null;
let isRecording     = false;
let sessionStart    = null;
let sessionTimer    = null;
let chunkCount      = 0;
let lineCount       = 0;

// Diagnostics state
let speechChunkCount    = 0;  // chunks where backend returned a result
let recentConfidences   = []; // rolling window for avg-conf
let diagPanelOpen       = true;

// Audio buffer (accumulate until chunk is ready)
let audioBuffer     = [];
let bufferLength    = 0;

// Speaker registry: speakerId → {label, color, duration, segmentCount}
const speakerRegistry = {};

// Subtitle history for export
const subtitleHistory = [];

// ── DOM Refs ──────────────────────────────────────────────
const startBtn       = () => document.getElementById('startBtn');
const stopBtn        = () => document.getElementById('stopBtn');
const transcriptFeed = () => document.getElementById('transcriptFeed');
const connectionBadge= () => document.getElementById('connectionBadge');
const vadDot         = () => document.getElementById('vadDot');
const levelBar       = () => document.getElementById('levelBar');
const levelVal       = () => document.getElementById('levelVal');
const partialLine    = () => document.getElementById('partialLine');
const partialSpeaker = () => document.getElementById('partialSpeaker');
const partialText    = () => document.getElementById('partialText');


// ══════════════════════════════════════════════════════════
// 1. WEBSOCKET CONNECTION
// ══════════════════════════════════════════════════════════

function getWsUrl() {
  const host = document.getElementById('wsHost').value.trim() || 'localhost:8000';
  return `ws://${host}/ws/audio`;
}

function connectWebSocket() {
  const url = getWsUrl();
  setConnectionStatus('connecting');
  showToast(`Connecting to ${url}…`);

  try {
    websocket = new WebSocket(url);
    websocket.binaryType = 'arraybuffer';

    websocket.onopen = () => {
      console.log('[WS] Connected');
      setConnectionStatus('connected');
      showToast('Connected to backend ✓');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Ignore keepalive pings from server
        if (data.type === 'ping') return;
        handleBackendMessage(data);
      } catch (e) {
        console.warn('[WS] Invalid JSON:', event.data, e);
      }
    };

    websocket.onerror = (err) => {
      console.error('[WS] Error', err);
      setConnectionStatus('disconnected');
    };

    websocket.onclose = (event) => {
      console.log('[WS] Closed', event.code, event.reason);
      setConnectionStatus('disconnected');
      if (isRecording) {
        showToast('WebSocket disconnected. Stopping recording.');
        stopRecording();
      }
    };

  } catch (err) {
    console.error('[WS] Failed to connect:', err);
    setConnectionStatus('disconnected');
    showToast(`Connection failed: ${err.message}`);
  }
}

function disconnectWebSocket() {
  if (websocket) {
    websocket.close();
    websocket = null;
  }
  setConnectionStatus('disconnected');
}


// ══════════════════════════════════════════════════════════
// 2. MICROPHONE CAPTURE (WEB AUDIO API)
//    Implemented from scratch — no external library
// ══════════════════════════════════════════════════════════

async function startRecording() {
  if (isRecording) return;

  // Connect WebSocket first
  connectWebSocket();

  // Wait a moment for WS to open
  await sleep(300);

  try {
    // Step 1: Request microphone permission
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,          // Mono
        sampleRate: { ideal: DEFAULT_SAMPLE_RATE },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    console.log('[Audio] Microphone access granted');

    // Step 2: Create AudioContext at target sample rate (16kHz)
    const targetSampleRate = parseInt(document.getElementById('sampleRate').value);
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: targetSampleRate
    });

    console.log(`[Audio] AudioContext sample rate: ${audioContext.sampleRate}Hz`);

    // Step 3: Create source from microphone stream
    const source = audioContext.createMediaStreamSource(mediaStream);

    // Step 4: Create ScriptProcessorNode for raw PCM access
    //         Buffer size 4096 = ~256ms at 16kHz
    //         This is the "from scratch" audio capture part
    const bufferSize = 4096;
    scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptProcessor.onaudioprocess = onAudioProcess;

    // Connect: source → processor → destination (for processing)
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    // Step 5: Initialize state
    isRecording  = true;
    sessionStart = Date.now();
    chunkCount   = 0;
    audioBuffer  = [];
    bufferLength = 0;

    // Start session timer
    sessionTimer = setInterval(updateSessionTimer, 1000);

    // Update UI
    startBtn().disabled = true;
    stopBtn().disabled  = false;
    setConnectionStatus('recording');
    clearWelcomeMessage();
    showToast('Recording started');

    console.log('[Recording] Started');

  } catch (err) {
    console.error('[Audio] Error:', err);
    let msg = 'Microphone error: ';
    if (err.name === 'NotAllowedError')   msg += 'Permission denied. Allow microphone access.';
    else if (err.name === 'NotFoundError') msg += 'No microphone found.';
    else msg += err.message;
    showToast(msg);
    disconnectWebSocket();
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  // Flush any remaining audio
  if (audioBuffer.length > 0) {
    sendAudioChunk(mergeBuffers(audioBuffer, bufferLength));
    audioBuffer  = [];
    bufferLength = 0;
  }

  // Cleanup audio
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }

  // Stop timer
  clearInterval(sessionTimer);

  // Disconnect WebSocket
  disconnectWebSocket();

  // Update UI
  startBtn().disabled = false;
  stopBtn().disabled  = true;
  setConnectionStatus('disconnected');
  setVAD(false);
  updateLevelMeter(0);
  partialLine().style.display = 'none';

  showToast('Recording stopped');
  console.log('[Recording] Stopped');
}


// ══════════════════════════════════════════════════════════
// 3. AUDIO PROCESSING — FROM SCRATCH
//    Called every ~256ms (bufferSize/sampleRate)
// ══════════════════════════════════════════════════════════

function onAudioProcess(event) {
  if (!isRecording) return;

  // Get raw float32 PCM samples from microphone
  const inputData = event.inputBuffer.getChannelData(0);

  // Copy to avoid buffer reuse issues
  const samples = new Float32Array(inputData);

  // ── Client-side VAD (energy-based, from scratch) ──────
  const rms = computeRMS(samples);
  const isSpeech = rms > 0.01;
  setVAD(isSpeech);
  updateLevelMeter(rms);

  // ── Accumulate into chunk buffer ──────────────────────
  audioBuffer.push(samples);
  bufferLength += samples.length;

  // ── Send chunk when we have enough audio ─────────────
  const chunkMs     = parseInt(document.getElementById('chunkMs').value) || DEFAULT_CHUNK_MS;
  const sampleRate  = audioContext ? audioContext.sampleRate : DEFAULT_SAMPLE_RATE;
  const chunkSamples = Math.floor(sampleRate * chunkMs / 1000);

  if (bufferLength >= chunkSamples) {
    // Merge all buffered arrays into one
    const merged = mergeBuffers(audioBuffer, bufferLength);

    // Take exactly chunkSamples
    const chunk = merged.slice(0, chunkSamples);

    // Keep remainder in buffer
    const remainder = merged.slice(chunkSamples);
    audioBuffer  = remainder.length > 0 ? [remainder] : [];
    bufferLength = remainder.length;

    // Send to backend
    sendAudioChunk(chunk);
  }
}

/**
 * Merge an array of Float32Arrays into a single Float32Array — FROM SCRATCH.
 *
 * @param {Float32Array[]} arrays - Array of buffers to merge
 * @param {number} totalLength   - Sum of all array lengths
 * @returns {Float32Array}
 */
function mergeBuffers(arrays, totalLength) {
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

/**
 * Compute Root Mean Square energy — FROM SCRATCH.
 * Used for client-side VAD indicator and level meter.
 *
 * Formula: RMS = sqrt( mean(x[i]^2) )
 *
 * @param {Float32Array} samples
 * @returns {number} RMS value (0 to 1)
 */
function computeRMS(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}


// ══════════════════════════════════════════════════════════
// 4. SEND AUDIO CHUNK TO BACKEND
// ══════════════════════════════════════════════════════════

function sendAudioChunk(float32Array) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    console.warn('[WS] Cannot send: WebSocket not open');
    return;
  }

  chunkCount++;
  document.getElementById('statChunks').textContent = chunkCount;

  // Convert Float32Array → raw bytes (ArrayBuffer)
  // The backend reads this with: np.frombuffer(raw_bytes, dtype=np.float32)
  const buffer = float32Array.buffer;

  websocket.send(buffer);
  console.debug(`[Audio] Sent chunk #${chunkCount}: ${float32Array.length} samples`);
}


// ══════════════════════════════════════════════════════════
// 5. HANDLE BACKEND MESSAGES
// ══════════════════════════════════════════════════════════

/**
 * Process a subtitle JSON from the backend.
 *
 * Expected format:
 * {
 *   "timestamp": "00:03",
 *   "speaker": "Speaker 1",
 *   "speaker_id": 0,
 *   "text": "Hello everyone",
 *   "confidence": 0.89,
 *   "color": "#4FC3F7",
 *   "is_partial": false
 * }
 */
function handleBackendMessage(data) {
  // Error handling
  if (data.error) {
    console.error('[Backend] Error:', data.error);
    showToast(`Backend error: ${data.error}`);
    return;
  }

  // Update partial line display (real-time feedback)
  if (data.is_partial) {
    updatePartialLine(data);
    return;
  }

  // Clear partial line
  partialLine().style.display = 'none';

  // Add to subtitle feed
  addSubtitleLine(data);

  // Update speaker registry
  updateSpeakerRegistry(data);

  // Update stats
  lineCount++;
  document.getElementById('statLines').textContent = lineCount;
  document.getElementById('statSpeakers').textContent = Object.keys(speakerRegistry).length;

  // Update diagnostics panel
  if (typeof data.confidence === 'number') {
    speechChunkCount++;
    recentConfidences.push(data.confidence);
    if (recentConfidences.length > 10) recentConfidences.shift();
    const avg = recentConfidences.reduce((a,b) => a+b, 0) / recentConfidences.length;
    document.getElementById('diagAvgConf').textContent = Math.round(avg * 100) + '%';
  }
  document.getElementById('diagTotalChunks').textContent = chunkCount;
  if (chunkCount > 0) {
    const ratio = Math.round((speechChunkCount / chunkCount) * 100);
    document.getElementById('diagSpeechRatio').textContent = ratio + '%';
  }

  // Store for export
  subtitleHistory.push(data);
}

// ══════════════════════════════════════════════════════════
// 6. SUBTITLE RENDERING — FROM SCRATCH
// ══════════════════════════════════════════════════════════

/**
 * Render a subtitle line in the transcript feed.
 * Uses CSS classes for speaker color coding.
 */
function addSubtitleLine(data) {
  // Remove welcome message if present
  clearWelcomeMessage();

  const feed = transcriptFeed();

  const line = document.createElement('div');
  line.className = `subtitle-line s-${data.speaker_id % 8}`;
  if (data.is_partial) line.classList.add('partial');

  // Format confidence percentage
  const confPct = Math.round((data.confidence || 0) * 100);

  line.innerHTML = `
    <div class="subtitle-meta">
      <span class="subtitle-timestamp">${data.timestamp}</span>
      <span class="subtitle-speaker">${escapeHtml(data.speaker)}</span>
    </div>
    <div class="subtitle-body">
      <div class="subtitle-text">${escapeHtml(data.text)}</div>
      <div class="subtitle-conf">conf: ${confPct}%</div>
    </div>
  `;

  // Set dynamic border color from backend + 5% tinted background
  if (data.color) {
    line.style.borderLeftColor = data.color;
    line.querySelector('.subtitle-speaker').style.color = data.color;
    // Parse hex color and apply as semi-transparent background tint
    const r = parseInt(data.color.slice(1,3),16);
    const g = parseInt(data.color.slice(3,5),16);
    const b = parseInt(data.color.slice(5,7),16);
    line.style.backgroundColor = `rgba(${r},${g},${b},0.05)`;
  }

  feed.appendChild(line);

  // Auto-scroll to bottom
  feed.scrollTop = feed.scrollHeight;
}

function updatePartialLine(data) {
  const line = partialLine();
  line.style.display = 'flex';

  const spElem = partialSpeaker();
  spElem.textContent = data.speaker + ': ';
  if (data.color) spElem.style.color = data.color;

  partialText().textContent = data.text;
}

function clearWelcomeMessage() {
  const welcome = transcriptFeed().querySelector('.welcome-message');
  if (welcome) welcome.remove();
}


// ══════════════════════════════════════════════════════════
// 7. SPEAKER REGISTRY — FROM SCRATCH
// ══════════════════════════════════════════════════════════

/**
 * Update the speaker registry UI with new speaker info.
 * Tracks each speaker's total speaking time.
 */
function updateSpeakerRegistry(data) {
  const sid = data.speaker_id;

  if (!speakerRegistry[sid]) {
    speakerRegistry[sid] = {
      label:        data.speaker,
      color:        data.color || '#4FC3F7',
      segments:     0,
      totalSeconds: 0
    };
  }

  const entry = speakerRegistry[sid];
  entry.segments++;
  entry.lastConf = data.confidence;
  // Estimate duration from text length (rough proxy)
  entry.totalSeconds += Math.max(1, data.text.split(' ').length * 0.4);

  renderSpeakerList();
}

function renderSpeakerList() {
  const list = document.getElementById('speakerList');
  list.innerHTML = '';

  // Sort by speaker ID
  const sorted = Object.entries(speakerRegistry)
    .sort(([a], [b]) => parseInt(a) - parseInt(b));

  if (sorted.length === 0) {
    list.innerHTML = '<p class="empty-hint">Detected speakers will appear here</p>';
    return;
  }

  for (const [sid, info] of sorted) {
    const badge = document.createElement('div');
    badge.className = 'speaker-badge';

    const durationFmt = formatSeconds(info.totalSeconds);
    const confPct = info.lastConf !== undefined
      ? `<span class="speaker-conf" style="color:#888;font-size:0.75em">conf: ${Math.round(info.lastConf * 100)}%</span>`
      : '';

    badge.innerHTML = `
      <span class="speaker-color" style="background:${info.color}"></span>
      <span class="speaker-name">${escapeHtml(info.label)}</span>
      <span class="speaker-time">${durationFmt}</span>
      ${confPct}
    `;
    list.appendChild(badge);
  }
}


// ══════════════════════════════════════════════════════════
// 8. UI HELPERS — FROM SCRATCH
// ══════════════════════════════════════════════════════════

function setConnectionStatus(status) {
  const badge = connectionBadge();
  const text  = badge.querySelector('.badge-text');
  const dot   = badge.querySelector('.badge-dot');

  badge.className = `badge badge--${status}`;
  const labels = {
    disconnected: 'Disconnected',
    connecting:   'Connecting…',
    connected:    'Connected',
    recording:    '● Recording'
  };
  text.textContent = labels[status] || status;
}

function setVAD(isSpeech) {
  const dot = vadDot();
  dot.className = `vad-dot vad-dot--${isSpeech ? 'speech' : 'silent'}`;
}

/**
 * Update the audio level meter bar — FROM SCRATCH.
 * Maps RMS (0-1) to percentage width, with color indication.
 *
 * @param {number} rms - 0 to 1
 */
function updateLevelMeter(rms) {
  const pct = Math.min(100, Math.round(rms * 400)); // Scale up for visual
  const bar = levelBar();
  bar.style.width = pct + '%';

  // Color: green < 70%, yellow 70-90%, red > 90%
  if (pct > 90) {
    bar.classList.add('hot');
  } else {
    bar.classList.remove('hot');
  }

  const db = rms > 1e-6 ? Math.round(20 * Math.log10(rms)) : -60;
  levelVal().textContent = db + 'dB';
}

function updateSessionTimer() {
  if (!sessionStart) return;
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  document.getElementById('statDuration').textContent = formatSeconds(elapsed);
}

/**
 * Format seconds as MM:SS — FROM SCRATCH.
 *
 * @param {number} secs
 * @returns {string}
 */
function formatSeconds(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function clearTranscript() {
  const feed = transcriptFeed();
  feed.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">🎙️</div>
      <p>Transcript cleared. Press <strong>Start Recording</strong> to begin again.</p>
    </div>
  `;
  partialLine().style.display = 'none';

  // Reset counters
  lineCount = 0;
  speechChunkCount = 0;
  recentConfidences.length = 0;
  Object.keys(speakerRegistry).forEach(k => delete speakerRegistry[k]);
  subtitleHistory.length = 0;

  document.getElementById('statLines').textContent = 0;
  document.getElementById('statSpeakers').textContent = 0;
  document.getElementById('diagAvgConf').textContent = '—';
  document.getElementById('diagSpeechRatio').textContent = '—';
  renderSpeakerList();
}

/**
 * Export transcript as a text file — FROM SCRATCH.
 */
function exportTranscript() {
  if (subtitleHistory.length === 0) {
    showToast('Nothing to export yet.');
    return;
  }

  // Build text content
  const lines = subtitleHistory.map(s =>
    `[${s.timestamp}] ${s.speaker}: ${s.text}`
  );
  const content = lines.join('\n');

  // Create and trigger download
  const blob = new Blob([content], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href  = url;
  link.download = `transcript_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
  link.click();
  URL.revokeObjectURL(url);

  showToast(`Exported ${subtitleHistory.length} lines.`);
}

function showToast(message, durationMs = 3000) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.style.display = 'block';
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    toast.style.display = 'none';
  }, durationMs);
}

/**
 * Escape HTML special characters to prevent XSS — FROM SCRATCH.
 */
function escapeHtml(str) {
  const map = { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' };
  return String(str).replace(/[&<>"']/g, c => map[c]);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toggleDiagnostics() {
  diagPanelOpen = !diagPanelOpen;
  const content  = document.getElementById('diagContent');
  const chevron  = document.getElementById('diagChevron');
  content.style.display  = diagPanelOpen ? '' : 'none';
  chevron.textContent    = diagPanelOpen ? '▼' : '▶';
}


// ══════════════════════════════════════════════════════════
// 9. INITIALIZATION
// ══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  console.log('[App] LiveTranscribe initialized');

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.ctrlKey) {
      e.preventDefault();
      if (!isRecording) startRecording();
      else stopRecording();
    }
    if (e.code === 'KeyL' && e.ctrlKey) {
      e.preventDefault();
      clearTranscript();
    }
  });

  // Check for browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Your browser does not support microphone access. Use Chrome or Firefox.', 8000);
    startBtn().disabled = true;
  }

  if (!window.WebSocket) {
    showToast('Your browser does not support WebSocket. Use Chrome or Firefox.', 8000);
    startBtn().disabled = true;
  }
});

// Handle page unload — clean up streams
window.addEventListener('beforeunload', () => {
  if (isRecording) stopRecording();
});
