// Browser client: mic -> WebSocket -> server, and server audio -> speakers.
//
// Messages FROM the server:
//   text JSON  {kind: "event" | "state" | "llm_delta" | "assistant_text", ...}
//   binary     raw PCM16 audio of the assistant's voice (sample rate given by the tts_start event)
// Messages TO the server:
//   binary     raw PCM16 mic audio, 16 kHz mono, 512-sample frames
//   text JSON  {type: "start" | "playback_start" | "playback_end", turn_id}

const $ = (id) => document.getElementById(id);
const chat = $("chat"), eventsEl = $("events");

let ws, ctx, serverState = "idle";

// ---------------------------------------------------------------- UI helpers
const STATUS_TEXT = {
  idle: "Not started", listening: "Listening…", "speaking-user": "You're speaking",
  responding: "Thinking…", playing: "Coach speaking", closed: "Disconnected",
};
function setStatus(s) {
  $("status").dataset.s = s;
  $("statusText").textContent = STATUS_TEXT[s] || s;
}

function bubble(cls, text = "") {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function logEvent(ev) {
  const row = document.createElement("div");
  row.className = "ev " + ev.type;
  const d = { ...ev.data };
  for (const k of Object.keys(d)) {
    if (typeof d[k] === "string" && d[k].length > 70) d[k] = d[k].slice(0, 70) + "…";
  }
  const details = Object.keys(d).length ? JSON.stringify(d) : "";
  row.innerHTML = `<span class="t">${ev.t_ms.toFixed(0)}</span><span class="turn">#${ev.turn_id}</span>` +
    `<span><span class="name">${ev.type}</span> <span class="d"></span></span>`;
  row.querySelector(".d").textContent = details;
  eventsEl.appendChild(row);
  eventsEl.scrollTop = eventsEl.scrollHeight;
}

// ---------------------------------------------------------------- latency waterfall
// One bar per turn: where the time went between "you stopped talking" and "coach starts talking".
const STAGES = [
  ["endpointing_ms", "VAD silence wait", "#9aa3b2"],
  ["asr_ms", "ASR", "#7c5cff"],
  ["llm_ttft_ms", "LLM first token", "#2f6fed"],
  ["llm_wait_ms", "LLM text wait", "#3fb4e8"],
  ["tts_ttfb_ms", "TTS first byte", "#1a9b5c"],
  ["overhead_ms", "Overhead", "#d98a00"],
];

function waterfall(b, targetMs) {
  const box = document.createElement("div");
  box.className = "wf";
  const scale = Math.max(b.perceived_ms, targetMs) * 1.05;
  const bar = document.createElement("div");
  bar.className = "wf-bar";
  for (const [key, label, color] of STAGES) {
    const seg = document.createElement("div");
    seg.style.width = (Math.max(0, b[key]) / scale) * 100 + "%";
    seg.style.background = color;
    seg.title = `${label}: ${b[key].toFixed(0)} ms`;
    bar.appendChild(seg);
  }
  const target = document.createElement("div");
  target.className = "wf-target";
  target.style.left = (targetMs / scale) * 100 + "%";
  target.title = `target ${targetMs} ms`;
  bar.appendChild(target);

  const ok = b.perceived_ms <= targetMs;
  const caption = document.createElement("div");
  caption.className = "wf-caption";
  if (b.degraded) {  // filler / fallback / text-only turns: only the end-to-end time is meaningful
    bar.remove();
    caption.innerHTML = `⚠️ degraded turn · <b class="${ok ? "ok" : "bad"}">${b.perceived_ms.toFixed(0)} ms</b> to first audio`;
    box.append(caption);
    return box;
  }
  caption.innerHTML = `<b class="${ok ? "ok" : "bad"}">${b.perceived_ms.toFixed(0)} ms</b> to first audio ` +
    `(target ${targetMs}) · ` + STAGES.map(([k, l]) => `${l} ${b[k].toFixed(0)}`).join(" · ");
  box.append(bar, caption);
  return box;
}

// ---------------------------------------------------------------- playback
const play = { turnId: null, sampleRate: 22050, nextTime: 0, pending: 0, started: false, ttsDone: false,
               sources: [], accepting: true };
const userBubbles = {}, botBubbles = {};

function resetPlayback(turnId, sampleRate) {
  Object.assign(play, { turnId, sampleRate, nextTime: 0, pending: 0, started: false, ttsDone: false,
                       sources: [], accepting: true });
}

// Barge-in: the server cut the coach off -> silence everything that is queued.
function stopPlayback() {
  for (const src of play.sources) { src.onended = null; try { src.stop(); } catch {} }
  Object.assign(play, { turnId: null, sources: [], pending: 0, accepting: false });
}

function playChunk(arrayBuffer) {
  if (!play.accepting) return;  // late audio from an interrupted turn
  const pcm = new Int16Array(arrayBuffer);
  const buf = ctx.createBuffer(1, pcm.length, play.sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const when = Math.max(play.nextTime, ctx.currentTime + 0.03);  // queue chunks back-to-back
  src.start(when);
  play.nextTime = when + buf.duration;
  play.pending++;
  play.sources.push(src);

  if (!play.started) {
    play.started = true;
    const turnId = play.turnId;
    // tell the server the moment sound actually comes out of the speakers
    setTimeout(() => {
      send({ type: "playback_start", turn_id: turnId });
      setStatus("playing");
    }, Math.max(0, (when - ctx.currentTime) * 1000));
  }
  src.onended = () => {
    play.pending--;
    play.sources = play.sources.filter((s) => s !== src);
    maybeFinishPlayback();
  };
}

function maybeFinishPlayback() {
  if (play.ttsDone && play.pending === 0 && play.turnId !== null) {
    send({ type: "playback_end", turn_id: play.turnId });
    play.turnId = null;
  }
}

// ---------------------------------------------------------------- server messages
function onServerMessage(msg) {
  if (msg.kind === "state") {
    serverState = msg.state;
    if (msg.state === "listening") setStatus("listening");
    if (msg.state === "responding" && $("status").dataset.s !== "playing") setStatus("responding");
    return;
  }
  if (msg.kind === "assistant_text") {           // greeting / canned lines (no LLM involved)
    const b = botBubbles[msg.turn_id];
    if (b) b.textContent += (b.textContent ? " " : "") + msg.text;
    else botBubbles[msg.turn_id] = bubble("bot", msg.text);
    return;
  }
  if (msg.kind === "assistant_note") {           // e.g. "Give me a second." while we retry
    note(`💭 ${msg.text}`);
    return;
  }
  if (msg.kind === "stop_audio") { stopPlayback(); return; }
  if (msg.kind === "tts_unavailable") { note("🔇 Voice is unavailable right now — replies are shown as text."); return; }
  if (msg.kind === "asr_unavailable") {
    note("🎙️ Speech recognition is unavailable — type your answer below.");
    $("textInput").classList.add("attention");
    $("textInput").focus();
    return;
  }
  if (msg.kind === "llm_delta") {               // reply streaming in, token by token
    if (!botBubbles[msg.turn_id]) botBubbles[msg.turn_id] = bubble("bot");
    botBubbles[msg.turn_id].textContent += msg.text;
    chat.scrollTop = chat.scrollHeight;
    return;
  }
  if (msg.kind !== "event") return;

  logEvent(msg);
  const t = msg.turn_id, d = msg.data;
  switch (msg.type) {
    case "speech_start":
      setStatus("speaking-user");
      userBubbles[t] = bubble("user partial", "…");
      break;
    case "speech_end":
      if (d.discarded) { userBubbles[t]?.remove(); setStatus("listening"); }
      break;
    case "asr_partial":
      if (userBubbles[t]) userBubbles[t].textContent = d.text;
      break;
    case "asr_final":
      if (userBubbles[t]) { userBubbles[t].className = "msg user"; userBubbles[t].textContent = d.text || "(nothing recognized)"; }
      break;
    case "turn_skipped":
      userBubbles[t]?.remove();
      break;
    case "tts_start":
      resetPlayback(t, d.sample_rate);
      break;
    case "tts_done":
      play.ttsDone = true;
      maybeFinishPlayback();
      break;
    case "turn_complete":
      if (d.breakdown && botBubbles[t]) botBubbles[t].appendChild(waterfall(d.breakdown, d.target_ms));
      break;
    case "text_input":
      userBubbles[t] = bubble("user", "⌨️ " + d.text);
      $("textInput").classList.remove("attention");
      break;
    case "stage_failed":
      note(`⚠️ ${d.stage.toUpperCase()} ${d.reason.replace("_", " ")}` +
           (d.elapsed_ms ? ` after ${d.elapsed_ms.toFixed(0)} ms` : "") + (d.model ? ` (${d.model})` : ""), "warn");
      break;
    case "fallback":
      note(`↪ fallback: ${d.action.replaceAll("_", " ")}`, "warn");
      break;
    case "barge_in":
      note("✋ You interrupted — coach stopped talking.");
      break;
    case "error":
      bubble("bot", `⚠️ ${d.stage} error: ${d.error}`);
      play.turnId = null;
      break;
  }
}

function note(text, cls = "") {
  const div = document.createElement("div");
  div.className = "note " + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// ---------------------------------------------------------------- typing instead of talking
$("textForm").onsubmit = (e) => {
  e.preventDefault();
  const text = $("textInput").value.trim();
  if (!text) return;
  stopPlayback();                       // typing while the coach talks interrupts it
  send({ type: "text_input", text });
  $("textInput").value = "";
};

// ---------------------------------------------------------------- fault injection panel
async function refreshFaults() {
  const f = await (await fetch("/faults")).json();
  for (const [comp, spec] of Object.entries(f)) {
    const sel = $("fault-" + comp);
    if (![...sel.options].some((o) => o.value === spec)) sel.add(new Option(spec, spec));
    sel.value = spec;
  }
}
$("faultsBtn").onclick = () => { $("faults").hidden = !$("faults").hidden; refreshFaults(); };
for (const comp of ["asr", "llm", "tts"]) {
  $("fault-" + comp).onchange = async (e) => {
    await fetch("/faults", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ [comp]: e.target.value }) });
    refreshFaults();
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---------------------------------------------------------------- start
async function start() {
  $("startBtn").disabled = true;
  chat.innerHTML = "";

  // One AudioContext for both mic capture and playback (must be created on a click).
  if (ctx) ctx.close();
  ctx = new AudioContext();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  await ctx.audioWorklet.addModule("/static/mic-worklet.js");
  const mic = ctx.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(ctx, "mic-processor");
  const mute = ctx.createGain();
  mute.gain.value = 0;                 // keep the graph running without echoing the mic
  mic.connect(worklet).connect(mute).connect(ctx.destination);

  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => send({ type: "start" });
  ws.onclose = () => { setStatus("closed"); $("startBtn").disabled = false; $("startBtn").textContent = "Restart"; };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") onServerMessage(JSON.parse(e.data));
    else playChunk(e.data);
  };

  worklet.port.onmessage = (e) => {
    // mic level meter
    const pcm = new Int16Array(e.data);
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) sum += (pcm[i] / 32768) ** 2;
    $("meter").style.width = Math.min(100, Math.sqrt(sum / pcm.length) * 400) + "%";
    if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
}

$("startBtn").onclick = () => start().catch((err) => {
  bubble("bot", "⚠️ " + err.message);
  $("startBtn").disabled = false;
});
