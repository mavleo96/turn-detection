// Mic capture + audio WebSocket for the live panel (see server.py for the wire format).
(() => {
  const SR = 16000;
  const FRAME = 512;
  const HEADER = 12; // uint32 seq + float64 client timestamp (ms)

  let ws = null, ctx = null, stream = null, node = null;
  let seq = 0, pingId = 0, pingTimer = null;
  let finalStatus = false; // done / error / stopped: not overwritten when the socket closes

  const $ = (id) => document.getElementById(id);
  const setStatus = (s) => { const el = $("td-status"); if (el) el.textContent = s; };
  const endWith = (s) => { setStatus(s); finalStatus = true; };
  const setButtons = (running) => {
    if ($("td-start")) $("td-start").disabled = running;
    if ($("td-stop")) $("td-stop").disabled = !running;
  };

  // Gradio renders the panel after this script loads, so delegate clicks from the document.
  document.addEventListener("click", (e) => {
    if (e.target.closest("#td-start")) start();
    else if (e.target.closest("#td-stop")) stop();
  });

  async function start() {
    if (ws) return;
    finalStatus = false;
    setButtons(true);
    setStatus("requesting mic…");
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      let src;
      try {
        ctx = new AudioContext({ sampleRate: SR });
        src = ctx.createMediaStreamSource(stream);
      } catch (err) {
        // Some browsers can't connect a mic to a context at another rate: resample in the worklet.
        if (ctx) ctx.close();
        ctx = new AudioContext();
        src = ctx.createMediaStreamSource(stream);
      }
      await ctx.audioWorklet.addModule("/td-static/pcm-worklet.js");
      node = new AudioWorkletNode(ctx, "pcm-frames", { processorOptions: { targetRate: SR, frameSize: FRAME } });
      src.connect(node);
      node.connect(ctx.destination); // keeps the worklet running; it outputs silence
    } catch (err) {
      stopAudio(); // releases the mic if it was granted
      setStatus("audio error: " + err.message);
      setButtons(false);
      return;
    }
    const resampled = ctx.sampleRate !== SR;

    setStatus("connecting…");
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(proto + location.host + "/ws/audio");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      seq = 0;
      node.port.onmessage = (ev) => sendFrame(ev.data);
      pingTimer = setInterval(ping, 2000);
      setStatus(resampled ? `streaming (browser at ${ctx.sampleRate} Hz, resampling)` : "streaming");
    };
    ws.onmessage = (ev) => handle(JSON.parse(ev.data));
    ws.onclose = () => {
      cleanup();
      setButtons(false);
      if (!finalStatus) setStatus("disconnected");
    };
    ws.onerror = () => endWith("WebSocket error");
  }

  function sendFrame(pcmBuffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const buf = new ArrayBuffer(HEADER + FRAME * 2);
    const dv = new DataView(buf);
    dv.setUint32(0, seq++, true);
    dv.setFloat64(4, performance.timeOrigin + performance.now(), true);
    new Int16Array(buf, HEADER).set(new Int16Array(pcmBuffer));
    ws.send(buf);
  }

  function ping() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping", id: ++pingId, client_ts: performance.now() }));
    }
  }

  function stopAudio() {
    if (node) { node.port.onmessage = null; node.disconnect(); node = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    if (ctx) { ctx.close(); ctx = null; }
    if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
  }

  function stop() {
    stopAudio();
    if (!ws) return;
    if (ws.readyState === WebSocket.CONNECTING) {
      // No session to finish yet; closing a connecting socket also fires onerror, so drop it.
      ws.onerror = null;
      ws.close();
      endWith("stopped");
    } else if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      setStatus("finishing…");
    }
  }

  function cleanup() {
    stopAudio();
    ws = null;
  }

  function segmentEl(id) {
    let el = document.getElementById("td-seg-" + id);
    if (!el) {
      el = document.createElement("div");
      el.id = "td-seg-" + id;
      el.className = "td-seg";
      el.innerHTML = '<span class="td-text td-partial"></span><span class="td-metrics td-dim"></span>';
      const box = $("td-transcript");
      box.appendChild(el);
    }
    return el;
  }

  const fmt = (v) => (v === null || v === undefined ? "–" : Math.round(v) + " ms");

  function handle(ev) {
    switch (ev.type) {
      case "session_start":
        $("td-transcript").innerHTML = "";
        break;
      case "pong":
        ws.send(JSON.stringify({ type: "rtt", id: ev.id, rtt_ms: performance.now() - ev.client_ts }));
        break;
      case "network":
        $("td-net").textContent = `network one-way ≈ ${ev.one_way_ms.toFixed(1)} ms`;
        break;
      case "vad_start":
        $("td-vad").textContent = "speaking";
        $("td-vad").classList.add("td-on");
        $("td-vad-lat").textContent = `start latency ${fmt(ev.latency_ms)}`;
        break;
      case "vad_end":
        $("td-vad").textContent = "silent";
        $("td-vad").classList.remove("td-on");
        $("td-vad-lat").textContent = `end latency ${fmt(ev.latency_ms)}`;
        break;
      case "asr_partial": {
        const t = segmentEl(ev.segment_id).querySelector(".td-text");
        t.textContent = ev.text;
        break;
      }
      case "asr_final": {
        if (!ev.text && ev.reason === "close") break;
        const el = segmentEl(ev.segment_id);
        const t = el.querySelector(".td-text");
        t.textContent = ev.text || "(no speech recognised)";
        t.classList.remove("td-partial");
        el.querySelector(".td-metrics").textContent = ev.reason === "vad_end"
          ? ` speech end→final ${fmt(ev.speech_end_to_final_ms)} · VAD fire→final ${fmt(ev.vad_fire_to_final_ms)}`
          : " (flushed at stop)";
        $("td-transcript").scrollTop = $("td-transcript").scrollHeight;
        break;
      }
      case "session_end":
        if (!finalStatus) endWith("done");
        if (ws) ws.close();
        break;
      case "error":
        endWith("error: " + ev.message);
        break;
    }
  }
})();
