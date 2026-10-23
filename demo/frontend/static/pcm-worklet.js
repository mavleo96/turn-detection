// AudioWorklet: mono mic audio -> 16 kHz Int16 frames of 512 samples, posted to the main thread.
// Normally the AudioContext already runs at 16 kHz. If the browser refused that (e.g. Firefox),
// the context runs at the device rate and we downsample here by averaging each output sample's
// input span (a crude anti-aliasing filter).
class PcmFrames extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.frameSize = o.frameSize || 512;
    this.ratio = sampleRate / (o.targetRate || 16000);
    this.buf = new Int16Array(this.frameSize);
    this.n = 0;
    this.acc = 0;
    this.cnt = 0;
    this.pos = 0;
  }

  push(s) {
    s = Math.max(-1, Math.min(1, s));
    this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.n === this.frameSize) {
      this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
      this.buf = new Int16Array(this.frameSize);
      this.n = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.push(ch[i]);
      return true;
    }
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.cnt++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.push(this.acc / this.cnt);
        this.acc = 0;
        this.cnt = 0;
        this.pos -= this.ratio;
      }
    }
    return true;
  }
}

registerProcessor("pcm-frames", PcmFrames);
