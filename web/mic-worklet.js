// AudioWorklet: runs on the browser's audio thread.
// Takes mic audio at the device rate (usually 48 kHz), resamples it to 16 kHz,
// converts to 16-bit integers, and posts 512-sample (32 ms) frames to the page.

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.step = sampleRate / this.targetRate; // input samples per output sample
    this.pos = 0;                             // fractional read position
    this.frame = new Int16Array(512);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;

    // linear interpolation resampler
    while (this.pos < input.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = input[i];
      const b = input[Math.min(i + 1, input.length - 1)];
      let s = a + (b - a) * frac;
      s = Math.max(-1, Math.min(1, s));
      this.frame[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.filled === this.frame.length) {
        this.port.postMessage(this.frame.buffer.slice(0));
        this.filled = 0;
      }
      this.pos += this.step;
    }
    this.pos -= input.length;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
