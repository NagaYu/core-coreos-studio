/*!
 * ============================================================================
 *  Core-CoreOS Studio · apple_optimized_core.js
 * ----------------------------------------------------------------------------
 *  On-device multimodal inference engine for Apple Silicon (M-series) browsers.
 *
 *  Design goals
 *  ------------
 *  1. ZERO data egress. Every byte of pixel / audio data is processed inside
 *     the browser sandbox. Nothing ever touches the network except the initial,
 *     cacheable model weights download from a public CDN.
 *
 *  2. Unified-Memory friendliness. On Apple Silicon the CPU, GPU (WebGPU) and
 *     Neural Engine share one physical memory pool. The single most important
 *     thing a web app can do is avoid *redundant* copies and keep large
 *     buffers short-lived so the allocator can recycle pages immediately.
 *     This module therefore:
 *       - decodes & down-samples media on the main thread into the *smallest*
 *         buffer the model needs, then hands ownership to the worker via
 *         Transferable objects (zero-copy move, not a structured clone),
 *       - aggressively closes ImageBitmaps, revokes Object URLs and closes
 *         AudioContexts the instant they are consumed,
 *       - reuses one OffscreenCanvas / 2D context instead of churning the GPU
 *         texture cache on every drag-and-drop.
 *
 *  3. Non-blocking UI. All ONNX / WebGPU work happens off the main thread in a
 *     dedicated module worker, so the 120 Hz ProMotion UI never drops a frame.
 *
 *  Public API  (see index.html for usage)
 *  --------------------------------------
 *      const studio = new CoreStudio();
 *      await studio.boot({ onBackend, onProgress, onLog });
 *      const result = await studio.recognizeImage(file, onProgress);
 *      const result = await studio.transcribeAudio(file, onProgress);
 *      studio.dispose();
 *
 *  No build step, no bundler — ships as a plain ES module.
 * ============================================================================
 */

'use strict';

/* The pinned, immutable CDN build of Transformers.js (v3 — WebGPU capable).
 * Pinning the version guarantees byte-identical, browser-cacheable weights and
 * a reproducible runtime for everyone who clones the repo. */
const TRANSFORMERS_CDN =
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

/* Model registry. These are small, permissively-licensed, ONNX-exported models
 * that are known to run on both the WebGPU and the WASM execution providers. */
const MODELS = Object.freeze({
  image: {
    task: 'image-classification',
    id: 'Xenova/vit-base-patch16-224',
    // The longest side we ever upload to the worker. Capping this bounds the
    // peak texture footprint without meaningfully hurting ViT accuracy.
    maxEdge: 768,
    topk: 5,
  },
  imageCaption: {
    task: 'image-to-text',
    // BLIP produces markedly richer, more accurate captions than ViT-GPT2 while
    // keeping the exact same image-to-text pipeline interface.
    id: 'Xenova/blip-image-captioning-base',
    // Captioning tolerates a slightly larger canvas; still bounded for memory.
    maxEdge: 768,
    maxNewTokens: 52,
  },
  audio: {
    task: 'automatic-speech-recognition',
    // whisper-base is meaningfully more accurate than whisper-tiny, still small
    // enough to run comfortably on-device. Multilingual, auto language detect.
    id: 'Xenova/whisper-base',
    sampleRate: 16000, // Whisper is hard-wired to 16 kHz mono.
  },
});

/* ---------------------------------------------------------------------------
 *  The worker source.
 *  Defined as a string and instantiated from a Blob so the whole engine ships
 *  as a single file. It is a *module* worker, which lets us `import()` the
 *  Transformers.js ES module and, crucially, lets `navigator.gpu` resolve so
 *  WebGPU runs entirely off the main thread.
 * ------------------------------------------------------------------------- */
const WORKER_SOURCE = /* js */ `
import {
  pipeline,
  env,
  RawImage,
} from '${TRANSFORMERS_CDN}';

// By default we pull weights from the public CDN (cached by the browser after
// the first run). When the app boots in OFFLINE mode (see configure()), we flip
// these so Transformers.js loads weights from a same-origin mirror under
// /static/models/ and makes ZERO external requests — true air-gap operation.
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

// Apply the runtime configuration handed down from the main thread.
function configure(cfg) {
  if (!cfg) return;
  if (cfg.offline) {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = cfg.localPath || '/static/models/';
  } else {
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
  }
}

// SharedArrayBuffer (and therefore multi-threaded WASM) is only available when
// the page is cross-origin isolated. Detect it and scale threads accordingly so
// we never crash on a non-isolated origin.
const CROSS_ORIGIN_ISOLATED = (typeof self.crossOriginIsolated !== 'undefined')
  ? self.crossOriginIsolated
  : false;
env.backends.onnx.wasm.numThreads = CROSS_ORIGIN_ISOLATED
  ? Math.max(1, Math.min(4, (self.navigator?.hardwareConcurrency || 4)))
  : 1;

// Probe for a real WebGPU adapter once. Apple Silicon exposes WebGPU in
// Safari 17+ and Chrome; if anything is missing we transparently fall back to
// the highly optimized WASM (SIMD) backend.
async function detectBackend() {
  try {
    if (!('gpu' in self.navigator)) return 'wasm';
    const adapter = await self.navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

let BACKEND = null;            // 'webgpu' | 'wasm'
const pipelineCache = new Map(); // task-id -> loaded pipeline (one per modality)

// Per-model precision tuning. On WebGPU we lean on fp16/fp32; on WASM we use
// quantized weights (q8) to shrink the download and the resident footprint —
// the Unified Memory pool is precious.
function dtypeFor(taskKey, backend) {
  if (taskKey === 'audio') {
    return backend === 'webgpu'
      ? { encoder_model: 'fp32', decoder_model_merged: 'fp32' }
      : { encoder_model: 'q8',  decoder_model_merged: 'q8' };
  }
  if (taskKey === 'imageCaption') {
    // Encoder-decoder captioner. fp32 on GPU for stability, q8 on CPU to keep
    // the resident footprint small in Unified Memory.
    return backend === 'webgpu' ? 'fp32' : 'q8';
  }
  // image classification
  return backend === 'webgpu' ? 'fp16' : 'q8';
}

async function getPipeline(taskKey, model, id) {
  const cacheKey = taskKey;
  if (pipelineCache.has(cacheKey)) return pipelineCache.get(cacheKey);

  if (!BACKEND) BACKEND = await detectBackend();

  const build = async (device) => pipeline(model.task, model.id, {
    device,
    dtype: dtypeFor(taskKey, device),
    progress_callback: (p) => {
      // Forward HF download/size progress to the UI.
      if (p && (p.status === 'progress' || p.status === 'done' || p.status === 'ready')) {
        self.postMessage({
          type: 'progress', id, scope: 'model',
          file: p.file, status: p.status,
          loaded: p.loaded, total: p.total,
          progress: p.progress,
        });
      }
    },
  });

  let pipe;
  try {
    pipe = await build(BACKEND);
  } catch (err) {
    // A WebGPU init failure on some drivers is recoverable by retrying on WASM.
    if (BACKEND === 'webgpu') {
      self.postMessage({ type: 'log', level: 'warn',
        message: 'WebGPU pipeline init failed, falling back to WASM: ' + (err?.message || err) });
      BACKEND = 'wasm';
      self.postMessage({ type: 'backend', backend: BACKEND });
      pipe = await build(BACKEND);
    } else {
      throw err;
    }
  }

  pipelineCache.set(cacheKey, pipe);
  return pipe;
}

self.onmessage = async (event) => {
  const msg = event.data || {};

  if (msg.type === 'config') {
    configure(msg.config);
    self.postMessage({ type: 'configured', offline: !!msg.config?.offline });
    return;
  }

  if (msg.type === 'detect') {
    BACKEND = await detectBackend();
    self.postMessage({ type: 'backend', backend: BACKEND, isolated: CROSS_ORIGIN_ISOLATED });
    return;
  }

  if (msg.type === 'warmup') {
    try {
      await getPipeline(msg.taskKey, msg.model, msg.id);
      self.postMessage({ type: 'warmed', id: msg.id, taskKey: msg.taskKey });
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'run-image') {
    const t0 = performance.now();
    try {
      const pipe = await getPipeline('image', msg.model, msg.id);
      // Reconstruct a RawImage from the transferred RGBA buffer (zero remote IO).
      const rgba = new Uint8ClampedArray(msg.buffer);
      const image = new RawImage(rgba, msg.width, msg.height, 4).rgb();
      const output = await pipe(image, { topk: msg.topk });
      self.postMessage({
        type: 'result', id: msg.id, kind: 'image',
        ms: Math.round(performance.now() - t0),
        backend: BACKEND,
        data: output,
      });
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'run-caption') {
    const t0 = performance.now();
    try {
      const pipe = await getPipeline('imageCaption', msg.model, msg.id);
      const rgba = new Uint8ClampedArray(msg.buffer);
      const image = new RawImage(rgba, msg.width, msg.height, 4).rgb();
      const output = await pipe(image, {
        max_new_tokens: msg.maxNewTokens || 52,
      });
      self.postMessage({
        type: 'result', id: msg.id, kind: 'caption',
        ms: Math.round(performance.now() - t0),
        backend: BACKEND,
        data: output,
      });
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'run-audio') {
    const t0 = performance.now();
    try {
      const pipe = await getPipeline('audio', msg.model, msg.id);
      const pcm = new Float32Array(msg.buffer); // 16 kHz mono Float32
      const output = await pipe(pcm, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: false,
      });
      self.postMessage({
        type: 'result', id: msg.id, kind: 'audio',
        ms: Math.round(performance.now() - t0),
        backend: BACKEND,
        data: output,
      });
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) });
    }
    return;
  }
};

self.postMessage({ type: 'ready' });
`;

/* ===========================================================================
 *  CoreStudio — the main-thread façade.
 * ========================================================================= */
export class CoreStudio {
  constructor() {
    /** @type {Worker|null} */
    this._worker = null;
    /** @type {string|null} */
    this._workerUrl = null;
    this._seq = 0;
    /** @type {Map<number, {resolve:Function, reject:Function, onProgress?:Function}>} */
    this._pending = new Map();
    this._backend = 'unknown';
    this._ready = false;
    this._offline = false; // true ⇒ weights served from same-origin mirror

    // Reusable scratch surfaces. Allocating these once (rather than per file)
    // keeps the GPU texture/canvas cache stable across many drag-and-drops.
    this._canvas = null;
    this._ctx = null;

    // Callbacks supplied at boot().
    this._onBackend = () => {};
    this._onLog = () => {};
  }

  get backend() { return this._backend; }
  get ready() { return this._ready; }
  get offline() { return this._offline; }

  /**
   * Boot the engine: spin up the worker, decide online/offline weight source,
   * and detect the best backend.
   * @param {{onBackend?:Function, onLog?:Function, offline?:boolean,
   *          localPath?:string}} [handlers]
   *   `offline` forces same-origin weights; if omitted it is auto-detected by
   *   probing for /static/models/manifest.json.
   */
  async boot(handlers = {}) {
    if (this._worker) return; // idempotent
    this._onBackend = handlers.onBackend || this._onBackend;
    this._onLog = handlers.onLog || this._onLog;

    const localPath = handlers.localPath || '/static/models/';
    this._offline = (typeof handlers.offline === 'boolean')
      ? handlers.offline
      : await this._probeOffline(localPath);

    const blob = new Blob([WORKER_SOURCE], { type: 'text/javascript' });
    this._workerUrl = URL.createObjectURL(blob);
    this._worker = new Worker(this._workerUrl, { type: 'module' });
    this._worker.onmessage = (e) => this._handleMessage(e.data);
    this._worker.onerror = (e) => this._onLog('error', e.message || 'Worker error');

    // Wait until the worker module has finished importing Transformers.js.
    await new Promise((resolve) => {
      const once = (data) => {
        if (data?.type === 'ready') { this._worker.removeEventListener('message', wrap); resolve(); }
      };
      const wrap = (e) => once(e.data);
      this._worker.addEventListener('message', wrap);
    });

    this._ready = true;
    // Hand the weight-source configuration to the worker before anything runs.
    this._worker.postMessage({
      type: 'config',
      config: { offline: this._offline, localPath },
    });
    this._onLog('info', this._offline
      ? `Offline mode: weights served locally from ${localPath}`
      : 'Online mode: weights stream once from CDN, then cache.');
    this._worker.postMessage({ type: 'detect' });
  }

  /** Detect a local model mirror by probing for its manifest. */
  async _probeOffline(localPath) {
    try {
      const res = await fetch(`${localPath}manifest.json`, { method: 'GET', cache: 'no-store' });
      return res.ok;
    } catch {
      return false;
    }
  }

  /** Optionally pre-download a modality's weights so the first real run is instant. */
  warmup(modality) {
    const model = MODELS[modality];
    if (!model || !this._worker) return Promise.resolve();
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ type: 'warmup', id, taskKey: modality, model, kind: modality });
    });
  }

  /* ------------------------------------------------------------------ *
   *  IMAGE  ·  decode → down-sample → transfer → classify
   * ------------------------------------------------------------------ */

  /**
   * @param {File|Blob} file
   * @param {(p:object)=>void} [onProgress]
   * @returns {Promise<{labels:{label:string,score:number}[], ms:number, backend:string}>}
   */
  async recognizeImage(file, onProgress) {
    this._assertReady();
    const model = MODELS.image;
    const { buffer, width, height } = await this._prepareImage(file, model.maxEdge);

    const result = await this._dispatch(
      { type: 'run-image', model, topk: model.topk, buffer, width, height },
      [buffer], // Transferable list — zero-copy move.
      onProgress,
    );

    return {
      labels: (result.data || []).map((d) => ({ label: d.label, score: d.score })),
      ms: result.ms,
      backend: result.backend,
    };
  }

  /**
   * Generate a natural-language caption for an image (image-to-text).
   * Shares the exact same memory-frugal decode/transfer path as recognizeImage.
   * @param {File|Blob} file
   * @param {(p:object)=>void} [onProgress]
   * @returns {Promise<{caption:string, ms:number, backend:string}>}
   */
  async captionImage(file, onProgress) {
    this._assertReady();
    const model = MODELS.imageCaption;
    const { buffer, width, height } = await this._prepareImage(file, model.maxEdge);

    const result = await this._dispatch(
      {
        type: 'run-caption',
        model, maxNewTokens: model.maxNewTokens,
        buffer, width, height,
      },
      [buffer],
      onProgress,
    );

    const first = Array.isArray(result.data) ? result.data[0] : result.data;
    const caption = String(first?.generated_text ?? '').trim();
    return { caption, ms: result.ms, backend: result.backend };
  }

  /**
   * Decode an image off the DOM, down-sample it to `maxEdge`, and return a
   * detachable RGBA buffer ready to be *moved* (zero-copy) into the worker.
   * Releases the decoded bitmap immediately — the biggest Unified Memory win.
   * @returns {Promise<{buffer:ArrayBuffer, width:number, height:number}>}
   */
  async _prepareImage(file, maxEdge) {
    // createImageBitmap uses the platform's hardware decoder.
    let bitmap = await createImageBitmap(file);
    const { width, height } = this._fitWithin(bitmap.width, bitmap.height, maxEdge);

    const { ctx } = this._scratch(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0, width, height);

    // Release the decoded source bitmap NOW (before inference even begins).
    bitmap.close();
    bitmap = null;

    const imageData = ctx.getImageData(0, 0, width, height);
    return { buffer: imageData.data.buffer, width, height };
  }

  /* ------------------------------------------------------------------ *
   *  AUDIO  ·  decode → resample to 16 kHz mono → transfer → transcribe
   * ------------------------------------------------------------------ */

  /**
   * @param {File|Blob} file
   * @param {(p:object)=>void} [onProgress]
   * @returns {Promise<{text:string, ms:number, backend:string}>}
   */
  async transcribeAudio(file, onProgress) {
    this._assertReady();
    const model = MODELS.audio;

    const arrayBuf = await file.arrayBuffer();
    const pcm = await this._decodeToMono16k(arrayBuf, model.sampleRate);

    const buffer = pcm.buffer;
    const result = await this._dispatch(
      { type: 'run-audio', model, buffer },
      [buffer],
      onProgress,
    );

    const text = typeof result.data?.text === 'string'
      ? result.data.text.trim()
      : String(result.data?.text ?? '').trim();

    return { text, ms: result.ms, backend: result.backend };
  }

  /**
   * Decode arbitrary audio (wav/mp3/m4a/…) and resample to mono 16 kHz Float32
   * using an OfflineAudioContext — fully on-device, then close the context so
   * the audio engine's buffers are freed immediately.
   * @returns {Promise<Float32Array>}
   */
  async _decodeToMono16k(arrayBuffer, targetRate) {
    const AC = window.AudioContext || window.webkitAudioContext;
    // A short-lived decode context. We only need it to parse the container.
    const decodeCtx = new AC();
    let decoded;
    try {
      decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
    } finally {
      // decodeAudioData keeps no further state we need.
      decodeCtx.close();
    }

    // Resample + downmix to mono at the target rate via OfflineAudioContext.
    const frames = Math.ceil(decoded.duration * targetRate);
    const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const offline = new OAC(1, Math.max(1, frames), targetRate);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    const channel = rendered.getChannelData(0);

    // Copy into a fresh, tightly-sized Float32Array we can transfer away. The
    // large `decoded` AudioBuffer goes out of scope right after and is GC'd.
    const out = new Float32Array(channel.length);
    out.set(channel);
    return out;
  }

  /* ------------------------------ internals ------------------------------ */

  _scratch(width, height) {
    if (!this._canvas) {
      // OffscreenCanvas keeps the surface off the DOM; fall back for older WebKit.
      this._canvas = ('OffscreenCanvas' in window)
        ? new OffscreenCanvas(width, height)
        : Object.assign(document.createElement('canvas'), { width, height });
      this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
    } else if (this._canvas.width !== width || this._canvas.height !== height) {
      this._canvas.width = width;
      this._canvas.height = height;
    }
    return { canvas: this._canvas, ctx: this._ctx };
  }

  _fitWithin(w, h, maxEdge) {
    const longest = Math.max(w, h);
    if (longest <= maxEdge) return { width: w, height: h };
    const scale = maxEdge / longest;
    return { width: Math.round(w * scale), height: Math.round(h * scale) };
  }

  _dispatch(message, transfer, onProgress) {
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onProgress });
      this._worker.postMessage({ ...message, id }, transfer || []);
    });
  }

  _handleMessage(data) {
    if (!data) return;
    switch (data.type) {
      case 'backend':
        this._backend = data.backend;
        this._onBackend(data.backend, { isolated: data.isolated });
        return;
      case 'log':
        this._onLog(data.level || 'info', data.message);
        return;
      case 'progress': {
        const entry = this._pending.get(data.id);
        if (entry?.onProgress) entry.onProgress(data);
        return;
      }
      case 'warmed': {
        const entry = this._pending.get(data.id);
        if (entry) { this._pending.delete(data.id); entry.resolve(true); }
        return;
      }
      case 'result': {
        const entry = this._pending.get(data.id);
        if (entry) { this._pending.delete(data.id); entry.resolve(data); }
        return;
      }
      case 'error': {
        const entry = this._pending.get(data.id);
        if (entry) { this._pending.delete(data.id); entry.reject(new Error(data.message)); }
        else this._onLog('error', data.message);
        return;
      }
    }
  }

  _assertReady() {
    if (!this._ready || !this._worker) {
      throw new Error('CoreStudio is not booted. Call await studio.boot() first.');
    }
  }

  /** Tear everything down and free every retained handle. */
  dispose() {
    if (this._worker) { this._worker.terminate(); this._worker = null; }
    if (this._workerUrl) { URL.revokeObjectURL(this._workerUrl); this._workerUrl = null; }
    this._pending.forEach((p) => p.reject?.(new Error('CoreStudio disposed')));
    this._pending.clear();
    this._canvas = null;
    this._ctx = null;
    this._ready = false;
  }
}

export { MODELS };
export default CoreStudio;
