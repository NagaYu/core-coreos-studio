<div align="center">

# 🛡️ Core-CoreOS Studio

### On-device multimodal AI, powered entirely by Apple Silicon.

**Image recognition · Speech-to-text — running 100% in your browser.**
**Zero data leak. Zero server cost. Zero compromise.**

`WebGPU` · `Transformers.js` · `Apple Silicon (M-series)` · `Privacy-first`

</div>

---

## Why this exists

Apple Silicon put a supercomputer in your lap. Its **Unified Memory** architecture
lets the CPU, GPU, and Neural Engine share one ultra-fast memory pool — which is
*exactly* the kind of hardware modern AI craves.

**Core-CoreOS Studio** unlocks that power from inside a plain browser tab. Drop in
an image or an audio file and it is recognized or transcribed **on your device**,
using WebGPU. Nothing is uploaded. Nothing is logged. There is no inference
backend to pay for, scale, or trust.

> For teams handling confidential audio recordings, medical images, legal
> documents, or any sensitive media: the data **physically never leaves the
> machine**. The only network request the app ever makes is a one-time,
> cacheable download of the open model weights from a public CDN. After that,
> you can disconnect from Wi-Fi entirely and it keeps working.

---

## ✨ Features

| | |
|---|---|
| 🖼️ **Vision · Classify** | Image classification with a Vision Transformer (ViT). Drop a photo, get ranked labels with confidence — in milliseconds. |
| 📝 **Vision · Describe** | Image captioning (ViT-GPT2). One tap switches the Vision card from labels to a natural-language description of the scene. |
| 🎙️ **Audio** | Speech-to-text with Whisper. Multilingual, automatic language detection, long-file chunking. |
| ✈️ **Air-gap mode** | Run [`fetch_models.py`](fetch_models.py) once to mirror the weights locally — the app then makes **zero external requests**, ever. |
| ⚡ **WebGPU first** | Runs on the Apple GPU via WebGPU; transparently falls back to highly optimized WASM SIMD. |
| 🧠 **Unified-Memory tuned** | Media is down-sampled to the model's exact footprint and **moved** (zero-copy) into a Web Worker. Bitmaps, object URLs, and audio contexts are released the instant they're consumed — no leaks, no GPU texture churn. |
| 🧵 **Never blocks the UI** | All inference happens off the main thread in a dedicated module worker, keeping the ProMotion UI buttery at 120 Hz. |
| 🔒 **Provably private** | A live "Network egress: 0 bytes" indicator. No analytics, no telemetry, no backend. |
| 🎨 **HIG-grade design** | Aluminium-silver + glassmorphism, full light/dark, reduced-motion aware, keyboard accessible. |

---

## 🚀 Quick start (Mac, ~30 seconds)

```bash
# 1. Clone
git clone https://github.com/your-name/core-coreos-studio.git
cd core-coreos-studio

# 2. Create an isolated environment & install the 3 tiny server deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch
uvicorn app.main:app --reload
```

Then open **<http://127.0.0.1:8000>** in **Safari 17+** or **Chrome** and drop in a file.

> 💡 On the very first run, each model (a few tens of MB) streams once from the
> CDN and is cached by the browser. Every run after that is fully offline.

### Even shorter

```bash
pip install -r requirements.txt && python -m app.main
```

---

## ✈️ Going fully offline (air-gap mode)

By default, model weights stream once from a public CDN and are then cached by
the browser. For environments that must make **zero external requests** — even
on first run — mirror the weights into the app itself:

```bash
python fetch_models.py        # downloads weights into app/static/models/
```

That's it. On the next page load the engine **auto-detects** the local mirror
(via `app/static/models/manifest.json`) and the privacy bar flips to
**"Weights: local mirror (air-gapped)"**. Every byte — code, weights, and your
media — now lives on your machine. To return to CDN mode, delete the
`app/static/models/` directory.

> The mirror is **git-ignored** on purpose: weights are large and should be
> fetched on demand, never committed. The repo stays tiny.

---

## 🧩 How it works

```
┌──────────────────────────── Your Mac (the browser tab) ────────────────────────────┐
│                                                                                     │
│   index.html (UI)                                                                    │
│        │  drop file                                                                  │
│        ▼                                                                             │
│   apple_optimized_core.js  ──┐  decode + down-sample on the main thread              │
│   (CoreStudio façade)        │  (createImageBitmap / OfflineAudioContext)           │
│        │                     │                                                       │
│        │  transfer buffer ───┘  ← zero-copy move, not a clone                        │
│        ▼                                                                             │
│   ┌─────────────── Web Worker (module) ───────────────┐                             │
│   │  Transformers.js  →  ONNX Runtime                  │                             │
│   │      WebGPU  (Apple GPU)   ──or──  WASM SIMD (CPU) │                             │
│   │  ViT · Whisper                                     │                             │
│   └────────────────────────────────────────────────────┘                           │
│        │  result (labels / transcript)                                              │
│        ▼                                                                             │
│   rendered in the glass UI                                                           │
│                                                                                     │
│   ✅ No data ever crosses this box's boundary.                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

       FastAPI (app/main.py) only ships these static files to localhost.
       It never receives your media. It is a file host, not an AI backend.
```

### Memory hygiene (the Apple Silicon part)

Unified Memory is shared and finite, so the engine is fanatical about it:

- **Down-sample before transfer.** Images are scaled so their longest edge ≤ 768 px
  *before* leaving the main thread, capping peak texture memory.
- **Zero-copy hand-off.** Pixel and PCM buffers are passed as
  [Transferable objects](https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Transferable_objects) —
  ownership *moves* to the worker; the bytes are never duplicated.
- **Immediate release.** `ImageBitmap.close()`, `URL.revokeObjectURL()`, and
  `AudioContext.close()` are called the moment a resource is consumed.
- **Stable scratch surfaces.** One reused `OffscreenCanvas` instead of allocating
  a new one per drop — the GPU texture cache stays calm.
- **Precision tuning.** `fp16` on WebGPU, quantized `q8` on WASM, to shrink both the
  download and the resident footprint.

---

## ⚙️ Configuration

Models are declared in [`app/static/apple_optimized_core.js`](app/static/apple_optimized_core.js)
in the `MODELS` map. Swap in any ONNX-exported model from the Hugging Face hub:

```js
const MODELS = {
  image:        { task: 'image-classification',         id: 'Xenova/vit-base-patch16-224',     ... },
  imageCaption: { task: 'image-to-text',                id: 'Xenova/vit-gpt2-image-captioning', ... },
  audio:        { task: 'automatic-speech-recognition', id: 'Xenova/whisper-tiny',             ... },
};
```

Want higher accuracy? Try `Xenova/whisper-base` or a larger ViT — the engine
auto-selects precision and backend for whatever you choose. If you swap models
**and** use air-gap mode, mirror the new ids by editing the `REPOS` list in
[`fetch_models.py`](fetch_models.py).

---

## 🖥️ Requirements

- **macOS** on Apple Silicon (M1 or newer) for full WebGPU acceleration
  *(also runs on Intel Macs, iPad, and iPhone via the WASM fallback).*
- **Safari 17+** or a recent **Chrome / Edge** with WebGPU enabled.
- **Python 3.9+** to run the local static host.

---

## 🔐 Privacy guarantee

| Question | Answer |
|---|---|
| Are my images / audio uploaded? | **No.** They are read with the File API and processed in-browser. |
| Does the server store anything? | **No.** `main.py` serves static files and exposes only `/` and `/healthz`. |
| What network requests are made? | Only the one-time model-weights download from a public CDN. Mirror them locally to go fully air-gapped. |
| Telemetry / analytics? | **None.** Inspect the Network tab — egress stays at 0 bytes during inference. |

---

## 📂 Project structure

```
core-coreos-studio/
├── app/
│   ├── main.py                      # tiny FastAPI static host
│   ├── static/
│   │   ├── apple_optimized_core.js  # the on-device inference engine + worker
│   │   └── models/                  # (optional) local weight mirror — git-ignored
│   └── templates/
│       └── index.html               # the glassmorphism studio UI
├── fetch_models.py                  # optional: mirror weights for air-gap mode
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📜 License

MIT — use it, fork it, ship it. Model weights are governed by their respective
licenses on the Hugging Face hub.

<div align="center">
<sub>Built for developers who believe the best place to run AI is the device in front of you.</sub>
</div>
