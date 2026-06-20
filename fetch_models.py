#!/usr/bin/env python3
"""
Core-CoreOS Studio · fetch_models.py
============================================================================
OPTIONAL — download the AI model weights into a same-origin mirror so the app
runs in fully air-gapped, ZERO-external-request mode.

Run once:
    python fetch_models.py

It mirrors the Hugging Face repos used by the engine into:
    app/static/models/<org>/<repo>/...
and writes app/static/models/manifest.json. On the next page load the engine
auto-detects that manifest and serves every weight locally — no CDN, no
internet. To go back online, just delete the app/static/models/ directory.

Stdlib only (urllib) — no extra dependencies. Re-running is safe: existing
files are skipped, so it resumes interrupted downloads.

Keep the IDs below in sync with the MODELS map in
app/static/apple_optimized_core.js.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Repos the engine loads (image classifier, image captioner, speech model).
REPOS = [
    "Xenova/vit-base-patch16-224",
    "Xenova/blip-image-captioning-base",
    "Xenova/whisper-base",
]

HF_API = "https://huggingface.co/api/models/{repo}"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"

# We only need config/tokenizer JSON and the ONNX graphs — skip the original
# PyTorch/TF checkpoints, READMEs, and images to keep the mirror lean.
KEEP_SUFFIXES = (".json", ".txt", ".onnx", ".onnx_data", ".bin")
SKIP_NAMES = {"pytorch_model.bin", "tf_model.h5", "model.safetensors", "flax_model.msgpack"}

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "app" / "static" / "models"


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "core-coreos-studio/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_files(repo: str) -> list[str]:
    """Return the repo's file paths via the HF API (handles nested onnx/ dirs)."""
    info = http_json(HF_API.format(repo=repo))
    files = [s["rfilename"] for s in info.get("siblings", [])]
    out = []
    for f in files:
        name = f.rsplit("/", 1)[-1]
        if name in SKIP_NAMES:
            continue
        if f.endswith(KEEP_SUFFIXES):
            out.append(f)
    return out


def download(repo: str, path: str, dest: Path) -> int:
    url = HF_FILE.format(repo=repo, path=path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "core-coreos-studio/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    print(f"→ Mirroring {len(REPOS)} model repo(s) into {DEST.relative_to(ROOT)}\n")
    manifest = {"generated": datetime.now(timezone.utc).isoformat(), "models": []}
    grand_total = 0

    for repo in REPOS:
        print(f"▼ {repo}")
        try:
            files = list_files(repo)
        except urllib.error.URLError as exc:
            print(f"  ✗ could not list files: {exc}")
            return 1

        repo_bytes = 0
        for path in files:
            dest = DEST / repo / path
            if dest.exists():
                print(f"  · skip {path} (exists)")
                continue
            try:
                size = download(repo, path, dest)
            except urllib.error.HTTPError as exc:
                # Not every dtype variant exists for every repo; that's fine.
                print(f"  · miss {path} (HTTP {exc.code})")
                continue
            except urllib.error.URLError as exc:
                print(f"  ✗ fail {path}: {exc}")
                return 1
            repo_bytes += size
            print(f"  ✓ {path}  ({human(size)})")

        grand_total += repo_bytes
        manifest["models"].append({"repo": repo, "files": len(files)})
        print(f"  = {human(repo_bytes)} downloaded\n")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✅ Done. {human(grand_total)} fetched. Wrote {DEST.relative_to(ROOT)}/manifest.json")
    print("   The studio will now run fully offline (air-gapped) on next load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
