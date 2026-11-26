"""AI Image Detector backend — free Render (Python) service.

Model: wiltodelta/raiw-source-classify (Apache 2.0, 67 MB, CPU-only).
The model runs locally on this server via the `remove-ai-watermarks`
package: no token, no API key, no login is needed anywhere — the
Flutter app just POSTs an image and gets back a probability.

Endpoints:
  GET  /        -> {"status": "ok", "model": ...}
  GET  /health  -> {"ready": true/false, "loading": bool}
  GET  /warmup  -> starts the model download/load in the background
  POST /detect  -> multipart field "image" (or raw image bytes).
                   Returns {"ai_probability": 0-1, "verdict": "ai"|"real",
                   "label": provider label or "unknown",
                   "version": model/export subfamily or null,
                   "model": "wiltodelta/raiw-source-classify"}

The model is loaded lazily on the first /detect (or /warmup) call so
the service starts fast and /health answers before the ~67 MB weights
have downloaded. On Render free (512 MB RAM, 0.1 CPU) the first call
after a cold start can take 30-60 seconds; the app waits up to 120s.
Never returns a made-up score: if the model is not ready, /detect
answers 503, and a bad image answers 400.
"""

import os
import tempfile
import threading
from typing import Any, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_ID = "wiltodelta/raiw-source-classify"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
# Inference se pehle image ko is size tak chhota kar diya jata hai
# (lambi side). Bari phone photos (4000px, 4-5 MB) slow hoti hain
# aur free server ki 512 MB memory bhar ke crash (502) kara deti
# hain; 1024px pe detection tez hoti hai aur accuracy pe koi
# khaas farq nahi parta.
MAX_SIDE_PX = 1024

app = FastAPI(title="AI Image Detector backend")

# The Flutter app and the web preview call this API from a browser /
# phone, so cross-origin requests must be allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_load_lock = threading.Lock()
_model_ready = False
_model_loading = False
_load_error: Optional[str] = None


def _get_field(result: Any, name: str, default: Any = None) -> Any:
    """Read `name` from either an object attribute or a dict key."""
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _load_model() -> None:
    """Download + load the raiw weights once. Runs under _load_lock."""
    global _model_ready, _model_loading, _load_error
    with _load_lock:
        if _model_ready or _model_loading:
            return
        _model_loading = True
    try:
        # Imported here (not at module top) so the service can start
        # and answer /health even if the package is still installing
        # or the weights have not downloaded yet.
        import remove_ai_watermarks as raiw  # noqa: F401  (import warms the package)

        # The weights (~67 MB) download on the first classify call,
        # not on import — so this flag only means the package imported
        # successfully. The first /detect after a cold start therefore
        # takes longer (download + load), and the app waits up to 120s.
        _model_ready = True
    except Exception as exc:  # noqa: BLE001 - report the real error
        _load_error = str(exc)
    finally:
        _model_loading = False


def _classify(path: str) -> dict:
    """Run raiw.classify_source on a saved image file."""
    import remove_ai_watermarks as raiw

    result = raiw.classify_source(path)
    probability = _get_field(result, "synthetic_probability")
    if probability is None:
        raise ValueError("Model ne koi probability nahi di.")
    probability = max(0.0, min(1.0, float(probability)))
    detected = _get_field(result, "synthetic_detected")
    if detected is None:
        detected = probability >= 0.5
    label = _get_field(result, "label") or "unknown"
    version = _get_field(result, "version_subfamily")
    return {
        "ai_probability": round(probability, 4),
        "verdict": "ai" if bool(detected) else "real",
        "label": str(label),
        "version": str(version) if version else None,
        "model": MODEL_ID,
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/health")
def health() -> dict:
    return {
        "ready": _model_ready,
        "loading": _model_loading,
        "error": _load_error,
        "model": MODEL_ID,
    }


@app.get("/warmup")
def warmup() -> dict:
    """Start the model load in the background (ping after deploy)."""
    thread = threading.Thread(target=_load_model, daemon=True)
    thread.start()
    return {"status": "loading", "model": MODEL_ID}


@app.post("/detect")
async def detect(request: Request, image: Optional[UploadFile] = File(None)):
    """Analyse one image. Multipart field "image", or raw image bytes."""
    data: bytes
    if image is not None:
        data = await image.read()
    else:
        data = await request.body()

    if not data:
        return JSONResponse(
            status_code=400, content={"error": "Koi image nahi mili."}
        )
    if len(data) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=400,
            content={"error": "Image 10 MB se bari hai, chhoti image bhejein."},
        )

    # Validate that the bytes are really an image before classifying.
    suffix = ".jpg"
    small_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with Image.open(tmp_path) as img:
            img.verify()
            suffix = "." + (img.format or "jpg").lower().replace("jpeg", "jpg")
        os.rename(tmp_path, tmp_path + suffix)
        tmp_path = tmp_path + suffix
        # Bari image ko inference se pehle chhota kar lo: RGB mein
        # badlo, lambi side MAX_SIDE_PX tak rakho, JPEG mein save karo.
        # Chhoti image pehle se hi ho to bilkul waisi ki waisi rehti hai.
        with Image.open(tmp_path) as img:
            img = img.convert("RGB")
            if max(img.size) > MAX_SIDE_PX:
                img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
            small_path = tmp_path + ".small.jpg"
            img.save(small_path, "JPEG", quality=90)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Ye valid image nahi hai."},
        )

    try:
        _load_model()
        if not _model_ready:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model load ho raha hai, thodi der baad dobara try karo."
                    + (f" ({_load_error})" if _load_error else "")
                },
            )
        result = _classify(small_path or tmp_path)
        return result
    except Exception as exc:  # noqa: BLE001 - report the real error
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis fail ho gaya: {exc}"},
        )
    finally:
        for path in (tmp_path, small_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860"))
    )
