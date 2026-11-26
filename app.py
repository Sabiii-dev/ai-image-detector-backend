"""AI Image Detector backend — free Render (Python) service.

Detection provider: Sightengine (genai model) via its hosted API.
No model runs on this server anymore, so results come back in
2-3 seconds instead of ~45 seconds, and there is no 67 MB model
to download or 512 MB memory crash on large photos.

Free plan: 2,000 operations/month; one genai check costs 5 ops,
so about 400 checks/month. When the quota is used up, Sightengine
answers with an error and /detect returns 429.

Setup: set SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET as
environment variables on Render (dashboard -> service ->
Environment). The keys never go into the app or the repo.

Endpoints:
  GET  /        -> {"status": "ok", "model": ...}
  GET  /health  -> {"ready": true/false, ...} (ready = keys set)
  GET  /warmup  -> {"status": "ok"} (nothing to warm up anymore)
  POST /detect  -> multipart field "image" (or raw image bytes).
                   Returns {"ai_probability": 0-1, "verdict": "ai"|"real",
                   "label": "sightengine", "version": null,
                   "model": "sightengine/genai"}

Images are resized to max 1024px before upload so checks stay
fast. Never returns a made-up score: if the keys are missing or
Sightengine errors, /detect answers 503/429/500 with the reason.
"""

import os
import tempfile
from typing import Any, Optional

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_ID = "sightengine/genai"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_SIDE_PX = 1024
SIGHTENGINE_URL = "https://api.sightengine.com/1.0/check.json"

app = FastAPI(title="AI Image Detector backend")

# The Flutter app and the web preview call this API from a browser /
# phone, so cross-origin requests must be allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_image(data: bytes) -> dict:
    """Send one image to Sightengine and map the answer to our contract."""
    api_user = os.environ.get("SIGHTENGINE_API_USER", "")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET", "")
    if not api_user or not api_secret:
        raise RuntimeError(
            "Sightengine keys server par set nahi hain."
        )

    files = {"media": ("image.jpg", data, "image/jpeg")}
    payload = {
        "models": "genai",
        "api_user": api_user,
        "api_secret": api_secret,
    }
    resp = requests.post(
        SIGHTENGINE_URL, data=payload, files=files, timeout=30
    )
    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"Sightengine se ghalat jawab aaya ({resp.status_code}).")

    if resp.status_code == 429 or (
        isinstance(body, dict) and body.get("status") == "failure"
    ):
        message = ""
        if isinstance(body, dict):
            error = body.get("error") or {}
            message = error.get("message", "") if isinstance(error, dict) else ""
        raise RuntimeError(
            "Sightengine quota khatam ya request reject: " + (message or str(resp.status_code))
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Sightengine error: HTTP {resp.status_code}.")

    genai = (body.get("type") or {}).get("genai") or {}
    probability = genai.get("ai_generated")
    if probability is None:
        raise RuntimeError("Sightengine ne koi probability nahi di.")
    probability = max(0.0, min(1.0, float(probability)))
    return {
        "ai_probability": round(probability, 4),
        "verdict": "ai" if probability >= 0.5 else "real",
        "label": "sightengine",
        "version": None,
        "model": MODEL_ID,
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/health")
def health() -> dict:
    ready = bool(
        os.environ.get("SIGHTENGINE_API_USER")
        and os.environ.get("SIGHTENGINE_API_SECRET")
    )
    return {
        "ready": ready,
        "loading": False,
        "error": None if ready else "Sightengine keys set nahi hain.",
        "model": MODEL_ID,
    }


@app.get("/warmup")
def warmup() -> dict:
    """Nothing to warm up (no local model) — kept so the app still works."""
    return {"status": "ok", "model": MODEL_ID}


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

    # Validate the bytes are really an image, then shrink large photos
    # to max 1024px so the upload and the check stay fast.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with Image.open(tmp_path) as img:
            img.verify()
        with Image.open(tmp_path) as img:
            img = img.convert("RGB")
            if max(img.size) > MAX_SIDE_PX:
                img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
            small_path = tmp_path + ".small.jpg"
            img.save(small_path, "JPEG", quality=90)
        with open(small_path, "rb") as fh:
            resized = fh.read()
        os.remove(small_path)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Ye valid image nahi hai."},
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    try:
        return _check_image(resized)
    except RuntimeError as exc:
        message = str(exc)
        status = 429 if "quota" in message.lower() else 503
        return JSONResponse(status_code=status, content={"error": message})
    except Exception as exc:  # noqa: BLE001 - report the real error
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis fail ho gaya: {exc}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860"))
    )
