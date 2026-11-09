"""
FastAPI Server for Balochistan Prompt Browser & Text-to-Image Generation Dashboard.
Serves static files, dataset API, and provides a backend /api/generate-image endpoint
supporting HF Inference API, OpenAI DALL-E, Pollinations AI, and Stable Horde free fallback.
All API keys are securely managed via environment variables on the backend.
"""
import os
import json
import time
import random
import base64
import socket
import threading
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
CLIENT_AGENT = "BalochistanBenchmark:1.0"
# Serialize free-tier calls so rapid UI clicks do not trip IP rate limits.
_GENERATE_LOCK = threading.Lock()
_LAST_HORDE_AT = 0.0
HORDE_MIN_INTERVAL_SEC = 8.0
HORDE_HOSTS = (
    "https://stablehorde.net",
    "https://aihorde.net",
)

# Optional: Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

app = FastAPI(title="Balochistan Prompt Browser & T2I Generator")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    width: int = 1024
    height: int = 1024
    enhance_faces: bool = True

DEFAULT_NEGATIVE = (
    "ugly, deformed, disfigured, mutated, bad anatomy, bad proportions, "
    "extra limbs, fused fingers, poorly drawn face, asymmetric eyes, "
    "blurry, lowres, jpeg artifacts, watermark, text, logo, "
    "cartoon, anime, 3d render, cgi, doll, mannequin, plastic skin"
)

FACE_NEGATIVE = (
    "melted face, distorted face, extra eyes, missing eyes, crossed eyes, "
    "bad teeth, cloned face, uncanny valley, wax figure, statue face, "
    "blurry face, low detail face, animal face, alien face"
)

# Photoreal models (prompt-faithful realism).
HORDE_FACE_MODELS = [
    "AbsoluteReality",
    "ICBINP - I Can't Believe It's Not Photography",
    "Juggernaut XL",
    "Realistic Vision",
    "Edge Of Realism",
    "majicMIX realistic",
    "Photon",
    "Deliberate",
]
def _clamp_size(width: int, height: int, max_side: int = 1024) -> tuple[int, int]:
    """Keep sizes even and bounded for quality backends."""
    w = max(512, min(int(width or 1024), max_side))
    h = max(512, min(int(height or 1024), max_side))
    return w - (w % 64), h - (h % 64)

def _fresh_seed(client_seed: int | None = None) -> int:
    """Always produce a unique seed so providers cannot return a cached twin."""
    base = int(client_seed or 0) & 0x7FFFFFFF
    mix = (time.time_ns() ^ random.getrandbits(32) ^ os.getpid()) & 0x7FFFFFFF
    seed = (base ^ mix ^ int(time.time() * 1000)) & 0x7FFFFFFF
    return seed if seed != 0 else random.randint(1, 2_147_483_647)

def _prompt_mentions_person(prompt: str) -> bool:
    t = (prompt or "").lower()
    keys = (
        "person", "people", "man", "woman", "boy", "girl", "face", "portrait",
        "baloch", "teacher", "craft", "fisherman", "shepherd", "human",
        "mother", "father", "child", "worker", "artisan",
    )
    return any(k in t for k in keys)

def build_image_prompt(prompt: str, seed: int = 0, enhance_faces: bool = False) -> str:
    """Keep the user's prompt as the scene; optionally add light face-quality cues."""
    scene = (prompt or "").strip()
    if not scene:
        return scene
    parts = [scene, "Photorealistic photograph, sharp detail, natural lighting, high quality"]
    if enhance_faces and _prompt_mentions_person(scene):
        parts.append(
            "natural realistic human face, clear eyes, detailed facial features, "
            "skin texture, symmetrical face, looking toward camera when appropriate"
        )
    parts.append(f"seed {seed}")
    return ". ".join(parts)

def _bytes_to_data_uri(img_bytes: bytes, content_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{content_type};base64,{b64}"

def _fetch_image_url(url: str, timeout: int = 90) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "image/*,*/*",
            "Referer": "https://pollinations.ai/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("content-type", "image/jpeg")

def generate_via_openai(prompt: str, api_key: str):
    url = "https://api.openai.com/v1/images/generations"
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": CLIENT_AGENT,
        }
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        b64 = res["data"][0]["b64_json"]
        return f"data:image/png;base64,{b64}"

def generate_via_huggingface(prompt: str, api_key: str):
    model_id = os.environ.get("HF_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {"inputs": prompt}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": CLIENT_AGENT,
        }
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        img_bytes = resp.read()
        content_type = resp.headers.get("content-type", "image/jpeg")
        return _bytes_to_data_uri(img_bytes, content_type)

def generate_via_pollinations(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024):
    """Pollinations image endpoint — prefer Flux at high resolution."""
    width, height = _clamp_size(width, height, max_side=1024)
    encoded = urllib.parse.quote(prompt)
    api_key = os.environ.get("POLLINATIONS_API_KEY") or os.environ.get("POLLINATIONS_KEY")
    # seed + unique bust CDN/prompt caches that ignore seed alone
    bust = f"seed={seed}&nologo=true&nofeed=true&unique={seed}-{time.time_ns()}"

    if api_key:
        urls = [
            f"https://gen.pollinations.ai/image/{encoded}?model=flux&width={width}&height={height}&{bust}",
            f"https://gen.pollinations.ai/image/{encoded}?model=flux&width=768&height=768&{bust}",
        ]
        auth_headers = {"Authorization": f"Bearer {api_key}"}
    else:
        urls = [
            f"https://image.pollinations.ai/prompt/{encoded}?model=flux&width={width}&height={height}&{bust}",
            f"https://image.pollinations.ai/prompt/{encoded}?model=flux&width=768&height=768&{bust}",
            f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&{bust}",
        ]
        auth_headers = {}

    last_error = None
    for attempt in range(2):
        for url in urls:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "image/*,*/*",
                        "Referer": "https://pollinations.ai/",
                        **auth_headers,
                    },
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    img_bytes = resp.read()
                    if not img_bytes or len(img_bytes) < 100:
                        raise RuntimeError("empty image payload")
                    content_type = resp.headers.get("content-type", "image/jpeg")
                    if "json" in content_type:
                        raise RuntimeError(img_bytes[:200].decode("utf-8", errors="replace"))
                    return _bytes_to_data_uri(img_bytes, content_type)
            except urllib.error.HTTPError as http_err:
                body = ""
                try:
                    body = http_err.read().decode("utf-8", errors="replace")[:180]
                except Exception:
                    pass
                if http_err.code == 402:
                    # Try next URL/quality tier; only fail after all options.
                    last_error = RuntimeError(
                        "Pollinations free quota exceeded (HTTP 402)."
                    )
                    continue
                last_error = RuntimeError(
                    f"Pollinations HTTP {http_err.code}: {http_err.reason} {body}".strip()
                )
            except (TimeoutError, socket.timeout) as e:
                last_error = RuntimeError(f"timed out: {e}")
            except Exception as e:
                last_error = RuntimeError(str(e))
        if attempt < 1:
            time.sleep(1.0)
    raise last_error or RuntimeError("Pollinations failed")

def _horde_json(url: str, *, data: bytes | None = None, api_key: str = "0000000000",
                method: str | None = None, timeout: int = 30) -> dict:
    headers = {"Client-Agent": CLIENT_AGENT, "apikey": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _submit_horde_with_retries(payload: dict, api_key: str) -> tuple[str, str]:
    """Submit to Horde, backing off on HTTP 429 rate limits."""
    global _LAST_HORDE_AT
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    # Honor spacing between submits from this process.
    wait = HORDE_MIN_INTERVAL_SEC - (time.time() - _LAST_HORDE_AT)
    if wait > 0:
        time.sleep(wait)

    for attempt in range(6):
        for host in HORDE_HOSTS:
            try:
                _LAST_HORDE_AT = time.time()
                job = _horde_json(
                    f"{host}/api/v2/generate/async",
                    data=body,
                    api_key=api_key,
                    method="POST",
                    timeout=45,
                )
                job_id = job.get("id")
                if not job_id:
                    raise RuntimeError(f"Stable Horde submit failed: {job}")
                return host, job_id
            except urllib.error.HTTPError as http_err:
                detail = ""
                retry_after = http_err.headers.get("Retry-After") if http_err.headers else None
                try:
                    detail = http_err.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                last_error = RuntimeError(
                    f"HTTP {http_err.code}: {http_err.reason} {detail}".strip()
                )
                if http_err.code == 429:
                    # Anonymous IP is throttled — wait and retry.
                    delay = float(retry_after) if retry_after and str(retry_after).isdigit() else min(
                        12 * (2 ** attempt), 90
                    )
                    time.sleep(delay)
                    break  # next attempt
                if http_err.code in (502, 503, 504):
                    time.sleep(3 + attempt)
                    continue
                # Other hard errors: try alternate host, else fail this attempt.
                continue
            except Exception as e:
                last_error = e
                time.sleep(2)
                continue
        else:
            continue
    raise RuntimeError(f"Stable Horde rate-limited / unavailable: {last_error}")

def _wait_horde_image(host: str, job_id: str, api_key: str, timeout_sec: int = 300) -> str:
    """Poll a Horde job and return a data-URI image."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(3)
        try:
            status = _horde_json(
                f"{host}/api/v2/generate/check/{job_id}",
                api_key=api_key,
                timeout=30,
            )
        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                time.sleep(8)
                continue
            raise
        if status.get("faulted"):
            raise RuntimeError("Stable Horde job faulted")
        if status.get("done"):
            break
    else:
        raise RuntimeError("Stable Horde timed out waiting for workers")

    final = _horde_json(
        f"{host}/api/v2/generate/status/{job_id}",
        api_key=api_key,
        timeout=45,
    )
    gens = final.get("generations") or []
    if not gens:
        raise RuntimeError(f"Stable Horde returned no generations: {final}")
    img_ref = gens[0].get("img") or ""
    if not img_ref:
        raise RuntimeError("Stable Horde generation missing image payload")
    if img_ref.startswith("http"):
        img_bytes, content_type = _fetch_image_url(img_ref, timeout=60)
        return _bytes_to_data_uri(img_bytes, content_type)
    return _bytes_to_data_uri(base64.b64decode(img_ref), "image/webp")

def generate_via_stable_horde(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024,
                              negative_prompt: str = "", enhance_faces: bool = True):
    """Photoreal generation via Horde; face restore when enhance_faces=True."""
    if enhance_faces:
        base_w, base_h = 768, 1024  # portrait frame — face fills more of the image
    else:
        base_w, base_h = _clamp_size(width, height, max_side=1024)
    api_key = os.environ.get("STABLE_HORDE_API_KEY") or "0000000000"
    neg = ", ".join(
        p for p in (
            negative_prompt.strip(),
            DEFAULT_NEGATIVE,
            FACE_NEGATIVE if enhance_faces else "",
        ) if p
    )

    # Escalate from best face quality → most compatible if workers reject jobs.
    if enhance_faces:
        attempts = [
            {
                "models": list(HORDE_FACE_MODELS),
                "post_processing": ["CodeFormers"],
                "steps": 40,
            },
            {
                "models": [
                    "AbsoluteReality",
                    "Realistic Vision",
                    "ICBINP - I Can't Believe It's Not Photography",
                    "Deliberate",
                ],
                "post_processing": ["GFPGAN"],
                "steps": 35,
            },
            {
                "models": [
                    "AbsoluteReality",
                    "Realistic Vision",
                    "Deliberate",
                    "Dreamshaper",
                ],
                "post_processing": [],
                "steps": 30,
            },
            {
                "models": [],
                "post_processing": [],
                "steps": 25,
            },
        ]
    else:
        attempts = [
            {
                "models": list(HORDE_FACE_MODELS),
                "post_processing": [],
                "steps": 28,
            },
            {
                "models": [],
                "post_processing": [],
                "steps": 25,
            },
        ]

    last_error: Exception | None = None
    for attempt in attempts:
        params = {
            "width": base_w,
            "height": base_h,
            "steps": attempt["steps"],
            "n": 1,
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 6.5,
            "seed": str(seed),
            "clip_skip": 2,
            "karras": True,
        }
        if attempt["post_processing"]:
            params["post_processing"] = attempt["post_processing"]
        payload = {
            "prompt": f"{prompt} ### {neg}",
            "params": params,
            "nsfw": False,
            "censor_nsfw": False,
            "models": attempt["models"],
            "r2": True,
            "shared": True,
            "slow_workers": True,
        }
        try:
            host, job_id = _submit_horde_with_retries(payload, api_key)
            return _wait_horde_image(host, job_id, api_key)
        except Exception as e:
            last_error = e
            time.sleep(2)
            continue
    raise RuntimeError(f"Stable Horde face generation failed: {last_error}")

@app.post("/api/generate-image")
def api_generate_image(req: GenerateRequest):
    prompt_text = req.prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    # Unique seed + prompt that follows the user's text (not a fixed template).
    seed = _fresh_seed(req.seed)
    enhance_faces = bool(req.enhance_faces)
    image_prompt = build_image_prompt(
        prompt_text, seed=seed, enhance_faces=enhance_faces
    )

    openai_key = os.environ.get("OPENAI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

    if not _GENERATE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another image is still generating. Wait for it to finish, then try again.",
        )

    error_logs = []
    try:
        if openai_key:
            try:
                image_uri = generate_via_openai(image_prompt, openai_key)
                return {
                    "status": "success",
                    "provider": "OpenAI DALL-E-3",
                    "image": image_uri,
                    "seed": seed,
                    "prompt_used": prompt_text,
                }
            except Exception as e:
                error_logs.append(f"OpenAI error: {e}")

        if hf_token:
            try:
                image_uri = generate_via_huggingface(image_prompt, hf_token)
                return {
                    "status": "success",
                    "provider": "Hugging Face Inference",
                    "image": image_uri,
                    "seed": seed,
                    "prompt_used": prompt_text,
                }
            except Exception as e:
                error_logs.append(f"HuggingFace error: {e}")

        if enhance_faces:
            try:
                image_uri = generate_via_stable_horde(
                    image_prompt,
                    seed=seed,
                    width=req.width,
                    height=req.height,
                    negative_prompt=req.negative_prompt,
                    enhance_faces=True,
                )
                return {
                    "status": "success",
                    "provider": "Stable Horde Face HQ (CodeFormers)",
                    "image": image_uri,
                    "seed": seed,
                    "prompt_used": prompt_text,
                }
            except Exception as e:
                error_logs.append(f"Stable Horde: {e}")

            try:
                image_uri = generate_via_pollinations(
                    image_prompt, seed=seed, width=req.width, height=req.height
                )
                return {
                    "status": "success",
                    "provider": "Pollinations Flux",
                    "image": image_uri,
                    "seed": seed,
                    "prompt_used": prompt_text,
                }
            except Exception as e:
                error_logs.append(f"Pollinations: {e}")
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Image generation failed (free APIs are rate-limited right now). "
                        "Wait ~1 minute and try again, or set STABLE_HORDE_API_KEY / "
                        f"POLLINATIONS_API_KEY in .env. Details: {' | '.join(error_logs)}"
                    ),
                )

        try:
            image_uri = generate_via_pollinations(
                image_prompt, seed=seed, width=req.width, height=req.height
            )
            return {
                "status": "success",
                "provider": "Pollinations Flux",
                "image": image_uri,
                "seed": seed,
                "prompt_used": prompt_text,
            }
        except Exception as e:
            error_logs.append(f"Pollinations: {e}")

        try:
            image_uri = generate_via_stable_horde(
                image_prompt,
                seed=seed,
                width=req.width,
                height=req.height,
                negative_prompt=req.negative_prompt,
                enhance_faces=False,
            )
            return {
                "status": "success",
                "provider": "Stable Horde (prompt-faithful)",
                "image": image_uri,
                "seed": seed,
                "prompt_used": prompt_text,
            }
        except Exception as e:
            error_logs.append(f"Stable Horde: {e}")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Image generation failed (free APIs are rate-limited right now). "
                    "Wait ~1 minute and try again, or set STABLE_HORDE_API_KEY / "
                    f"POLLINATIONS_API_KEY in .env. Details: {' | '.join(error_logs)}"
                ),
            )
    finally:
        _GENERATE_LOCK.release()

# Mount data directory so frontend can fetch prompts
app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")

# Mount web directory
app.mount("/web", StaticFiles(directory=str(PROJECT_ROOT / "web"), html=True), name="web")

@app.get("/")
def read_root():
    return FileResponse(PROJECT_ROOT / "web" / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
