"""Shared helpers for model backends: errors, secrets, images, HTTP, schema check."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 8
IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class ModelError(RuntimeError):
    """A backend call failed (HTTP error, bad output, schema mismatch)."""


# ---- secrets ---------------------------------------------------------------

def _parse_dotenv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip("'\"")
        out[k.strip().removeprefix("export ").strip()] = v
    return out


def get_secret(name: str, env: Optional[dict] = None, dotenv_paths: Optional[list] = None) -> Optional[str]:
    """Environment first, then .env (cwd, then repo root). Empty string = unset."""
    env = os.environ if env is None else env
    if env.get(name):
        return env[name]
    if env is not os.environ and dotenv_paths is None:
        return None                      # tests pass an explicit env: never read real files
    for p in dotenv_paths or [Path.cwd() / ".env", REPO_ROOT / ".env"]:
        try:
            v = _parse_dotenv(Path(p).read_text()).get(name)
        except OSError:
            continue
        if v:
            return v
    return None


def redact(text: str, *secrets: Optional[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


# ---- images ----------------------------------------------------------------

def load_image(path: str) -> tuple[str, str]:
    """-> (mime, base64). Raises ModelError for a missing/odd/oversized image."""
    p = Path(path)
    mime = IMAGE_MIME.get(p.suffix.lower())
    if mime is None:
        raise ModelError(f"unsupported image type: {p.suffix!r}")
    try:
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise ModelError(f"image too large ({size} bytes): {p.name}")
        return mime, base64.b64encode(p.read_bytes()).decode()
    except OSError as e:
        raise ModelError(f"cannot read image {p.name}: {e.strerror}") from None


def load_images(paths) -> list[tuple[str, str]]:
    if len(paths) > MAX_IMAGES:
        raise ModelError(f"at most {MAX_IMAGES} images per request")
    return [load_image(p) for p in paths]


# ---- HTTP ------------------------------------------------------------------

Transport = Callable[[str, dict, dict, float], dict]   # (url, headers, json_body, timeout) -> json


def urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (https URLs we build)
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read(600).decode("utf-8", "replace")
        raise ModelError(f"HTTP {e.code} from {url.split('?')[0]}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ModelError(f"request to {url.split('?')[0]} failed: {type(e).__name__}: {e}") from None


# ---- JSON schema (small subset; enough for our result schemas) --------------

def check_schema(obj: Any, schema: dict, path: str = "$") -> None:
    """Raise ValueError if obj does not satisfy schema (type, enum, required,
    properties, additionalProperties, items, min/max, length)."""
    t = schema.get("type")
    types = {"object": dict, "array": (list, tuple), "string": str, "boolean": bool,
             "null": type(None), "number": (int, float), "integer": int}
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        ok = False
        for x in ts:
            py = types[x]
            if isinstance(obj, bool) and x in ("number", "integer"):
                continue
            if isinstance(obj, py):
                ok = True
        if not ok:
            raise ValueError(f"{path}: expected {t}")
    if "enum" in schema and obj not in schema["enum"]:
        raise ValueError(f"{path}: not one of {schema['enum']}")
    if isinstance(obj, str):
        if len(obj) > schema.get("maxLength", 10**9) or len(obj) < schema.get("minLength", 0):
            raise ValueError(f"{path}: bad length")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if obj < schema.get("minimum", float("-inf")) or obj > schema.get("maximum", float("inf")):
            raise ValueError(f"{path}: out of range")
    if isinstance(obj, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", ()):
            if k not in obj:
                raise ValueError(f"{path}: missing {k}")
        for k, v in obj.items():
            if k in props:
                check_schema(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path}: unexpected key {k}")
    if isinstance(obj, (list, tuple)):
        if len(obj) > schema.get("maxItems", 10**9) or len(obj) < schema.get("minItems", 0):
            raise ValueError(f"{path}: bad item count")
        if "items" in schema:
            for i, v in enumerate(obj):
                check_schema(v, schema["items"], f"{path}[{i}]")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json_text(text: str) -> Optional[dict]:
    """Parse a model's JSON reply (tolerates a code fence). None if not an object."""
    try:
        v = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None
