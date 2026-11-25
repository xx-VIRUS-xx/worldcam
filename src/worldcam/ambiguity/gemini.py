"""Gemini worker runner: executes ambiguity engine steps via Gemini API + WorldcamTools.

Uses the Gemini `generateContent` REST endpoint over stdlib `urllib` (zero extra
dependencies), with:
  - API key via `x-goog-api-key` header (read from `GEMINI_API_KEY` env or `.env`,
    held in `_Secret` and redacted from every error/log).
  - Native Gemini function calling (`functionDeclarations`) backed directly by
    `WorldcamTools`, enforcing the exact same per-doubt scope, read-only `graph_read`,
    image path confinement, and no-delete security rules as the MCP server.
  - Multimodal vision input: confined snapshot crops attached as `inlineData` so
    Gemini can inspect people, animals, and objects to extract visual metadata and
    word specific questions.
  - Automatic structured-output finalization (`responseMimeType: "application/json"`,
    `responseJsonSchema`) if the model ends its tool loop with prose instead of JSON.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from worldcam.ambiguity.file_store import open_store
from worldcam.ambiguity.tools import TOOL_NAMES, ToolInputError, WorldcamTools, confine_image
from worldcam.ambiguity.worker import MODES, WorkerResult, load_schema, validate_result
from worldcam.graph.api import Doubt, ENTITY_TYPES, FACT_SOURCES, GraphStore
from worldcam.graph.config import load_env

log = logging.getLogger("worldcam.ambiguity.gemini")

PROMPTS = Path(__file__).with_name("prompts")
DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


class GeminiError(RuntimeError):
    """Gemini API call failed. Never contains the API key."""


class _Secret:
    __slots__ = ("_v",)

    def __init__(self, v: str):
        self._v = v

    def get(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__


class GeminiTransport(Protocol):
    def post(self, url: str, headers: dict[str, str], body: bytes,
             timeout: float) -> tuple[int, bytes]:
        """POST JSON and return (http_status, response_bytes)."""


class UrllibGeminiTransport:
    def __init__(self):
        self._ctx = ssl.create_default_context()

    def post(self, url: str, headers: dict[str, str], body: bytes,
             timeout: float) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            try:
                data = e.read()
            except Exception:
                data = b""
            return e.code, data
        except urllib.error.URLError as e:
            raise ConnectionError(f"URLError: {type(e.reason).__name__}") from None


def get_gemini_setting(name: str, env_path: Optional[Path] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    return load_env(env_path).get(name) or None


def redact_key(text: str, api_key: Optional[str] = None) -> str:
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "<redacted>", text)


def _extract_json_object(text: str) -> dict:
    s = (text or "").strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start:end + 1])
        raise


def gemini_function_declarations() -> list[dict[str, Any]]:
    """Gemini `functionDeclarations` matching the 13 WorldcamTools methods."""
    decls = [
        {
            "name": "get_doubt",
            "description": "Read the doubt this run is about: kind, status, question, owner answer, history, about_ids, sightings.",
            "parameters": {
                "type": "object",
                "properties": {"doubt_id": {"type": "string"}},
                "required": ["doubt_id"],
            },
        },
        {
            "name": "person_context",
            "description": "Read an entity (person, animal, or object), its facts, relations, recent sightings, and clusters.",
            "parameters": {
                "type": "object",
                "properties": {"person_id": {"type": "string"}},
                "required": ["person_id"],
            },
        },
        {
            "name": "search_people",
            "description": "Find known entities (people, animals, objects) by name or alias (case-insensitive substring).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "graph_read",
            "description": "Run a parameterised read-only Cypher query (Neo4j only; returns an error on the test file store).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string"},
                    "params": {"type": "object"},
                    "limit": {"type": "integer"},
                },
                "required": ["cypher"],
            },
        },
        {
            "name": "upsert_person",
            "description": (
                "Create an entity (name required) or update one in this doubt's scope. "
                "Supports people, animals, objects, and other entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "person_id": {"type": "string"},
                    "entity_type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "label": {"type": "string"},
                },
            },
        },
        {
            "name": "label_face_cluster",
            "description": "Attach a face cluster from this doubt to a person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "person_id": {"type": "string"},
                },
                "required": ["cluster_id", "person_id"],
            },
        },
        {
            "name": "add_fact",
            "description": (
                "Add a metadata fact to an entity in scope. key is snake_case (e.g. relation_to_owner, "
                "species_or_breed, color, appearance, brand_or_model, owner_or_user, purpose, usual_zone, "
                "visit_pattern, works_at, role, note). source: owner|observed|inferred. confidence: 0..1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "source": {"type": "string", "enum": list(FACT_SOURCES)},
                    "confidence": {"type": "number"},
                },
                "required": ["person_id", "key", "value", "source", "confidence"],
            },
        },
        {
            "name": "relate",
            "description": "Add a relation edge a -[relation]-> b (e.g. colleague, partner, friend, pet_of, belongs_to, used_by).",
            "parameters": {
                "type": "object",
                "properties": {
                    "a_id": {"type": "string"},
                    "relation": {"type": "string"},
                    "b_id": {"type": "string"},
                    "source": {"type": "string", "enum": list(FACT_SOURCES)},
                },
                "required": ["a_id", "relation", "b_id", "source"],
            },
        },
        {
            "name": "set_persona_summary",
            "description": "Set or refresh the concise persona summary for a person, animal, or object in scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["person_id", "summary"],
            },
        },
        {
            "name": "merge_people",
            "description": "Merge drop_id into keep_id (only allowed for a conflict doubt about both).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_id": {"type": "string"},
                    "drop_id": {"type": "string"},
                },
                "required": ["keep_id", "drop_id"],
            },
        },
        {
            "name": "ask_owner",
            "description": "Queue one question to the owner (engine sends it via Telegram). Max 3 rounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doubt_id": {"type": "string"},
                    "text": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional quick-reply button labels (e.g. ['New entity', 'Ignore']).",
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["doubt_id", "text"],
            },
        },
        {
            "name": "resolve_doubt",
            "description": "Mark an answered doubt resolved after writing all collected metadata, relations, and persona.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doubt_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["doubt_id", "summary"],
            },
        },
        {
            "name": "dismiss_doubt",
            "description": "Dismiss the doubt when nothing useful can be asked or learned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doubt_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["doubt_id", "reason"],
            },
        },
    ]
    assert tuple(d["name"] for d in decls) == TOOL_NAMES
    return decls


class GeminiClient:
    """Minimal REST client for Gemini `generateContent` with retry and key redaction."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        base_url: str = DEFAULT_GEMINI_BASE,
        transport: Optional[GeminiTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        request_timeout: float = 45.0,
    ):
        key = (api_key or "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY must be non-empty")
        self._key = _Secret(key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._transport = transport or UrllibGeminiTransport()
        self._sleep = sleep
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r}, api_key=<secret>, base_url={self.base_url!r})"

    def _redact(self, s: str) -> str:
        return redact_key(s, self._key.get())

    def generate_content(self, payload: dict[str, Any], *, timeout: Optional[float] = None) -> dict[str, Any]:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._key.get(),
            "User-Agent": "worldcam",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_timeout = timeout or self.request_timeout
        attempt = 0
        while True:
            err: Optional[str] = None
            try:
                status, raw = self._transport.post(url, headers, body, req_timeout)
            except OSError as e:
                err = f"{type(e).__name__}: {self._redact(str(e))}"
                status, raw = 0, b""
            if err is None:
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    data = None
                if 200 <= status < 300 and isinstance(data, dict) and "candidates" in data:
                    return data
                msg = ""
                if isinstance(data, dict) and isinstance(data.get("error"), dict):
                    msg = str(data["error"].get("message") or data["error"].get("status") or "")
                msg = self._redact(msg or f"HTTP {status}")
                if status in (429, 500, 502, 503, 504):
                    err = msg
                else:
                    raise GeminiError(f"Gemini generateContent failed [{status}]: {msg}")
            if attempt >= self.max_retries:
                raise GeminiError(f"Gemini generateContent failed after {attempt + 1} attempts: {err}")
            delay = min(20.0, 1.5 * (2 ** attempt)) + random.uniform(0, 0.25)
            log.warning("gemini call failed (%s); retry %d in %.1fs", err, attempt + 1, delay)
            self._sleep(delay)
            attempt += 1


class GeminiWorkerRunner:
    """Runs one stateless Gemini function-calling loop per doubt step."""

    def __init__(
        self,
        store_spec: Optional[str] = None,
        snapshot_dir: str = "captures/snapshots",
        *,
        store: Optional[GraphStore] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 120.0,
        max_turns: int = 12,
        max_rounds: int = 3,
        include_images: bool = True,
        client: Optional[GeminiClient] = None,
        transport: Optional[GeminiTransport] = None,
        env_path: Optional[Path] = None,
    ):
        if store is None:
            if not store_spec or (store_spec != "neo4j" and not store_spec.startswith("file:")):
                raise ValueError("store_spec must be 'neo4j' or 'file:<path>' (or pass store=...)")
        self.store_spec = store_spec
        self._store = store
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        self.model = model or get_gemini_setting("WORLDCAM_GEMINI_MODEL", env_path) or DEFAULT_GEMINI_MODEL
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.include_images = include_images
        self._env_path = env_path
        self._explicit_api_key = api_key
        self._transport = transport
        self._client = client
        self.last_response: Optional[dict[str, Any]] = None

    def __repr__(self) -> str:
        return (f"GeminiWorkerRunner(model={self.model!r}, store_spec={self.store_spec!r}, "
                f"snapshot_dir={self.snapshot_dir!r})")

    def _get_client(self) -> GeminiClient:
        if self._client is not None:
            return self._client
        key = self._explicit_api_key or get_gemini_setting("GEMINI_API_KEY", self._env_path)
        if not key:
            raise GeminiError("GEMINI_API_KEY is not set (environment or .env)")
        self._client = GeminiClient(key, model=self.model, transport=self._transport)
        return self._client

    def prompt(self, mode: str, doubt_id: str) -> str:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return (PROMPTS / f"{mode}.md").read_text(encoding="utf-8").format(doubt_id=doubt_id)

    def system_instruction(self) -> str:
        return (PROMPTS / "system.md").read_text(encoding="utf-8")

    def _image_parts(self, doubt: Doubt) -> list[dict[str, Any]]:
        if not self.include_images:
            return []
        parts: list[dict[str, Any]] = []
        for p in doubt.snapshot_paths[:2]:
            try:
                real = confine_image(p, self.snapshot_dir)
                ext = os.path.splitext(real)[1].lower()
                mime = _MIME_BY_EXT.get(ext)
                if not mime:
                    continue
                with open(real, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                parts.append({"inlineData": {"mimeType": mime, "data": b64}})
            except (ToolInputError, OSError) as e:
                log.debug("skipping snapshot image %s for %s: %s", p, doubt.id, e)
        return parts

    def _dispatch_tool(self, tools: WorldcamTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        fn = getattr(tools, name)
        try:
            res = fn(**(args or {}))
            return {"result": res}
        except (ToolInputError, TypeError, ValueError, KeyError) as e:
            return {"error": str(e)}

    def _open_store(self) -> tuple[GraphStore, bool]:
        if self._store is not None:
            return self._store, False
        assert self.store_spec is not None
        return open_store(self.store_spec), True

    def run(self, mode: str, doubt: Doubt) -> WorkerResult:
        t0 = time.monotonic()
        deadline = t0 + self.timeout_s
        try:
            client = self._get_client()
            task_prompt = self.prompt(mode, doubt.id)
        except Exception as e:
            return WorkerResult(False, error=str(e), wall_s=time.monotonic() - t0)

        store, owns_store = self._open_store()
        try:
            tools = WorldcamTools(store, self.snapshot_dir, doubt.id, max_rounds=self.max_rounds)
            user_parts: list[dict[str, Any]] = [{"text": task_prompt}] + self._image_parts(doubt)
            contents: list[dict[str, Any]] = [{"role": "user", "parts": user_parts}]
            tool_spec = [{"functionDeclarations": gemini_function_declarations()}]
            sys_inst = {"parts": [{"text": self.system_instruction()}]}

            final_text = ""
            total_tokens = 0
            for _turn in range(self.max_turns):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return WorkerResult(False, error=f"timeout after {self.timeout_s:.0f}s",
                                        wall_s=time.monotonic() - t0)
                payload = {
                    "systemInstruction": sys_inst,
                    "contents": contents,
                    "tools": tool_spec,
                    "generationConfig": {"temperature": 0.2},
                }
                resp = client.generate_content(payload, timeout=min(rem, client.request_timeout))
                self.last_response = resp
                total_tokens += int((resp.get("usageMetadata") or {}).get("totalTokenCount") or 0)
                cand = (resp.get("candidates") or [{}])[0]
                model_content = cand.get("content") or {"role": "model", "parts": []}
                parts = model_content.get("parts") or []
                contents.append(model_content)

                fcalls = [p["functionCall"] for p in parts if isinstance(p, dict) and "functionCall" in p]
                if not fcalls:
                    final_text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
                    break

                fresps = []
                for fc in fcalls:
                    fname = fc.get("name", "")
                    fargs = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                    out = self._dispatch_tool(tools, fname, fargs)
                    fresps.append({"functionResponse": {"name": fname, "response": out}})
                contents.append({"role": "user", "parts": fresps})
            else:
                return WorkerResult(False, error=f"gemini exceeded max_turns={self.max_turns}",
                                    wall_s=time.monotonic() - t0)

            obj = self._parse_or_finalize(client, sys_inst, contents, final_text, deadline)
            wall = time.monotonic() - t0
            ok = obj["action"] != "error"
            return WorkerResult(
                ok=ok,
                action=obj["action"],
                summary=obj["summary"],
                error=None if ok else obj["summary"],
                wall_s=wall,
                data={**obj, "model": self.model, "total_tokens": total_tokens},
            )
        except Exception as e:
            return WorkerResult(False, error=redact_key(str(e)), wall_s=time.monotonic() - t0)
        finally:
            if owns_store and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass

    def _parse_or_finalize(
        self,
        client: GeminiClient,
        sys_inst: dict[str, Any],
        contents: list[dict[str, Any]],
        text: str,
        deadline: float,
    ) -> dict[str, Any]:
        if text.strip():
            try:
                return validate_result(_extract_json_object(text))
            except (ValueError, json.JSONDecodeError):
                pass
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"timeout after {self.timeout_s:.0f}s")
        # Separate structured-output turn (without `tools`, because Gemini rejects
        # responseMimeType='application/json' alongside function declarations on many models).
        finalize_contents = list(contents) + [{
            "role": "user",
            "parts": [{"text": "Return ONLY the final JSON object matching the required output schema."}],
        }]
        payload = {
            "systemInstruction": sys_inst,
            "contents": finalize_contents,
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseJsonSchema": load_schema(),
            },
        }
        resp = client.generate_content(payload, timeout=min(rem, client.request_timeout))
        self.last_response = resp
        cand = (resp.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        raw = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
        return validate_result(_extract_json_object(raw))
