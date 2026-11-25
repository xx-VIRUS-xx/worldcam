"""`gemini` backend: REST generateContent (urllib, no SDK).

- key from GEMINI_API_KEY (env, then .env), sent in the `x-goog-api-key` header (never in the
  URL, never logged); errors are redacted.
- images as inlineData, JSON output via responseMimeType + responseJsonSchema,
- tool loops via functionDeclarations (parametersJsonSchema). Gemini cannot combine function
  calling with JSON-mode output, so a required output schema becomes one extra function,
  `final_answer`, whose arguments are the result.
UNVERIFIED against the live API on this machine (no key); payload shapes follow the public docs.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from .api import AgentRunRequest, AgentRunResult, GenRequest, GenResponse, NotConfigured, ToolSpec
from .util import (ModelError, Transport, check_schema, get_secret, load_images, parse_json_text, redact,
                   urllib_transport)

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
KEY_VAR = "GEMINI_API_KEY"
FINAL_TOOL = "final_answer"


def require_key(env: Optional[dict] = None) -> str:
    key = get_secret(KEY_VAR, env)
    if not key:
        raise NotConfigured(f"gemini backend needs {KEY_VAR} (set it in the environment or in .env)")
    return key


def _image_parts(paths) -> list[dict]:
    return [{"inlineData": {"mimeType": m, "data": b}} for m, b in load_images(paths)]


def _usage(resp: dict) -> dict:
    u = resp.get("usageMetadata") or {}
    return {"input_tokens": u.get("promptTokenCount", 0),
            "output_tokens": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)}


def _parts(resp: dict) -> list[dict]:
    cands = resp.get("candidates") or []
    if not cands:
        fb = resp.get("promptFeedback") or {}
        raise ModelError(f"gemini returned no candidates (blocked: {fb.get('blockReason', 'unknown')})")
    return (cands[0].get("content") or {}).get("parts") or []


def _text(parts: list[dict]) -> str:
    return "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))


class _GeminiBase:
    def __init__(self, model: str, api_key: str, *, transport: Transport = urllib_transport,
                 timeout_s: float = 60.0):
        self.model = model
        self._key = api_key
        self.transport = transport
        self.timeout_s = timeout_s
        self.name = f"gemini:{model}"

    def _call(self, body: dict, timeout: Optional[float] = None) -> dict:
        try:
            return self.transport(f"{BASE}/{self.model}:generateContent", {"x-goog-api-key": self._key},
                                  body, timeout or self.timeout_s)
        except ModelError as e:
            raise ModelError(redact(str(e), self._key)) from None


class GeminiClient(_GeminiBase):
    def build_payload(self, req: GenRequest) -> dict:
        gen: dict[str, Any] = {"maxOutputTokens": req.max_tokens}
        if req.json_schema is not None:
            gen["responseMimeType"] = "application/json"
            gen["responseJsonSchema"] = req.json_schema
        body: dict[str, Any] = {"contents": [{"role": "user",
                                              "parts": _image_parts(req.images) + [{"text": req.prompt}]}],
                                "generationConfig": gen}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        return body

    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.perf_counter()
        resp = self._call(self.build_payload(req))
        text = _text(_parts(resp))
        data = None
        if req.json_schema is not None:
            data = parse_json_text(text)
            if data is None:
                raise ModelError("gemini reply is not a JSON object")
            try:
                check_schema(data, req.json_schema)
            except ValueError as e:
                raise ModelError(f"gemini reply violates schema: {e}") from None
        return GenResponse(text, data=data, model=self.name, usage=_usage(resp),
                           latency_ms=(time.perf_counter() - t0) * 1000)


class GeminiToolRunner(_GeminiBase):
    @staticmethod
    def declarations(specs, output_schema: Optional[dict]) -> list[dict]:
        decl = [{"name": s.name, "description": s.description, "parametersJsonSchema": s.input_schema}
                for s in specs]
        if output_schema is not None:
            decl.append({"name": FINAL_TOOL,
                         "description": "Call this exactly once when done, with the final result as arguments.",
                         "parametersJsonSchema": output_schema})
        return decl

    def build_payload(self, req: AgentRunRequest, contents: list) -> dict:
        body: dict[str, Any] = {"contents": contents}
        decl = self.declarations(req.toolset.specs, req.output_schema)
        if decl:
            body["tools"] = [{"functionDeclarations": decl}]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        return body

    def run(self, req: AgentRunRequest) -> AgentRunResult:
        t0 = time.perf_counter()
        deadline = t0 + req.timeout_s
        specs = {s.name: s for s in req.toolset.specs}
        contents: list[dict] = [{"role": "user", "parts": _image_parts(req.images) + [{"text": req.prompt}]}]
        calls, tin, tout = 0, 0, 0

        def done(ok, **kw):
            return AgentRunResult(ok, model=self.name, tool_calls=calls,
                                  usage={"input_tokens": tin, "output_tokens": tout},
                                  latency_ms=(time.perf_counter() - t0) * 1000, **kw)
        try:
            for _ in range(req.max_turns):
                left = deadline - time.perf_counter()
                if left <= 0:
                    return done(False, error=f"timeout after {req.timeout_s:.0f}s")
                resp = self._call(self.build_payload(req, contents), timeout=min(self.timeout_s, left))
                u = _usage(resp)
                tin, tout = tin + u["input_tokens"], tout + u["output_tokens"]
                parts = _parts(resp)
                fcs = [p["functionCall"] for p in parts if "functionCall" in p]
                if not fcs:
                    text = _text(parts)
                    if req.output_schema is None:
                        return done(True, text=text)
                    contents += [{"role": "model", "parts": parts or [{"text": ""}]},
                                 {"role": "user", "parts": [{"text": f"Finish by calling {FINAL_TOOL}."}]}]
                    continue
                contents.append({"role": "model", "parts": parts})   # echoes thought signatures
                replies = []
                for fc in fcs:
                    name, args = fc.get("name"), fc.get("args") or {}
                    if name == FINAL_TOOL and req.output_schema is not None:
                        try:
                            check_schema(args, req.output_schema)
                        except ValueError as e:
                            replies.append({"functionResponse": {"name": name,
                                            "response": {"error": f"schema violation: {e}; call again"}}})
                            continue
                        return done(True, data=args, text=json.dumps(args))
                    calls += 1
                    spec: Optional[ToolSpec] = specs.get(name)
                    if spec is None:
                        out: Any = {"error": f"unknown tool {name!r}"}
                    else:
                        try:
                            out = spec.fn(args)
                        except (ValueError, KeyError, PermissionError) as e:
                            out = {"error": str(e)}
                    replies.append({"functionResponse": {"name": name,
                                    "response": out if isinstance(out, dict) else {"result": out}}})
                contents.append({"role": "user", "parts": replies})
            return done(False, error=f"no final answer within {req.max_turns} turns")
        except ModelError as e:
            return done(False, error=str(e))
