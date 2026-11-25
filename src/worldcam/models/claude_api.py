"""`claude-api` backend (optional): Anthropic Messages API directly via urllib, one-shot only.

Needs ANTHROPIC_API_KEY and a full model id (e.g. `claude-api:claude-haiku-5-5`; no aliases).
Images as base64 blocks; JSON output via a forced `respond` tool whose input_schema is the schema.
UNVERIFIED live (no API key here); shapes follow the public Messages API.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from .api import GenRequest, GenResponse, NotConfigured
from .util import ModelError, Transport, check_schema, get_secret, load_images, redact, urllib_transport

URL = "https://api.anthropic.com/v1/messages"
KEY_VAR = "ANTHROPIC_API_KEY"


def require_key(env: Optional[dict] = None) -> str:
    key = get_secret(KEY_VAR, env)
    if not key:
        raise NotConfigured(f"claude-api backend needs {KEY_VAR} (use the `claude` backend for the CLI login)")
    return key


class ClaudeApiClient:
    def __init__(self, model: str, api_key: str, *, transport: Transport = urllib_transport,
                 timeout_s: float = 60.0):
        if not model or model == "default":
            raise NotConfigured("claude-api needs an explicit model id, e.g. claude-api:claude-haiku-5-5")
        self.model, self._key, self.transport, self.timeout_s = model, api_key, transport, timeout_s
        self.name = f"claude-api:{model}"

    def build_payload(self, req: GenRequest) -> dict:
        content = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": b}}
                   for m, b in load_images(req.images)] + [{"type": "text", "text": req.prompt}]
        body = {"model": self.model, "max_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": content}]}
        if req.system:
            body["system"] = req.system
        if req.json_schema is not None:
            body["tools"] = [{"name": "respond", "description": "Return the result.",
                              "input_schema": req.json_schema}]
            body["tool_choice"] = {"type": "tool", "name": "respond"}
        return body

    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.perf_counter()
        try:
            resp = self.transport(URL, {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
                                  self.build_payload(req), self.timeout_s)
        except ModelError as e:
            raise ModelError(redact(str(e), self._key)) from None
        blocks = resp.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        data = None
        if req.json_schema is not None:
            data = next((b.get("input") for b in blocks if b.get("type") == "tool_use"), None)
            if not isinstance(data, dict):
                raise ModelError("claude-api reply has no tool_use result")
            try:
                check_schema(data, req.json_schema)
            except ValueError as e:
                raise ModelError(f"claude-api reply violates schema: {e}") from None
            text = text or json.dumps(data)
        u = resp.get("usage") or {}
        return GenResponse(text, data=data, model=self.name,
                           usage={"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)},
                           latency_ms=(time.perf_counter() - t0) * 1000)
