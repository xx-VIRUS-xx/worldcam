"""`jev` backend (TypeSafe AI Jev, typed decisions) for the `decide` task. One-shot only.

API shape: the only REST documentation found (2026-11-25) is AI/ML API's
https://docs.aimlapi.com/api-references/decision-models/typesafe/jev :
    POST https://api.aimlapi.com/v1/decisions   Authorization: Bearer <key>
    {"model": "typesafe/jev", "state": "<text>", "questions": {name: {"type": "noul"|"choice"|"score"|"box",
        "instructions": "...", "criteria": ...}}}   ->   {"model", "answers": {name: {...}}, "usage": {...}}
    noul   -> {"type":"noul","noul":<P(yes)>}
    choice -> {"choice","confidence","probabilities"}     (criteria = {option_key: description})
    score  -> {"score","confidence","legend","probabilities"}   (criteria = [levels, lowest first])
TypeSafe's own docs (docs.typesafe.ai) and endpoint (sources disagree: api.typesafe.ai/v1/systemone) could not
be fetched; set JEV_BASE_URL to use another endpoint. NOT verified live (no key, early access).

Mapping from GenRequest (our adapter convention, not TypeSafe's): `prompt` = the state; `json_schema`
= {"type":"object","properties":{name: {"x-jev": {"type": "noul", "instructions": "...", "criteria": ...}}}}.
`data[name]` = P(yes) for noul, the chosen key for choice, the score for score; `text` = JSON of the raw answers.
Key: JEV_API_KEY (or AIMLAPI_KEY when using the AI/ML API endpoint).
"""
from __future__ import annotations

import json
import time
from typing import Optional

from .api import GenRequest, GenResponse, NotConfigured
from .util import ModelError, Transport, get_secret, redact, urllib_transport

DEFAULT_URL = "https://api.aimlapi.com/v1/decisions"
DEFAULT_MODEL = "typesafe/jev"
KEY_VARS = ("JEV_API_KEY", "AIMLAPI_KEY")


def find_key(env: Optional[dict] = None) -> Optional[str]:
    for v in KEY_VARS:
        k = get_secret(v, env)
        if k:
            return k
    return None


def require_key(env: Optional[dict] = None) -> str:
    key = find_key(env)
    if not key:
        raise NotConfigured("jev backend needs JEV_API_KEY (or AIMLAPI_KEY for the AI/ML API endpoint); "
                            "Jev is early access and its REST API is documented only via aimlapi.com")
    return key


class JevClient:
    def __init__(self, model: str, api_key: str, *, url: str = DEFAULT_URL,
                 transport: Transport = urllib_transport, timeout_s: float = 30.0):
        self.model = DEFAULT_MODEL if model in ("", "default") else model
        self._key, self.url, self.transport, self.timeout_s = api_key, url, transport, timeout_s
        self.name = f"jev:{self.model}"

    def build_payload(self, req: GenRequest) -> dict:
        if not req.json_schema or "properties" not in req.json_schema:
            raise ModelError("jev needs json_schema.properties.<name>['x-jev'] question definitions")
        if req.images:
            raise ModelError("jev adapter does not send images")
        qs = {}
        for name, p in req.json_schema["properties"].items():
            q = p.get("x-jev")
            if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
                raise ModelError(f"question {name!r} lacks an x-jev {{type, instructions}} definition")
            qs[name] = q
        return {"model": self.model, "state": req.prompt if not req.system else f"{req.system}\n\n{req.prompt}",
                "questions": qs}

    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.perf_counter()
        body = self.build_payload(req)
        try:
            resp = self.transport(self.url, {"Authorization": f"Bearer {self._key}"}, body, self.timeout_s)
        except ModelError as e:
            raise ModelError(redact(str(e), self._key)) from None
        answers = resp.get("answers")
        if not isinstance(answers, dict):
            raise ModelError("jev reply has no answers object")
        data = {}
        for name, a in answers.items():
            kind = a.get("type") or body["questions"].get(name, {}).get("type")
            data[name] = a.get("noul") if kind == "noul" else a.get("choice", a.get("score", a))
        u = resp.get("usage") or {}
        return GenResponse(json.dumps(answers), data=data, model=self.name,
                           usage={"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)},
                           latency_ms=(time.perf_counter() - t0) * 1000)
