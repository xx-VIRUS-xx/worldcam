"""Bounded usage log: what each task costs. Never stores prompts, images or outputs."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Optional

# USD per million tokens (input, output). Estimates only; unknown models -> cost None.
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
}


def estimate_cost(model: str, usage: dict) -> Optional[float]:
    if usage.get("cost_usd") is not None:
        return float(usage["cost_usd"])
    name = model.split(":", 1)[-1]
    p = PRICES.get(name)
    if not p or "input_tokens" not in usage:
        return None
    return (usage.get("input_tokens", 0) * p[0] + usage.get("output_tokens", 0) * p[1]) / 1e6


class UsageLog:
    def __init__(self, maxlen: int = 500, path: Optional[str] = None):
        self.records: deque = deque(maxlen=maxlen)
        self.path = path
        self._lock = threading.Lock()

    def record(self, task: str, model: str, usage: Optional[dict] = None, latency_ms: float = 0.0,
               ok: bool = True) -> dict:
        usage = usage or {}
        rec = {"ts": round(time.time(), 1), "task": task, "model": model, "ok": ok,
               "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
               "latency_ms": round(latency_ms, 1), "cost_usd": estimate_cost(model, usage)}
        with self._lock:
            self.records.append(rec)
            if self.path:
                try:
                    with open(self.path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except OSError:
                    pass                      # logging must never break a run
        return rec

    def summary(self) -> dict:
        """task -> {calls, errors, input_tokens, output_tokens, cost_usd, avg_latency_ms}."""
        out: dict = {}
        with self._lock:
            recs = list(self.records)
        for r in recs:
            s = out.setdefault(r["task"], {"calls": 0, "errors": 0, "input_tokens": 0,
                                           "output_tokens": 0, "cost_usd": 0.0, "_lat": 0.0})
            s["calls"] += 1
            s["errors"] += 0 if r["ok"] else 1
            s["input_tokens"] += r["input_tokens"] or 0
            s["output_tokens"] += r["output_tokens"] or 0
            s["cost_usd"] += r["cost_usd"] or 0.0
            s["_lat"] += r["latency_ms"]
        for s in out.values():
            s["avg_latency_ms"] = round(s.pop("_lat") / s["calls"], 1)
            s["cost_usd"] = round(s["cost_usd"], 6)
        return out
