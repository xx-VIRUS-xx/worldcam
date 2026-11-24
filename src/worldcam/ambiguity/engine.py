"""AmbiguityEngine: drives doubts through open -> asked -> answered -> resolved.

Stateless between calls: every decision is derived from the GraphStore, so the
process can be killed and restarted at any time. Bookkeeping lives in
Doubt.detail:
  outbox        question queued by the worker's ask_owner, not yet sent
  sent_ts       [ts, ...] when questions for this doubt went out (rate limit)
  attempts      consecutive failed worker runs; retry_after: backoff deadline
  answered_at, dismissed_at, dismissed_reason, ignored_by_owner

Call tick(now) periodically (e.g. every 30 s and right after handle_reply).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from worldcam.ambiguity.tools import ToolInputError, confine_image
from worldcam.ambiguity.worker import WorkerResult, WorkerRunner
from worldcam.graph.api import Doubt, GraphStore
from worldcam.messenger.base import Choice, Notifier, OutgoingQuestion, OwnerReply

log = logging.getLogger(__name__)
ALL = 10_000


@dataclass
class EngineConfig:
    max_questions_per_hour: int = 6
    quiet_start_hour: int = 23          # local time, inclusive
    quiet_end_hour: int = 8             # local time, exclusive
    max_rounds: int = 3
    expiry_days: float = 7.0
    max_attempts: int = 3               # failed worker runs before a doubt is dismissed
    retry_backoff_s: float = 300.0
    max_runs_per_tick: int = 3
    snapshot_dir: str = "captures/snapshots"
    ignore_values: tuple[str, ...] = ("ignore",)
    max_answer_chars: int = 1000


@dataclass
class TickReport:
    expired: list[str] = field(default_factory=list)
    asked_runs: list[str] = field(default_factory=list)
    resolve_runs: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    quiet: bool = False


class AmbiguityEngine:
    def __init__(self, store: GraphStore, notifier: Notifier, worker_runner: WorkerRunner,
                 config: Optional[EngineConfig] = None):
        self.store = store
        self.notifier = notifier
        self.worker = worker_runner
        self.cfg = config or EngineConfig()

    # -- policy helpers -------------------------------------------------------
    def is_quiet(self, now: float) -> bool:
        h = time.localtime(now).tm_hour
        s, e = self.cfg.quiet_start_hour, self.cfg.quiet_end_hour
        return (h >= s or h < e) if s > e else (s <= h < e)

    def sent_last_hour(self, now: float) -> int:
        return sum(1 for d in self.store.list_doubts(None, ALL)
                   for t in d.detail.get("sent_ts", ()) if now - t < 3600)

    def _pending_outbox(self) -> list[Doubt]:
        return [d for d in self.store.list_doubts("asked", ALL) if d.detail.get("outbox")]

    def _budget(self, now: float) -> int:
        return self.cfg.max_questions_per_hour - self.sent_last_hour(now) - len(self._pending_outbox())

    # -- main loop --------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> TickReport:
        now = time.time() if now is None else now
        rep = TickReport(quiet=self.is_quiet(now))
        self._expire(now, rep)
        runs = 0
        # Resolve runs don't message the owner by themselves (a follow-up goes to the
        # outbox and is rate-limited at send time), so they also run in quiet hours.
        for d in self.store.list_doubts("answered", ALL):
            if runs >= self.cfg.max_runs_per_tick:
                break
            if self._ready(d, now):
                runs += 1
                rep.resolve_runs.append(d.id)
                self._run("resolve", d, now, rep)
        self._flush(now, rep)
        if not rep.quiet:
            for d in self.store.list_doubts("open", ALL):
                if runs >= self.cfg.max_runs_per_tick or self._budget(now) <= 0:
                    break
                if self._ready(d, now):
                    runs += 1
                    rep.asked_runs.append(d.id)
                    self._run("ask", d, now, rep)
            self._flush(now, rep)
        return rep

    def _ready(self, d: Doubt, now: float) -> bool:
        return now >= d.detail.get("retry_after", 0)

    def _expire(self, now: float, rep: TickReport) -> None:
        limit = self.cfg.expiry_days * 86400
        for st in ("open", "asked"):
            for d in self.store.list_doubts(st, ALL):
                if now - d.created > limit:
                    self.store.update_doubt(d.id, status="dismissed", detail=dict(
                        _without(d.detail, "outbox"), dismissed_reason="expired", dismissed_at=now))
                    rep.expired.append(d.id)

    def _flush(self, now: float, rep: TickReport) -> None:
        """Deliver questions queued by workers (status asked, message_ref None)."""
        if rep.quiet:
            return
        sent = self.sent_last_hour(now)
        for d in sorted(self._pending_outbox(), key=lambda d: d.detail["outbox"].get("queued_at", 0)):
            if sent >= self.cfg.max_questions_per_hour:
                break
            ob = d.detail["outbox"]
            images = []
            for p in ob.get("image_paths", ())[:2]:
                try:
                    images.append(confine_image(p, self.cfg.snapshot_dir))  # re-check in this process
                except ToolInputError as e:
                    log.warning("dropping image for %s: %s", d.id, e)
            choices = [Choice(c["label"], c["value"]) for c in ob.get("choices", ())]
            if not any(c.value in self.cfg.ignore_values for c in choices):
                choices.append(Choice("Ignore", self.cfg.ignore_values[0]))
            try:
                ref = self.notifier.ask(OutgoingQuestion(d.id, ob["text"], tuple(images), tuple(choices)))
            except Exception as e:  # keep it queued; try again next tick
                log.warning("notifier failed for %s: %s", d.id, e)
                self.store.update_doubt(d.id, last_error=f"send failed: {e}"[:300])
                break
            self.store.update_doubt(d.id, message_ref=str(ref), detail=dict(
                _without(d.detail, "outbox"), sent_ts=list(d.detail.get("sent_ts", ())) + [now]))
            sent += 1
            rep.sent.append(d.id)

    def _run(self, mode: str, d: Doubt, now: float, rep: TickReport) -> None:
        prev = d.status
        try:
            res = self.worker.run(mode, d)
        except Exception as e:
            res = WorkerResult(False, error=f"runner crashed: {type(e).__name__}: {e}")
        after = self.store.get_doubt(d.id)
        progressed = after is not None and (
            after.status in ("resolved", "dismissed")
            or (after.status == "asked" and after.detail.get("outbox")))
        if res.ok and progressed:
            self.store.update_doubt(d.id, last_error=None,
                                    detail=_without(after.detail, "attempts", "retry_after"))
            return
        rep.failed.append(d.id)
        err = (res.error or f"worker returned {res.action!r} but doubt is {after.status if after else 'gone'}")[:300]
        if after is None:
            return
        attempts = after.detail.get("attempts", 0) + 1
        if progressed:
            # The side effect (queued question / resolution) happened; keep it, record the error.
            self.store.update_doubt(d.id, last_error=err)
            return
        if attempts >= self.cfg.max_attempts:
            self.store.update_doubt(d.id, status="dismissed", last_error=err, detail=dict(
                after.detail, attempts=attempts, dismissed_reason="worker failed", dismissed_at=now))
            return
        self.store.update_doubt(d.id, status=prev, last_error=err, detail=dict(
            after.detail, attempts=attempts, retry_after=now + self.cfg.retry_backoff_s * attempts))

    # -- owner replies ----------------------------------------------------------
    def handle_reply(self, reply: OwnerReply, now: Optional[float] = None) -> Optional[str]:
        """Store the owner's reply on the matching doubt. Returns the doubt id or None.
        The resolve run happens on the next tick()."""
        now = time.time() if now is None else now
        d = self.store.find_doubt_by_message(str(reply.message_ref))
        if d is None or d.status not in ("asked", "answered"):
            return None
        if reply.choice_value and reply.choice_value.strip().lower() in self.cfg.ignore_values:
            self.store.update_doubt(d.id, status="dismissed", answer=reply.choice_value, detail=dict(
                d.detail, ignored_by_owner=True, dismissed_reason="owner ignored", dismissed_at=now))
            return d.id
        text = (reply.text or reply.choice_value or "").strip()
        if not text:
            return None
        if d.status == "answered" and d.answer:  # owner sent a second message before the resolve run
            text = f"{d.answer}\n{text}"
        self.store.update_doubt(d.id, status="answered", answer=text[: self.cfg.max_answer_chars],
                                detail=dict(d.detail, answered_at=now))
        return d.id

    def poll_once(self, timeout_s: float = 25.0, now: Optional[float] = None) -> TickReport:
        """Convenience loop body: read replies, then tick. A poller error (e.g.
        TelegramError on a 409 from a second poller) is logged and the tick still runs."""
        try:
            replies = self.notifier.poll_replies(timeout_s)
        except Exception as e:
            log.warning("poll_replies failed: %s", e)
            replies = ()
        for r in replies:
            self.handle_reply(r, now)
        return self.tick(now)


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
