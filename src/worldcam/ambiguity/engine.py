"""AmbiguityEngine: drives doubts through open -> asked -> answered -> resolved.

Stateless between calls: every decision is derived from the GraphStore, so the
process can be killed and restarted at any time. Bookkeeping lives in
Doubt.detail:
  outbox        question queued by the worker's ask_owner, not yet sent
  sent_ts       [ts, ...] when questions for this doubt went out (rate limit)
  attempts      consecutive failed worker runs; retry_after: backoff deadline
  answered_at, dismissed_at, dismissed_reason, ignored_by_owner

Call tick(now) periodically (e.g. every 30 s and right after handle_reply).

Two delivery modes behind the same surface:
  legacy  (no dispatcher): one owner via a Notifier, a fixed per-hour cap + quiet hours.
  multi   (dispatcher given): UncertaintyManager ranks doubts and picks WHO to ask by
          permission (owner is the fallback); the Dispatcher applies each user's
          AttentionBudget (quiet hours, caps, ignore back-off) and `view_images`.
          Replies come in through handle_user_reply(user, InboundMessage).
          `detail` additionally records asked_to / asked_role / asked_channel and
          confirmed_by on answers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from worldcam.ambiguity.tools import ToolInputError, confine_image
from worldcam.ambiguity.worker import WorkerResult, WorkerRunner
from worldcam.channels.api import InboundMessage, OutboundMessage
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
    unanswered_after_h: float = 12.0    # multi mode: no answer after this long counts as "ignored"


@dataclass
class TickReport:
    expired: list[str] = field(default_factory=list)
    asked_runs: list[str] = field(default_factory=list)
    resolve_runs: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    quiet: bool = False


class AmbiguityEngine:
    def __init__(self, store: GraphStore, notifier: Optional[Notifier], worker_runner: WorkerRunner,
                 config: Optional[EngineConfig] = None, *, dispatcher=None, uncertainty=None):
        self.store = store
        self.notifier = notifier
        self.worker = worker_runner
        self.cfg = config or EngineConfig()
        self.dispatcher = dispatcher
        self.multi = dispatcher is not None
        self.uncertainty = uncertainty
        if self.multi and uncertainty is None:
            from worldcam.agent.uncertainty import UncertaintyManager
            self.uncertainty = UncertaintyManager(store, dispatcher.users, budget=dispatcher.budget)

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
        rep = TickReport(quiet=False if self.multi else self.is_quiet(now))
        self._expire(now, rep)
        if self.multi:
            self._sweep_unanswered(now)
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
            for d in self._open_in_order(now):
                if runs >= self.cfg.max_runs_per_tick:
                    break
                if self.multi:
                    if (len(self._pending_outbox()) >= 2 * self.cfg.max_runs_per_tick
                            or not self._ready(d, now) or self.uncertainty.ask_target(d, now) is None):
                        continue          # nobody may be interrupted about this right now
                elif self._budget(now) <= 0:
                    break
                if self._ready(d, now):
                    runs += 1
                    rep.asked_runs.append(d.id)
                    self._run("ask", d, now, rep)
            self._flush(now, rep)
        return rep

    def _open_in_order(self, now: float) -> list:
        opened = self.store.list_doubts("open", ALL)
        if self.multi:
            return [r.doubt for r in self.uncertainty.rank_doubts(opened, now)]   # best value/cost first
        return opened

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
        if self.multi:
            return self._flush_multi(now, rep)
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

    def _outbound(self, d: Doubt) -> OutboundMessage:
        ob = d.detail["outbox"]
        images = []
        for p in ob.get("image_paths", ())[:2]:
            try:
                images.append(confine_image(p, self.cfg.snapshot_dir))
            except ToolInputError as e:
                log.warning("dropping image for %s: %s", d.id, e)
        choices = [Choice(c["label"], c["value"]) for c in ob.get("choices", ())]
        if not any(c.value in self.cfg.ignore_values for c in choices):
            choices.append(Choice("Ignore", self.cfg.ignore_values[0]))
        return OutboundMessage(ob["text"], tuple(images), tuple(choices))

    def _recipient(self, d: Doubt, now: float):
        """Follow-up rounds go back to whoever was asked before; otherwise pick by permission."""
        uid = d.detail.get("asked_to")
        if uid:
            u = self.dispatcher.users.get(uid)
            if u is not None and u.active:
                return u
        return self.uncertainty.ask_target(d, now)

    def _flush_multi(self, now: float, rep: TickReport) -> None:
        for d in sorted(self._pending_outbox(), key=lambda d: d.detail["outbox"].get("queued_at", 0)):
            user = self._recipient(d, now)
            if user is None:
                continue
            prio = int(d.detail.get("priority", 1))
            res = self.dispatcher.send(user, self._outbound(d), kind="question", priority=prio,
                                       profile=d.detail.get("profile"), now=now)
            if not res.sent:
                if res.reason == "send_failed":
                    self.store.update_doubt(d.id, last_error="send failed")
                continue                      # budget/quiet: stays queued; another doubt may fit another user
            self.store.update_doubt(d.id, message_ref=res.ref, detail=dict(
                _without(d.detail, "outbox"), sent_ts=list(d.detail.get("sent_ts", ())) + [now],
                asked_to=user.id, asked_role=user.role, asked_channel=res.channel))
            rep.sent.append(d.id)

    def _sweep_unanswered(self, now: float) -> None:
        """A question nobody answered within `unanswered_after_h` counts as ignored (back-off
        for that user) and returns to `open` so it can be re-asked later or to someone else."""
        limit = self.cfg.unanswered_after_h * 3600.0
        for d in self.store.list_doubts("asked", ALL):
            sent = d.detail.get("sent_ts") or ()
            uid = d.detail.get("asked_to")
            if (not uid or d.detail.get("outbox") or not sent or now - sent[-1] < limit
                    or d.detail.get("unanswered_recorded") == sent[-1]):
                continue
            self.dispatcher.budget.record(uid, "ignored", now)
            self.store.update_doubt(d.id, status="open", detail=dict(d.detail, unanswered_recorded=sent[-1]))

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

    def handle_user_reply(self, user, msg: InboundMessage, now: Optional[float] = None) -> Optional[str]:
        """Multi-user reply. Matches on the question's message ref and on who was asked.
        An answer from someone who may not confirm the topic does NOT settle the doubt
        (see UncertaintyManager.record_answer). Returns the doubt id or None."""
        now = time.time() if now is None else now
        if msg.reply_to_ref:
            d = self.store.find_doubt_by_message(str(msg.reply_to_ref))
        else:
            d = self.pending_question_for(user, msg, now)      # typed without using Telegram's "reply"
        if d is None or d.status not in ("asked", "answered", "open"):
            return None
        asked = d.detail.get("asked_to")
        if asked and asked != user.id:
            return None
        budget = self.dispatcher.budget if self.dispatcher is not None else None
        if msg.choice_value and msg.choice_value.strip().lower() in self.cfg.ignore_values:
            self.store.update_doubt(d.id, status="dismissed", answer=msg.choice_value, detail=dict(
                d.detail, ignored_by_owner=True, ignored_by=user.id, dismissed_reason="owner ignored",
                dismissed_at=now))
            if budget is not None:
                budget.record(user.id, "ignored", now)
            return d.id
        text = (msg.text or msg.choice_value or "").strip()
        if not text:
            return None
        if budget is not None:
            budget.record(user.id, "answered", now)
        if self.uncertainty is None:
            return None
        self.uncertainty.record_answer(d, user, text, now, self.cfg.max_answer_chars)
        return d.id

    def pending_question_for(self, user, msg: InboundMessage, now: Optional[float] = None):
        """The newest question this user was asked and hasn't answered, if `msg` is plain text.
        People type an answer; they do not use Telegram's reply gesture. Commands (/x) and chat
        quick-replies are never answers."""
        now = time.time() if now is None else now
        txt = (msg.text or "").strip()
        if not txt or txt.startswith("/") or (msg.choice_value or "").startswith("chat:"):
            return None
        limit = self.cfg.unanswered_after_h * 3600.0
        mine = [d for d in self.store.list_doubts("asked", 10_000)
                if d.detail.get("asked_to") == user.id and now - d.updated < limit]
        return max(mine, key=lambda d: d.updated) if mine else None

    def poll_once(self, timeout_s: float = 25.0, now: Optional[float] = None) -> TickReport:
        """Convenience loop body: read replies, then tick. A poller error (e.g.
        TelegramError on a 409 from a second poller) is logged and the tick still runs."""
        try:
            replies = self.notifier.poll_replies(timeout_s) if self.notifier is not None else ()
        except Exception as e:
            log.warning("poll_replies failed: %s", e)
            replies = ()
        for r in replies:
            self.handle_reply(r, now)
        return self.tick(now)


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
