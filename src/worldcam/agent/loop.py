"""AutonomousAgent: the long-running loop behind the camera.

One thread, one BOUNDED trigger queue. Triggers come from events, doubts, chat and
from an internal scheduler (fixed + random-jitter times). Each trigger runs one task;
every task is crash-isolated. The loop never messages a user directly: it creates
doubts (which the AmbiguityEngine asks) or hands messages to the Dispatcher, which
applies the attention budget and permissions.

Deterministic tests: inject `clock` and `rng`, never call `start()`, and drive the
loop with `step(now)`.
"""
from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

from ..brain.api import BrainEvent, new_event
from ..channels.api import OutboundMessage
from ..graph.api import Doubt, new_id
from ..users.api import User, can
from .api import LOOP_TASKS, TRIGGER_KINDS, MemoryContext, Profile, Trigger
from .profiles import ProfileSet, in_hours, is_state_rule, render, rule_matches_event
from .uncertainty import UncertaintyManager

log = logging.getLogger(__name__)
MOVEMENT_KINDS = ("entered", "left", "zone_changed", "seen", "activity_changed")


@dataclass
class AgentConfig:
    queue_size: int = 256
    poll_s: float = 0.5
    # task -> (every_s, jitter 0-1); a task absent here is never self-scheduled
    schedule: dict = field(default_factory=lambda: {
        "curiosity": (3600.0, 0.3), "look_around": (1800.0, 0.3),
        "consolidate": (6 * 3600.0, 0.2), "alert_review": (300.0, 0.1)})
    cameras: tuple = ("default",)
    alert_lookback_s: float = 900.0
    review_event_limit: int = 200
    chat_quiet_s: float = 2 * 3600.0       # skip a check-in for someone who just chatted
    checkin_retry_s: float = 1800.0
    dedupe_size: int = 2000


class AutonomousAgent:
    def __init__(self, ctx: MemoryContext, dispatcher, profiles: Union[ProfileSet, Sequence[Profile]], *,
                 uncertainty: Optional[UncertaintyManager] = None, config: Optional[AgentConfig] = None,
                 clock: Optional[Callable[[], float]] = None, rng: Optional[random.Random] = None,
                 describe_scene: Optional[Callable[..., Any]] = None,
                 miner: Optional[Callable[[float], Any]] = None,
                 summarizer: Optional[Callable[[float], Any]] = None,
                 engine_tick: Optional[Callable[[], Any]] = None):
        self.ctx, self.dispatcher = ctx, dispatcher
        self.profiles = profiles if isinstance(profiles, ProfileSet) else ProfileSet(profiles)
        self.cfg = config or AgentConfig()
        self.clock = clock or ctx.clock
        self.rng = rng or random.Random()
        self.uncertainty = uncertainty or UncertaintyManager(
            ctx.graph, ctx.users, budget=dispatcher.budget, profiles=self.profiles,
            routines=ctx.routines, clock=self.clock)
        self.describe_scene, self.miner, self.summarizer = describe_scene, miner, summarizer
        self.engine_tick = engine_tick
        self._q: queue.Queue = queue.Queue(maxsize=self.cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._next: dict = {}
        self._alerted: OrderedDict = OrderedDict()      # dedupe keys -> ts (bounded)
        self._last_review = 0.0
        self._last_chat: dict = {}
        self.stats = {"handled": 0, "dropped": 0, "errors": {}, "runs": {}}
        self.last_results: dict = {}
        self.errors: deque = deque(maxlen=50)

    # ---- AgentLoop protocol ---------------------------------------------------------
    def start(self) -> "AutonomousAgent":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="worldcam-agent", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5.0)
        self._thread = None

    def submit(self, trigger: Trigger) -> bool:
        if trigger.kind not in TRIGGER_KINDS:
            return False
        try:
            self._q.put_nowait(trigger)
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    # ---- thread ------------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._schedule_due(self.clock())
                try:
                    t = self._q.get(timeout=self.cfg.poll_s)
                except queue.Empty:
                    continue
                self._handle(t)
            except Exception:                       # the loop itself must not die
                log.exception("agent loop iteration failed")

    def step(self, now: Optional[float] = None, max_triggers: int = 1000) -> int:
        """Synchronous: enqueue due scheduled work, then handle everything queued."""
        now = self.clock() if now is None else now
        self._schedule_due(now)
        n = 0
        while n < max_triggers:
            try:
                t = self._q.get_nowait()
            except queue.Empty:
                break
            self._handle(t, now)
            n += 1
        return n

    # ---- scheduler -----------------------------------------------------------------------
    def _jittered(self, every: float, jitter: float) -> float:
        return max(1.0, every * (1.0 + jitter * self.rng.uniform(-1.0, 1.0)))

    def _schedule_due(self, now: float) -> None:
        for task, (every, jitter) in self.cfg.schedule.items():
            if task not in self._next:      # first run after a random fraction of the interval
                self._next[task] = now + every * self.rng.uniform(0.1, 1.0)
            if now >= self._next[task]:
                self._next[task] = now + self._jittered(every, jitter)
                self.submit(Trigger("random" if jitter else "schedule", now, {"task": task}))
        for c in self.profiles.checkins:
            key = "checkin:" + c["id"]
            every = c["every_hours"] * 3600.0
            if key not in self._next:
                self._next[key] = now + every * self.rng.uniform(0.1, 1.0)
            if now >= self._next[key]:
                self._next[key] = now + self._jittered(every, c.get("jitter", 0.0))
                self.submit(Trigger("random" if c.get("jitter") else "schedule", now,
                                    {"task": "checkin", "checkin_id": c["id"]}, profile=c["profile"]))

    # ---- dispatch ----------------------------------------------------------------------------
    def _handle(self, t: Trigger, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.stats["handled"] += 1
        if t.kind == "event":
            self._guard("event", self._on_event, t, now)
        elif t.kind == "chat":
            self._guard("chat", self._on_chat, t, now)
        elif t.kind == "doubt":
            self._guard("doubt", self._tick_engine)
        else:
            task = t.payload.get("task")
            if task in LOOP_TASKS:
                self.run_task(task, now, **{k: v for k, v in t.payload.items() if k != "task"})
            else:
                log.warning("unknown scheduled task %r", task)

    def _guard(self, name: str, fn: Callable, *a) -> Any:
        """Crash isolation: an exception is logged and counted; the loop continues."""
        try:
            return fn(*a)
        except Exception as e:
            log.exception("agent task %s failed", name)
            self.stats["errors"][name] = self.stats["errors"].get(name, 0) + 1
            self.errors.append((name, f"{type(e).__name__}: {e}"))
            return None

    def run_task(self, task: str, now: Optional[float] = None, **kw) -> Any:
        now = self.clock() if now is None else now
        fn = {"curiosity": self.curiosity, "look_around": self.look_around, "consolidate": self.consolidate,
              "checkin": self.checkin, "alert_review": self.alert_review}[task]
        self.stats["runs"][task] = self.stats["runs"].get(task, 0) + 1
        res = self._guard(task, lambda: fn(now, **kw))
        self.last_results[task] = res
        return res

    def _tick_engine(self) -> None:
        if self.engine_tick:
            self.engine_tick()

    # ---- event / chat triggers ----------------------------------------------------------------------
    def _on_event(self, t: Trigger, now: float) -> None:
        e = t.payload.get("event") or self.ctx.events.get_event(t.payload.get("event_id", ""))
        if e is not None:
            self._eval_event_rules(e, now)

    def _on_chat(self, t: Trigger, now: float) -> None:
        uid = t.payload.get("user_id")
        if not uid:
            return
        self._last_chat[uid] = now
        if t.payload.get("curiosity", True):
            u = self.ctx.users.get(uid)
            if u is not None:
                self.curiosity(now, prefer_user=u)

    # ---- tasks ------------------------------------------------------------------------------------------
    def curiosity(self, now: float, prefer_user: Optional[User] = None) -> Optional[str]:
        """Find the most valuable gap somebody can be asked about; create ONE doubt for it."""
        if self.uncertainty.open_curiosity_count() >= self.uncertainty.cfg.max_open_curiosity:
            return None
        gap = self.uncertainty.best_gap(now)
        if gap is None:
            return None
        d = self.uncertainty.gap_to_doubt(gap, now)
        if prefer_user is not None:
            d.detail["prefer_user"] = prefer_user.id
        self.ctx.graph.create_doubt(d)
        self._tick_engine()
        return d.id

    def look_around(self, now: float) -> list:
        """Fresh snapshot per camera, handed to the scene-description callback."""
        got = []
        if self.ctx.look_now is None:
            return got
        for cam in self.cfg.cameras:
            path = self.ctx.look_now(cam)
            if not path:
                continue
            got.append((cam, path))
            if self.describe_scene is not None:
                self.describe_scene(cam, path, cloud_ok=self.profiles.cloud_images_ok(), ts=now)
        return got

    def consolidate(self, now: float) -> list:
        """Run mining/summary hooks, then PROPOSE duplicate merges as conflict doubts."""
        if self.miner is not None:
            self._guard("consolidate.miner", self.miner, now)
        if self.summarizer is not None:
            self._guard("consolidate.summary", self.summarizer, now)
        return self._propose_merges(now)

    def _propose_merges(self, now: float) -> list:
        taken = self.uncertainty._existing_keys(now)
        people = [p for p in self.ctx.graph.list_people(1000) if p.kind != "owner"]
        by_name: dict = {}
        for p in sorted(people, key=lambda p: (p.first_seen, p.id)):
            for nm in {x.strip().casefold() for x in (p.name, *p.aliases) if x and x.strip()}:
                by_name.setdefault(nm, []).append(p)
        made = []
        for nm, ps in by_name.items():
            for i in range(1, len(ps)):
                keep, drop = ps[0], ps[i]
                if keep.id == drop.id:
                    continue
                key = "merge:" + ":".join(sorted((keep.id, drop.id)))
                if key in taken:
                    continue
                taken.add(key)
                d = Doubt(id=new_id("db"), kind="conflict", about_ids=(keep.id, drop.id),
                          question=f"Are these two the same person: {keep.name or nm} and {drop.name or nm}?",
                          created=now, updated=now,
                          detail={"origin": "consolidate", "gap_key": key, "proposal": "merge",
                                  "keep_id": keep.id, "drop_id": drop.id, "topic": "identity", "priority": 1})
                self.ctx.graph.create_doubt(d)       # never merge here: the answer decides
                made.append(d.id)
        if made:
            self._tick_engine()
        return made

    def checkin(self, now: float, checkin_id: Optional[str] = None) -> list:
        sent = []
        for c in self.profiles.checkins:
            if checkin_id is not None and c["id"] != checkin_id:
                continue
            if not in_hours(c.get("hours"), now):
                self._next["checkin:" + c["id"]] = min(self._next.get("checkin:" + c["id"], now + 1e12),
                                                       now + self.cfg.checkin_retry_s)
                continue
            for u in self._checkin_audience(c):
                if now - self._last_chat.get(u.id, -1e12) < self.cfg.chat_quiet_s:
                    continue
                res = self.dispatcher.send(u, OutboundMessage(c["prompt"], silent=bool(c.get("quiet"))),
                                           kind="checkin", priority=int(c.get("priority", 1)),
                                           profile=c["profile"], now=now)
                if res.sent:
                    sent.append((c["id"], u.id))
        return sent

    def _checkin_audience(self, c: dict) -> list:
        if c.get("audience") == "self":
            return [u for u in self.ctx.users.list()
                    if u.person_id and c["profile"] in u.profiles and can(u, "chat")]
        return self.dispatcher.recipients(c.get("to_permission", "receive_companion"), c["profile"])

    # ---- alerts -----------------------------------------------------------------------------------------------
    def alert_review(self, now: float) -> list:
        fired = []
        since = max(self._last_review, now - self.cfg.alert_lookback_s) if self._last_review else now - self.cfg.alert_lookback_s
        evs = self.ctx.events.query(since=since, limit=self.cfg.review_event_limit, newest_first=False)
        for e in evs:
            fired += self._eval_event_rules(e, now)
        fired += self._eval_state_rules(now)
        self._last_review = now
        return fired

    def _mark(self, key: tuple, now: float) -> None:
        self._alerted[key] = now
        self._alerted.move_to_end(key)
        while len(self._alerted) > self.cfg.dedupe_size:
            self._alerted.popitem(last=False)

    def _cooling(self, rule: dict, subject: str, now: float) -> bool:
        ts = self._alerted.get(("cool", rule["id"], subject))
        return ts is not None and now - ts < rule.get("cooldown_min", 0) * 60.0

    def _eval_event_rules(self, e: BrainEvent, now: float) -> list:
        fired = []
        for rule in self.profiles.rules:
            if is_state_rule(rule) or ("ev", rule["id"], e.id) in self._alerted:
                continue
            if e.profiles and rule["profile"] not in e.profiles:
                continue
            if not rule_matches_event(rule, e):
                continue
            subject = e.subject_id or e.zone or "?"
            if self._cooling(rule, subject, now):
                self._mark(("ev", rule["id"], e.id), now)
                continue
            text = render(rule["text"], zone=e.zone or "camera", time=time.strftime("%H:%M", time.localtime(e.ts)),
                          subject=e.subject_label or "someone", activity=e.activity or "")
            if self._fire(rule, text, now, e.snapshot_paths, e.id, subject):
                self._mark(("ev", rule["id"], e.id), now)
                fired.append((rule["id"], e.id))
        return fired

    def _eval_state_rules(self, now: float) -> list:
        fired = []
        for rule in self.profiles.rules:
            w = rule["when"]
            if not is_state_rule(rule) or not in_hours(w.get("hours"), now):
                continue
            if "routine_state" in w:
                for st in self.ctx.routines.status(now):
                    if st.state != w["routine_state"]:
                        continue
                    r = st.routine
                    key = ("st", rule["id"], r.id, time.strftime("%Y-%m-%d", time.localtime(now)))
                    if key in self._alerted or self._cooling(rule, r.id, now):
                        continue
                    text = render(rule["text"], routine=r.description, subject=r.subject_id)
                    if self._fire(rule, text, now, (), None, r.id):
                        self._mark(key, now)
                        fired.append((rule["id"], r.id))
            if "no_movement_minutes" in w:
                mins = int(w["no_movement_minutes"])
                if self._cooling(rule, "nomove", now):
                    continue
                recent = [e for e in self.ctx.events.query(since=now - mins * 60.0, limit=50)
                          if e.kind in MOVEMENT_KINDS]
                alive = [e for e in self.ctx.events.query(since=now - 86400.0, limit=50) if e.kind in MOVEMENT_KINDS]
                if not recent and alive:        # silence only counts if the system was seeing things today
                    if self._fire(rule, render(rule["text"], minutes=mins), now, (), None, "nomove"):
                        fired.append((rule["id"], "nomove"))
        return fired

    def _fire(self, rule: dict, text: str, now: float, images: Sequence[str], event_id: Optional[str],
              subject: str) -> bool:
        results = self.dispatcher.alert(rule["to_permission"], text, priority=rule["priority"],
                                        profile=rule["profile"], images=images, now=now)
        got = [r.user_id for r in results if r.sent]
        if not got:
            return False
        self._alerted[("cool", rule["id"], subject)] = now
        try:
            self.ctx.events.add_event(new_event(
                "alert", ts=now, source="inferred", profiles=(rule["profile"],), subject_id=None,
                subject_label=subject, detail={"rule": rule["id"], "event_id": event_id, "sent_to": got,
                                               "priority": rule["priority"], "text": text}))
        except Exception:
            log.exception("could not record alert event")
        return True
