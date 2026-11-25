"""The whole agent: eyes -> brain -> autonomous agent -> chat, on one core.

  camera -> motion gate -> YOLO -> tracker -> best snapshots
    people   -> faces (ArcFace) -> graph match -> doubts
    animals/objects -> appearance embedding -> graph match -> doubts
    every track -> EventRecorder -> timeline (BrainEvents)
  AutonomousAgent: event + schedule + random loops (curiosity, look-around,
    consolidate, check-ins, alert review) under a per-user attention budget
  AmbiguityEngine: doubts -> worker (claude -p / gemini via ModelRouter) -> Telegram
  ChannelGateway: whitelisted users -> chat (memory-grounded) or doubt replies

  worldcam-run --source 0                         # profiles from --profiles, Telegram from .env
  worldcam-run --source 0 --channel log           # no Telegram: messages are logged (dev)
  worldcam-run --source 0 --no-engine             # eyes + brain only: never ask, never chat
  worldcam-run --source 0 --profiles cctv,companion,medical --scene   # + vision scene descriptions

Threads (all bounded; nothing queues frames):
  main      grabber -> PerceptionLoop -> submit work, record events
  identity  newest best snapshot per person track -> faces -> resolver -> doubts
  scene     (--scene) latest gated snapshot -> vision model -> fuser -> activity events
  agent     triggers + schedule -> curiosity / look-around / consolidate / check-in / alerts
  gateway   one Telegram poller -> doubt replies or chat
Ctrl-C stops everything.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..capture import LatestFrameGrabber
from ..sources import DEFAULT_SOURCE, make_source
from ..sources.dual import DualGrabber

log = logging.getLogger("worldcam.run")

PERSON = "person"
STATE_DIR = Path("captures")


# ----------------------------------------------------------------------- clocks
def to_wall(upd, off: float):
    """Frame/track times are time.monotonic() (seconds since boot). Everything stored in the
    graph, timeline and doubts must be wall-clock epoch seconds (a monotonic value reads as
    1970 and makes the engine expire the doubt as 'older than 7 days'). Convert once, here."""
    from dataclasses import replace

    def tr(t):
        b = t.best
        return replace(t, first_ts=t.first_ts + off, last_ts=t.last_ts + off,
                       stationary_since=(t.stationary_since + off if t.stationary_since else None),
                       best=(replace(b, ts=b.ts + off) if b is not None else None))
    return replace(upd, active=[tr(t) for t in upd.active], ended=[tr(t) for t in upd.ended])


# --------------------------------------------------------------- identity stage
def signal_from_result(res):
    """ResolveResult -> IdentitySignal (None when there is nothing to judge)."""
    from ..ambiguity.triggers import IdentitySignal
    from ..identity.resolver import Decision

    if res.decision is Decision.NO_FACE or res.cached:
        return None
    cands = tuple((c.person_id, float(c.score)) for c in res.candidates if c.person_id)
    return IdentitySignal(kind=res.decision.value, track_id=res.identity.track_id,
                          cluster_id=res.cluster_id, candidates=cands, crop_path=res.crop_path,
                          ts=res.ts, person_id=res.person_id)


class IdentityWorker(threading.Thread):
    """Runs face analysis off the perception thread. `submit` keeps only the
    newest best snapshot per track (a dict, bounded by live tracks), so a slow
    face model drops stale snapshots instead of building a backlog."""

    def __init__(self, analyzer, resolver, detector, max_pending: int = 64,
                 on_result: Optional[Callable] = None, on_doubt: Optional[Callable] = None):
        super().__init__(name="identity", daemon=True)
        self.analyzer, self.resolver, self.detector = analyzer, resolver, detector
        self.on_result = on_result              # called with every ResolveResult (timeline)
        self.on_doubt = on_doubt                # called with a new doubt id (wakes the question engine)
        self.max_pending = max_pending
        self._pending: dict[int, tuple] = {}
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._done_seq: dict[int, int] = {}
        self.stats = {"submitted": 0, "replaced": 0, "analyzed": 0, "faces": 0,
                      "doubts": 0, "errors": 0, "dropped_full": 0}
        self.decisions: dict[str, int] = {}

    def submit(self, track, ended: bool = False) -> None:
        best = track.best
        if best is None or best.crop is None:
            if ended:
                self.resolver.forget(track.track_id)
                self._done_seq.pop(track.track_id, None)
            return
        if not ended and self._done_seq.get(track.track_id) == best.seq:
            return
        with self._cv:
            if track.track_id in self._pending:
                self.stats["replaced"] += 1
                ended = ended or self._pending[track.track_id][1]
            elif len(self._pending) >= self.max_pending:
                self.stats["dropped_full"] += 1
                return
            self._pending[track.track_id] = (track, ended)
            self.stats["submitted"] += 1
            self._cv.notify()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                tid = next(iter(self._pending))
                track, ended = self._pending.pop(tid)
            try:
                self._process(track, ended)
            except Exception:
                self.stats["errors"] += 1
                log.exception("identity failed for track %s", tid)

    def _process(self, track, ended: bool) -> None:
        best = track.best
        if self._done_seq.get(track.track_id) != best.seq:
            faces = self.analyzer.analyze(best.crop, track, best.ts)
            self._done_seq[track.track_id] = best.seq
            self.stats["analyzed"] += 1
            self.stats["faces"] += len(faces)
            res = self.resolver.resolve(track, faces)
            self.decisions[res.decision.value] = self.decisions.get(res.decision.value, 0) + 1
            if self.on_result is not None:
                try:
                    self.on_result(res)
                except Exception:
                    log.exception("on_result failed")
            sig = signal_from_result(res)
            if sig is not None and self.detector is not None:
                did = self.detector.observe(sig)
                if did:
                    self.stats["doubts"] += 1
                    if self.on_doubt is not None:
                        self.on_doubt(did)
            log.info("track %s: %s %s (q=%.2f, cluster=%s)", track.track_id, res.decision.value,
                     res.identity.name, res.quality, res.cluster_id)
        if ended:
            self.resolver.forget(track.track_id)
            self._done_seq.pop(track.track_id, None)


# ------------------------------------------------------------------ scene stage
class SceneWorker(threading.Thread):
    """(--scene) Gated vision calls off the perception thread. Holds ONE pending
    item: a newer gated update replaces an unprocessed older one."""

    def __init__(self, describer, fuser, events, profile_names, camera, on_event=None):
        super().__init__(name="scene", daemon=True)
        self.describer, self.fuser, self.events = describer, fuser, events
        self.profile_names, self.camera, self.on_event = tuple(profile_names), camera, on_event
        self._item = None
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.stats = {"submitted": 0, "described": 0, "events": 0, "errors": 0}

    def submit(self, tracks) -> None:
        with self._cv:
            self._item = list(tracks)
            self.stats["submitted"] += 1
            self._cv.notify()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while self._item is None and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                tracks, self._item = self._item, None
            try:
                self._process(tracks)
            except Exception:
                self.stats["errors"] += 1
                log.exception("scene description failed")

    def _process(self, tracks) -> None:
        people = [t for t in tracks if t.label == PERSON and t.best is not None and t.best.crop is not None]
        if not people:
            return
        hyp = self.describer.describe(people, {"profiles": None})
        self.stats["described"] += 1
        others = [t.label for t in tracks if t.label != PERSON]
        for t in people:
            zone = t.zones[0] if t.zones else None
            res = self.fuser.fuse_detailed(hyp, t, others, max(0.0, t.last_ts - t.first_ts))
            ev = self.fuser.event_if_changed(res, t, camera=self.camera, profiles=self.profile_names)
            if ev is not None:
                self.events.add_event(ev)
                self.stats["events"] += 1
                if self.on_event:
                    self.on_event(ev)
                log.info("scene: track %s %s (%.2f) zone=%s", t.track_id, res.activity, res.confidence, zone)


# --------------------------------------------------------------------- channels
class LogChannel:
    """--channel log: nothing is sent anywhere; messages are logged. Never receives.
    It stands in for Telegram (same channel name), so your real owner account is used
    and no separate dev user is needed."""
    name = "telegram"

    def __init__(self):
        self._n = 0

    def send(self, recipient_id, msg):
        self._n += 1
        log.warning("MESSAGE to %s: %s | images=%s choices=%s", recipient_id, msg.text,
                    list(msg.images), [c.label for c in msg.choices])
        return f"log_{self._n}"

    def poll(self, timeout_s=25.0):
        time.sleep(min(timeout_s, 5.0))
        return ()


def find_claude() -> Optional[str]:
    p = shutil.which("claude")
    if p:
        return p
    alt = Path.home() / ".local" / "bin" / "claude"
    return str(alt) if alt.exists() else None


def build_router(path: str = "models.json"):
    """Explicit per-task model config; claude-only defaults when no models.json exists
    (Claude CLI handles images too, so vision works without a Gemini key)."""
    import json

    from ..models.router import ConfiguredModelRouter
    if Path(path).exists():
        cfg = json.loads(Path(path).read_text())
    else:
        cfg = {"agent": "claude:sonnet", "chat": "claude:sonnet", "question": "claude:haiku",
               "resolve": "claude:sonnet", "vision_crop": "claude:sonnet", "vision_scene": "claude:sonnet",
               "decide": "claude:haiku", "summarize": "claude:sonnet", "options": {}}
    opts = dict(cfg.get("options") or {})
    cb = find_claude()
    if cb and opts.get("claude_bin", "claude") in ("claude", None):
        opts["claude_bin"] = cb
    opts.setdefault("usage_log", str(STATE_DIR / "usage.jsonl"))
    cfg["options"] = opts
    return ConfiguredModelRouter(cfg)


# ------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help=f"main (full-res) source spec (default {DEFAULT_SOURCE}; env WORLDCAM_SOURCE)")
    ap.add_argument("--sub", default=None, help="optional substream spec (e.g. CCTV low-res stream)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sub-width", type=int, default=640, help="detection copy width")
    ap.add_argument("--camera", default="default", help="camera name stored on sightings/events")
    ap.add_argument("--profiles", default="cctv",
                    help="comma list of agent profiles: cctv,companion,medical or a path to a profile .json")
    ap.add_argument("--store", default="neo4j", help="graph: 'neo4j' or 'file:<path>' (tests)")
    ap.add_argument("--events-db", default=str(STATE_DIR / "brain.db"), help="SQLite timeline/routines file")
    ap.add_argument("--channel", choices=("telegram", "log"), default="telegram")
    ap.add_argument("--no-engine", action="store_true",
                    help="eyes + brain only: no doubts, no agent loop, no chat")
    ap.add_argument("--no-chat", action="store_true", help="don't poll the channel for user messages")
    ap.add_argument("--scene", action="store_true",
                    help="gated vision scene descriptions -> activity events (uses the vision model)")
    ap.add_argument("--no-motion", action="store_true", help="run the detector on every frame")
    ap.add_argument("--quiet-hours", default=None, metavar="H-H|off",
                    help="no questions between these local hours for every user, e.g. 23-8; "
                         "'off' disables (default: each user's own setting, 23-8 unless changed)")
    ap.add_argument("--worker", choices=("router", "auto", "gemini", "claude"), default="router",
                    help="doubt worker backend (router = models.json, explicit, no silent fallback)")
    ap.add_argument("--worker-model", default=None, help="model for a forced --worker backend")
    ap.add_argument("--models", default="models.json", help="per-task model config (see models.example.json)")
    ap.add_argument("--entity-labels", default=None,
                    help="comma-separated non-person YOLO labels to track for entity personas "
                         "(default: cat,dog,backpack,handbag,suitcase,laptop; 'none' disables)")
    ap.add_argument("--animal-conf", type=float, default=0.6,
                    help="minimum YOLO confidence for animals (blankets and hands get mistaken for cats)")
    ap.add_argument("--entity-cooldown", type=float, default=300.0,
                    help="seconds before re-recording the same non-person entity")
    ap.add_argument("--match-t", type=float, default=0.45)
    ap.add_argument("--ask-t", type=float, default=0.35)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--stats-every", type=float, default=30.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    # fail fast on config problems, before loading any model
    if args.worker == "gemini" and not args.no_engine:
        from ..ambiguity.gemini import get_gemini_setting
        if not get_gemini_setting("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is not set (environment or .env); --worker gemini needs it. "
                  "Use --worker router (models.json) or claude.", file=sys.stderr)
            return 2
    snapshot_dir = os.path.abspath(args.snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots"))
    os.makedirs(snapshot_dir, exist_ok=True)       # absolute: stored paths must work from any cwd
    STATE_DIR.mkdir(exist_ok=True)

    # heavy imports only here, so --help works without the extras
    from ..agent.api import MemoryContext, Trigger
    from ..agent.budget import SqliteAttentionBudget
    from ..agent.dispatcher import Dispatcher
    from ..agent.loop import AutonomousAgent
    from ..agent.profiles import ProfileSet, load_profiles
    from ..agent.uncertainty import UncertaintyManager
    from ..ambiguity.engine import AmbiguityEngine, EngineConfig
    from ..ambiguity.entities import DEFAULT_ENTITY_LABELS, EntityObserver
    from ..ambiguity.file_store import open_store
    from ..ambiguity.triggers import DoubtDetector, TriggerConfig
    from ..brain.recorder import EventRecorder
    from ..brain.store import SqliteEventStore
    from ..channels.gateway import ChannelGateway
    from ..chat.handler import AgentChatHandler
    from ..identity import FaceAnalyzer, IdentityResolver
    from ..models.api import NotConfigured
    from ..perception import FrameDiffMotionGate, IouTracker, PerceptionLoop
    from ..perception.detector import YoloDetector
    from ..perception.reid import EntityEmbedder
    from ..users.store import open_directory

    if args.entity_labels is None:
        entity_labels: tuple[str, ...] = DEFAULT_ENTITY_LABELS
    elif args.entity_labels.strip().lower() in ("none", "off", ""):
        entity_labels = ()
    else:
        entity_labels = tuple(x.strip().lower() for x in args.entity_labels.split(",") if x.strip())

    # ---- brain: graph + timeline ------------------------------------------------
    store = open_store(args.store)
    events = SqliteEventStore(args.events_db)
    profile_set = ProfileSet(load_profiles([p.strip() for p in args.profiles.split(",") if p.strip()]))
    profile_names = tuple(profile_set.names())

    # ---- eyes: faces, entity re-id, recorder ------------------------------------
    analyzer = FaceAnalyzer(snapshot_dir=snapshot_dir)
    log.info("loading face model ...")
    analyzer.warmup()
    resolver = IdentityResolver(store, match_t=args.match_t, ask_t=args.ask_t, camera=args.camera)
    doubts = None if args.no_engine else DoubtDetector(
        store, TriggerConfig(match_t=args.match_t, ask_t=args.ask_t))
    agent = None   # set below; the recorder callback needs it lazily
    recorder = EventRecorder(events, camera=args.camera,
                             on_event=lambda e: agent and agent.submit(
                                 Trigger("event", e.ts, {"event_id": e.id})))
    def nudge_engine(*_):
        """Wake the question engine (new doubt, new reply, or the periodic tick)."""
        if agent is not None:
            agent.submit(Trigger("doubt", time.time(), {}))

    identity = IdentityWorker(analyzer, resolver, doubts, on_result=recorder.on_resolve,
                              on_doubt=nudge_engine)
    entity_observer = (
        EntityObserver(store, doubts, embedder=EntityEmbedder("auto"), snapshot_dir=snapshot_dir,
                       camera=args.camera, labels=entity_labels, cooldown_s=args.entity_cooldown)
        if entity_labels else None)

    grabber_ref: dict = {}

    def look_now(camera: str) -> Optional[str]:
        g = grabber_ref.get("g")
        f = (g.main.peek() if hasattr(g, "main") else g.peek()) if g is not None else None
        if f is None:
            return None
        import cv2
        p = Path(snapshot_dir) / f"look_{camera}_{int(time.time() * 1000)}.jpg"
        return str(p) if cv2.imwrite(str(p), f.image, [cv2.IMWRITE_JPEG_QUALITY, 90]) else None

    ctx = MemoryContext(graph=store, events=events, routines=events, users=None, look_now=look_now)

    # ---- agent + chat -----------------------------------------------------------
    channel = gateway = scene = None
    engine = None
    if not args.no_engine:
        users = open_directory(STATE_DIR / "users.db")
        ctx.users = users
        try:
            router = build_router(args.models)
        except NotConfigured as e:
            log.error("model config problem: %s", e)
            return 2
        log.info("models: %s", router.describe())

        if args.channel == "telegram":
            from ..channels.telegram_channel import TelegramChannel
            from ..messenger.telegram import get_setting
            token = get_setting("TELEGRAM_BOT_TOKEN")
            if not token:
                log.error("TELEGRAM_BOT_TOKEN missing (.env); run worldcam-telegram-setup or use --channel log")
                return 2
            channel = TelegramChannel(token, snapshot_dir=snapshot_dir)
        else:
            channel = LogChannel()
            if not any(u.role == "owner" for u in users.list()):    # fresh install, no Telegram yet
                users.add("Local owner", "owner", "telegram", "local", profiles=("cctv", "companion", "medical"))
        owners = [u for u in users.list() if u.role == "owner"]
        if not owners:
            log.error("no owner user: set TELEGRAM_OWNER_CHAT_ID in .env (bootstrap) or run worldcam-users add")
            return 2

        budget = SqliteAttentionBudget(str(STATE_DIR / "attention.db"))
        if args.quiet_hours:
            qh = args.quiet_hours.strip().lower()
            window = (None, None) if qh in ("off", "none") else tuple(int(x) for x in qh.split("-"))
            for u in users.list():
                budget.set_quiet_hours(u.id, *window)
            log.info("quiet hours set to %s for %d user(s)", qh, len(users.list()))
        dispatcher = Dispatcher(users, [channel], budget)
        uncertainty = UncertaintyManager(store, users, budget=budget, profiles=profile_set, routines=events)
        if args.worker == "router":
            from ..ambiguity.worker import RoutedWorkerRunner
            runner = RoutedWorkerRunner(args.store, snapshot_dir, router)
        else:
            from ..ambiguity.worker import make_worker_runner
            runner = make_worker_runner(args.worker, args.store, snapshot_dir, model=args.worker_model)
        engine = AmbiguityEngine(store, None, runner, EngineConfig(snapshot_dir=snapshot_dir),
                                 dispatcher=dispatcher, uncertainty=uncertainty)

        def engine_tick():
            rep = engine.tick()
            for did in rep.resolve_runs:          # worker labeled/merged: drop stale cached identities
                d = store.get_doubt(did)
                for i in (d.about_ids if d else ()):
                    if i.startswith("fc_"):
                        resolver.invalidate(cluster_id=i)
                    elif i.startswith("p_"):
                        resolver.invalidate(person_id=i)
            if rep.sent or rep.resolve_runs or rep.failed:
                log.info("engine: sent=%s asked=%s resolved=%s failed=%s", rep.sent,
                         rep.asked_runs, rep.resolve_runs, rep.failed)
            return rep

        describe_cb = None
        if args.scene:
            from ..perception.eventgate import ActivityEventGate
            from ..perception.fuser import RuleActivityFuser
            from ..perception.scene import SceneDescriber
            describer = SceneDescriber(router.client("vision_scene"), profile_set.profiles,
                                       cloud=True, include_full_frame=False)
            fuser = RuleActivityFuser()
            scene_gate = ActivityEventGate(profile_set.profiles)
            scene = SceneWorker(describer, fuser, events, profile_names, args.camera,
                                on_event=lambda e: agent and agent.submit(
                                    Trigger("event", e.ts, {"event_id": e.id})))
        agent = AutonomousAgent(ctx, dispatcher, profile_set, uncertainty=uncertainty,
                                describe_scene=describe_cb, engine_tick=engine_tick)

        if not args.no_chat:
            chat = AgentChatHandler(
                ctx, router=router, snapshot_dir=snapshot_dir,
                is_doubt_reply=lambda m: bool(
                    (m.reply_to_ref and store.find_doubt_by_message(str(m.reply_to_ref)))
                    or (m.choice_value and not m.choice_value.startswith("chat:"))))

            def on_message(user, msg):
                try:
                    if chat.is_doubt_reply(msg):
                        engine.handle_user_reply(user, msg)
                        nudge_engine()          # run the resolve step now
                    else:
                        out = chat.handle(user, msg)
                        if out is not None:
                            channel.send(user.channel_ids.get(channel.name, msg.sender_id), out)
                    agent.submit(Trigger("chat", time.time(), {"user_id": user.id}))
                except Exception:
                    log.exception("message handling failed")

            gateway = ChannelGateway(channel, users, on_message)

    # ---- eyes: grabber + perception loop ---------------------------------------
    main_src = make_source(args.source, args.width, args.height, args.fps)
    if args.sub:
        grabber = DualGrabber(make_source(args.sub, 640, 360, args.fps), main_src, main_ring=5)
    else:
        grabber = LatestFrameGrabber(main_src)
    grabber_ref["g"] = grabber
    from ..perception.filters import ANIMALS, FilteredDetector
    detector = FilteredDetector(YoloDetector(), {a: args.animal_conf for a in ANIMALS})
    best_labels = (PERSON,) + tuple(l for l in entity_labels if l != PERSON)
    loop = PerceptionLoop(grabber, detector, IouTracker(best_labels=best_labels),
                          gate=None if args.no_motion else FrameDiffMotionGate(),
                          sub_width=args.sub_width)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def _ticker():                 # releases quiet-hour queues, expiries, outbox, resolve runs
        nudge_engine()             # first tick right away: asks doubts left open by earlier runs
        while not stop.wait(20.0):
            nudge_engine()
    if agent is not None:
        threading.Thread(target=_ticker, name="engine-ticker", daemon=True).start()
    identity.start()
    if scene:
        scene.start()
    if agent:
        agent.start()
    if gateway:
        gateway.start()
    t0 = time.monotonic()
    next_stats = t0 + args.stats_every
    next_flush = t0 + 5.0
    try:
        with grabber:
            log.info("running: source=%s graph=%s profiles=%s engine=%s channel=%s scene=%s", args.source,
                     args.store, ",".join(profile_names), "off" if args.no_engine else "on",
                     channel.name if channel else "-", bool(scene))
            wall_off = time.time() - time.monotonic()
            for upd in loop.run(seconds=args.seconds or None):
                upd = to_wall(upd, wall_off)
                recorder.on_track_update(upd)
                for t in upd.active:
                    if t.label == PERSON:
                        identity.submit(t)
                    elif entity_observer is not None:
                        if entity_observer.observe(t):
                            nudge_engine()
                for t in upd.ended:
                    if t.label == PERSON:
                        identity.submit(t, ended=True)
                    elif entity_observer is not None:
                        if entity_observer.observe(t, ended=True):
                            nudge_engine()
                if scene is not None and scene_gate.should_fire(upd):
                    scene.submit(upd.active)
                now = time.monotonic()
                if now >= next_flush:
                    next_flush = now + 5.0
                    recorder.flush()
                if args.stats_every and now >= next_stats:
                    next_stats = now + args.stats_every
                    log.info("stats: frames=%d skip=%.2f identity=%s decisions=%s entities=%s scene=%s",
                             loop.frames, loop.skip_ratio, identity.stats, identity.decisions,
                             entity_observer.stats if entity_observer else "off",
                             scene.stats if scene else "off")
                if stop.is_set():
                    break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        stop.set()
        for t in (identity, scene):
            if t:
                t.stop()
        if agent:
            agent.stop()
        if gateway:
            gateway.stop()
        identity.join(timeout=5)
        recorder.flush()
    log.info("done: frames=%d identity=%s decisions=%s", loop.frames, identity.stats, identity.decisions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
