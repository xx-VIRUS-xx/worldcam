"""The whole pipeline: camera -> detection/tracking -> faces/entities -> graph -> doubts
-> Gemini / claude -p worker -> Telegram -> owner's answer -> graph.

  worldcam-run --source 0                        # webcam, Neo4j, Telegram from .env
  worldcam-run --source 0 --worker gemini        # force Gemini API worker
  worldcam-run --source 1 --sub 0 ...            # (not recommended: different cameras)
  worldcam-run --source 0 --notifier log         # no Telegram: questions are only logged
  worldcam-run --source 0 --no-engine            # perception + identity + graph only
  worldcam-run --source clip.mp4 --store file:captures/graph.json --no-engine  # offline test

Threads (all bounded, nothing queues frames):
  main      grabber -> PerceptionLoop (motion gate, YOLO, tracker, best snapshot)
  identity  newest best snapshot per person track -> FaceAnalyzer -> IdentityResolver
            -> GraphStore -> DoubtDetector. Holds at most one pending item per track.
            Non-person tracks (animals, objects) go to EntityObserver.
  engine    AmbiguityEngine.poll_once(): Telegram replies, then worker runs and sends.
Ctrl-C stops everything.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Optional

from ..capture import LatestFrameGrabber
from ..sources import make_source
from ..sources.dual import DualGrabber

log = logging.getLogger("worldcam.run")

PERSON = "person"


# --------------------------------------------------------------- identity stage
def signal_from_result(res):
    """ResolveResult -> IdentitySignal (None when there is nothing to judge).
    Mapping documented in docs/ambiguity.md."""
    from ..ambiguity.triggers import IdentitySignal
    from ..identity.resolver import Decision

    if res.decision is Decision.NO_FACE or res.cached:
        return None  # no face, or already judged for this track (no new sighting stored)
    cands = tuple((c.person_id, float(c.score)) for c in res.candidates if c.person_id)
    return IdentitySignal(kind=res.decision.value, track_id=res.identity.track_id,
                          cluster_id=res.cluster_id, candidates=cands, crop_path=res.crop_path,
                          ts=res.ts, person_id=res.person_id)


class IdentityWorker(threading.Thread):
    """Runs face analysis off the perception thread. `submit` keeps only the
    newest best snapshot per track (a dict, bounded by live tracks), so a slow
    face model drops stale snapshots instead of building a backlog."""

    def __init__(self, analyzer, resolver, detector, max_pending: int = 64):
        super().__init__(name="identity", daemon=True)
        self.analyzer, self.resolver, self.detector = analyzer, resolver, detector
        self.max_pending = max_pending
        self._pending: dict[int, tuple] = {}      # track_id -> (track, ended)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._done_seq: dict[int, int] = {}       # track_id -> best.seq already analyzed
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
            return  # this snapshot was already analyzed
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
                tid = next(iter(self._pending))       # oldest-submitted track first
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
            sig = signal_from_result(res)
            if sig is not None and self.detector is not None:
                if self.detector.observe(sig):
                    self.stats["doubts"] += 1
            log.info("track %s: %s %s (q=%.2f, cluster=%s)", track.track_id, res.decision.value,
                     res.identity.name, res.quality, res.cluster_id)
        if ended:
            self.resolver.forget(track.track_id)
            self._done_seq.pop(track.track_id, None)


# ----------------------------------------------------------------- engine stage
class LogNotifier:
    """Stand-in when Telegram isn't configured: questions are logged, never answered."""

    def __init__(self):
        self._n = 0

    def ask(self, q):
        self._n += 1
        log.warning("QUESTION (not sent, --notifier log) doubt=%s: %s images=%s choices=%s",
                    q.doubt_id, q.text, list(q.image_paths), [c.label for c in q.choices])
        return f"log_{self._n}"

    def notify(self, text):
        log.info("NOTE (not sent): %s", text)

    def poll_replies(self, timeout_s=25.0):
        time.sleep(min(timeout_s, 5.0))
        return ()


class EngineThread(threading.Thread):
    def __init__(self, engine, store, resolver, poll_s: float = 25.0):
        super().__init__(name="engine", daemon=True)
        self.engine, self.store, self.resolver, self.poll_s = engine, store, resolver, poll_s
        self._stop = threading.Event()
        self.reports = {"sent": 0, "asked_runs": 0, "resolve_runs": 0, "failed": 0, "expired": 0}

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                rep = self.engine.poll_once(timeout_s=self.poll_s)
            except Exception:
                log.exception("engine tick failed")
                self._stop.wait(10)
                continue
            for k in self.reports:
                self.reports[k] += len(getattr(rep, k))
            # the worker (another process) may have labeled/merged clusters:
            # drop cached identities so tracks pick up the new names
            for did in rep.resolve_runs:
                d = self.store.get_doubt(did)
                for i in (d.about_ids if d else ()):
                    if i.startswith("fc_"):
                        self.resolver.invalidate(cluster_id=i)
                    elif i.startswith("p_"):
                        self.resolver.invalidate(person_id=i)
            if rep.sent or rep.resolve_runs or rep.failed:
                log.info("engine: sent=%s asked=%s resolved=%s failed=%s", rep.sent,
                         rep.asked_runs, rep.resolve_runs, rep.failed)


# ------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="main (full-res) source spec")
    ap.add_argument("--sub", default=None, help="optional substream spec (e.g. CCTV low-res stream)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sub-width", type=int, default=640, help="detection copy width")
    ap.add_argument("--camera", default="default", help="camera name stored on sightings")
    ap.add_argument("--store", default="neo4j", help="'neo4j' or 'file:<path>' (tests)")
    ap.add_argument("--notifier", choices=("telegram", "log"), default="telegram")
    ap.add_argument("--no-engine", action="store_true", help="don't ask questions")
    ap.add_argument("--no-motion", action="store_true", help="run the detector on every frame")
    ap.add_argument("--worker", choices=("auto", "gemini", "claude"), default="auto",
                    help="ambiguity worker backend (auto prefers Gemini if GEMINI_API_KEY is set)")
    ap.add_argument("--worker-model", default=None, help="model for worker (e.g. gemini-2.5-flash or sonnet)")
    ap.add_argument("--entity-labels", default=None,
                    help="comma-separated non-person YOLO labels to track for entity personas "
                         "(default: animals + movable objects; 'none' disables)")
    ap.add_argument("--entity-cooldown", type=float, default=300.0,
                    help="seconds before re-recording the same non-person entity label/zone")
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
    logging.getLogger("neo4j").setLevel(logging.WARNING)  # schema-setup notices are noise
    import os
    snapshot_dir = args.snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # heavy imports only here, so --help works without the extras
    from ..ambiguity.entities import DEFAULT_ENTITY_LABELS, EntityObserver
    from ..ambiguity.file_store import open_store
    from ..ambiguity.gemini import get_gemini_setting
    from ..ambiguity.triggers import DoubtDetector, TriggerConfig
    from ..identity import FaceAnalyzer, IdentityResolver
    from ..perception import FrameDiffMotionGate, IouTracker, PerceptionLoop
    from ..perception.detector import YoloDetector

    if not args.no_engine and args.worker == "gemini" and not get_gemini_setting("GEMINI_API_KEY"):
        print("error: --worker gemini requires GEMINI_API_KEY in the environment or .env",
              file=sys.stderr)
        return 2

    if args.entity_labels is None:
        entity_labels: tuple[str, ...] = DEFAULT_ENTITY_LABELS
    elif args.entity_labels.strip().lower() in ("none", "off", ""):
        entity_labels = ()
    else:
        entity_labels = tuple(x.strip().lower() for x in args.entity_labels.split(",") if x.strip())

    try:
        store = open_store(args.store)
    except Exception as e:
        if args.store == "neo4j":
            print(
                f"error: could not connect to Neo4j ({type(e).__name__}: {e}).\n"
                "Start Neo4j with: colima start && docker compose up -d\n"
                "Or run without Neo4j using: --store file:captures/graph.pkl",
                file=sys.stderr,
            )
            return 2
        raise
    analyzer = FaceAnalyzer(snapshot_dir=snapshot_dir)
    log.info("loading face model ...")
    analyzer.warmup()
    resolver = IdentityResolver(store, match_t=args.match_t, ask_t=args.ask_t, camera=args.camera)
    doubts = None if args.no_engine else DoubtDetector(
        store, TriggerConfig(match_t=args.match_t, ask_t=args.ask_t))
    identity = IdentityWorker(analyzer, resolver, doubts)
    entity_observer = (
        EntityObserver(store, doubts, snapshot_dir=snapshot_dir, camera=args.camera,
                       labels=entity_labels, cooldown_s=args.entity_cooldown)
        if entity_labels else None
    )

    engine_thread = None
    if not args.no_engine:
        from ..ambiguity.engine import AmbiguityEngine, EngineConfig
        from ..ambiguity.worker import make_worker_runner
        if args.notifier == "telegram":
            from ..messenger.telegram import TelegramNotifier
            notifier = TelegramNotifier.from_env(snapshot_dir=snapshot_dir)
        else:
            notifier = LogNotifier()
        runner = make_worker_runner(args.worker, args.store, snapshot_dir, model=args.worker_model)
        engine = AmbiguityEngine(store, notifier, runner, EngineConfig(snapshot_dir=snapshot_dir))
        engine_thread = EngineThread(engine, store, resolver)

    main_src = make_source(args.source, args.width, args.height, args.fps)
    if args.sub:
        grabber = DualGrabber(make_source(args.sub, 640, 360, args.fps), main_src, main_ring=5)
    else:
        grabber = LatestFrameGrabber(main_src)
    detector = YoloDetector()
    best_labels = (PERSON,) + tuple(l for l in entity_labels if l != PERSON)
    loop = PerceptionLoop(grabber, detector, IouTracker(best_labels=best_labels),
                          gate=None if args.no_motion else FrameDiffMotionGate(),
                          sub_width=args.sub_width)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    identity.start()
    if engine_thread:
        engine_thread.start()
    t0 = time.monotonic()
    next_stats = t0 + args.stats_every
    try:
        with grabber:
            log.info("running: source=%s store=%s engine=%s", args.source, args.store,
                     "off" if args.no_engine else args.notifier)
            for upd in loop.run(seconds=args.seconds or None):
                for t in upd.active:
                    if t.label == PERSON:
                        identity.submit(t)
                    elif entity_observer is not None:
                        entity_observer.observe(t)
                for t in upd.ended:
                    if t.label == PERSON:
                        identity.submit(t, ended=True)
                    elif entity_observer is not None:
                        entity_observer.observe(t, ended=True)
                now = time.monotonic()
                if args.stats_every and now >= next_stats:
                    next_stats = now + args.stats_every
                    log.info("stats: frames=%d skip=%.2f identity=%s decisions=%s entities=%s engine=%s",
                             loop.frames, loop.skip_ratio, identity.stats, identity.decisions,
                             entity_observer.stats if entity_observer else "off",
                             engine_thread.reports if engine_thread else "off")
                if stop.is_set():
                    break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        identity.stop()
        if engine_thread:
            engine_thread.stop()
        identity.join(timeout=5)
        if engine_thread:
            engine_thread.join(timeout=30)  # a poll can take up to poll_s
    log.info("done: frames=%d identity=%s decisions=%s", loop.frames, identity.stats, identity.decisions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
