"""Live smoke test: real `claude -p` worker + file: store + in-memory notifier.

    PYTHONPATH=src python -m worldcam.ambiguity.smoke --dir /tmp/wc-smoke \
        --reply "That's my colleague Rahul from work, he visits on Fridays"

Seeds an unknown_face doubt (3 sightings of one face cluster, dummy crop),
runs engine.tick() (worker in `ask` mode), feeds the reply to
engine.handle_reply(), runs engine.tick() again (worker in `resolve` mode) and
prints the resulting graph. Uses the owner's Claude login; no camera, no Neo4j,
no Telegram.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from worldcam.ambiguity.engine import AmbiguityEngine, EngineConfig
from worldcam.ambiguity.file_store import FileGraphStore
from worldcam.ambiguity.triggers import DoubtDetector, IdentitySignal
from worldcam.ambiguity.worker import ClaudeWorkerRunner
from worldcam.graph.api import Sighting, new_id
from worldcam.messenger.base import FakeNotifier, OwnerReply


class TimedRunner(ClaudeWorkerRunner):
    def run(self, mode, doubt):
        res = super().run(mode, doubt)
        try:
            env = json.loads(self.last_stdout)
            trimmed = {k: env.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms",
                                               "total_cost_usd", "permission_denials", "structured_output")}
        except Exception:
            trimmed = {"raw": self.last_stdout[:300]}
        print(f"\n--- worker {mode} ({res.wall_s:.1f}s wall) ok={res.ok} error={res.error}")
        print(json.dumps(trimmed, indent=1))
        return res


def seed(store: FileGraphStore, snaps: str) -> str:
    os.makedirs(snaps, exist_ok=True)
    crop = os.path.join(snaps, "door_crop_001.jpg")
    try:
        import numpy as np, cv2  # noqa: E401
        img = np.full((96, 96, 3), 180, np.uint8)
        cv2.circle(img, (48, 44), 28, (90, 120, 200), -1)
        cv2.imwrite(crop, img)
    except Exception:
        open(crop, "wb").write(b"\xff\xd8\xff\xe0dummy\xff\xd9")
    owner = store.upsert_person(name="Owner", kind="owner")
    store.upsert_person(name="Prabhat", kind="household")
    rnd = random.Random(7)
    base = [rnd.gauss(0, 1) for _ in range(512)]
    t0 = time.time() - 3 * 3600
    cid = None
    for i in range(3):
        emb = [b + rnd.gauss(0, 0.05) for b in base]
        _, cid = store.add_face(emb, quality=0.9, ts=t0 + i * 3600, snapshot_path=crop, cluster_id=cid)
        store.add_sighting(Sighting(new_id("s"), t0 + i * 3600, "door", 10 + i, cluster_id=cid,
                                    zones=("door",), snapshot_path=crop))
    did = DoubtDetector(store).observe(IdentitySignal("unknown_cluster", track_id=12, cluster_id=cid,
                                                      crop_path=crop, ts=t0 + 7200))
    assert did, "detector did not open a doubt"
    print(f"seeded owner={owner} cluster={cid} doubt={did}")
    return did


def show(store: FileGraphStore, did: str) -> None:
    g = store.snapshot()
    d = g.doubts[did]
    print("\n=== graph after run ===")
    print(json.dumps({"doubt": {"status": d.status, "rounds": d.rounds, "question": d.question,
                                "answer": d.answer, "last_error": d.last_error,
                                "resolution": d.detail.get("resolution"),
                                "dismissed_reason": d.detail.get("dismissed_reason")},
                      "people": [{"id": p.id, "name": p.name, "kind": p.kind,
                                  "persona": p.persona_summary} for p in g.people.values()],
                      "clusters": {cid: c["person_id"] for cid, c in g.clusters.items()},
                      "facts": [{"person": f.person_id, "key": f.key, "value": f.value,
                                 "source": f.source, "confidence": f.confidence} for f in g._facts.values()],
                      "relations": g._rels}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reply", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--claude", default="claude")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    path, snaps = os.path.join(a.dir, "graph.pkl"), os.path.join(a.dir, "snapshots")
    if os.path.exists(path):
        os.remove(path)
    store = FileGraphStore(path)
    did = seed(store, snaps)
    runner = TimedRunner(f"file:{path}", snaps, claude_bin=a.claude, model=a.model)
    notifier = FakeNotifier()
    # quiet hours disabled so the smoke test runs at any time of day
    eng = AmbiguityEngine(store, notifier, runner, EngineConfig(snapshot_dir=snaps, quiet_start_hour=0,
                                                                quiet_end_hour=0))
    rep = eng.tick()
    print("tick 1:", rep)
    if not notifier.sent:
        show(store, did)
        return 1
    ref, q = notifier.sent[-1]
    print("\n>>> question sent to owner:", json.dumps({"ref": ref, "text": q.text, "images": q.image_paths,
                                                      "choices": [c.value for c in q.choices]}))
    print(">>> owner replies:", a.reply)
    eng.handle_reply(OwnerReply(ref, text=a.reply, ts=time.time()))
    rep = eng.tick()
    print("tick 2:", rep)
    if len(notifier.sent) > 1:
        print(">>> follow-up sent:", notifier.sent[-1][1].text)
    show(store, did)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
