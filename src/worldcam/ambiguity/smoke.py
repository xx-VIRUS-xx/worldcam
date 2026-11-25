"""Live smoke test: real Gemini or `claude -p` worker + file: store + in-memory notifier.

    PYTHONPATH=src python -m worldcam.ambiguity.smoke --dir /tmp/wc-smoke \
        --worker gemini --entity-kind animal \
        --reply "That's Bruno, our golden retriever, he loves sleeping by the door"

Seeds an unknown_face (person) or unknown_entity (animal/object) doubt,
runs engine.tick() (worker in `ask` mode), feeds the reply to
engine.handle_reply(), runs engine.tick() again (worker in `resolve` mode) and
prints the resulting graph. No camera, no Neo4j, no Telegram.
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
from worldcam.ambiguity.worker import ClaudeWorkerRunner, GeminiWorkerRunner, make_worker_runner
from worldcam.graph.api import Sighting, new_id
from worldcam.messenger.base import FakeNotifier, OwnerReply


class TimedClaudeRunner(ClaudeWorkerRunner):
    def run(self, mode, doubt):
        res = super().run(mode, doubt)
        try:
            env = json.loads(self.last_stdout)
            trimmed = {k: env.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms",
                                               "total_cost_usd", "permission_denials", "structured_output")}
        except Exception:
            trimmed = {"raw": self.last_stdout[:300]}
        print(f"\n--- claude {mode} ({res.wall_s:.1f}s wall) ok={res.ok} error={res.error}")
        print(json.dumps(trimmed, indent=1))
        return res


class TimedGeminiRunner(GeminiWorkerRunner):
    def run(self, mode, doubt):
        res = super().run(mode, doubt)
        print(f"\n--- gemini {mode} ({res.wall_s:.1f}s wall, model={self.model}) ok={res.ok} error={res.error}")
        print(json.dumps(res.data, indent=1))
        return res


def _dummy_crop(path: str, kind: str) -> None:
    try:
        import cv2
        import numpy as np
        img = np.full((128, 128, 3), 210, np.uint8)
        if kind == "animal":
            cv2.ellipse(img, (64, 72), (38, 28), 0, 0, 360, (60, 140, 210), -1)
            cv2.circle(img, (44, 46), 14, (60, 140, 210), -1)
        elif kind == "object":
            cv2.rectangle(img, (28, 28), (100, 108), (40, 40, 45), -1)
            cv2.rectangle(img, (40, 46), (88, 70), (50, 70, 200), -1)
        else:
            cv2.circle(img, (64, 56), 32, (90, 120, 200), -1)
        cv2.imwrite(path, img)
    except Exception:
        with open(path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0dummy\xff\xd9")


def seed(store: FileGraphStore, snaps: str, entity_kind: str = "person") -> str:
    os.makedirs(snaps, exist_ok=True)
    crop = os.path.join(snaps, f"door_{entity_kind}_001.jpg")
    _dummy_crop(crop, entity_kind)
    owner = store.upsert_person(name="Owner", kind="owner")
    store.upsert_person(name="Prabhat", kind="household")
    t0 = time.time() - 3 * 3600

    if entity_kind == "person":
        rnd = random.Random(7)
        base = [rnd.gauss(0, 1) for _ in range(512)]
        cid = None
        for i in range(3):
            emb = [b + rnd.gauss(0, 0.05) for b in base]
            _, cid = store.add_face(emb, quality=0.9, ts=t0 + i * 3600, snapshot_path=crop, cluster_id=cid)
            store.add_sighting(Sighting(new_id("s"), t0 + i * 3600, "door", 10 + i, cluster_id=cid,
                                        zones=("door",), snapshot_path=crop, entity_label="person"))
        did = DoubtDetector(store).observe(IdentitySignal("unknown_cluster", track_id=12, cluster_id=cid,
                                                          crop_path=crop, ts=t0 + 7200))
        assert did, "detector did not open a doubt"
        print(f"seeded owner={owner} cluster={cid} doubt={did}")
        return did

    label = "dog" if entity_kind == "animal" else "backpack"
    sid = new_id("s")
    store.add_sighting(Sighting(sid, t0 + 7200, "door", 21, zones=("door",), snapshot_path=crop,
                                entity_label=label))
    did = DoubtDetector(store).observe(IdentitySignal(
        kind="unknown_entity", track_id=21, crop_path=crop, ts=t0 + 7200,
        entity_type=entity_kind, entity_label=label, entity_key=f"ek_{entity_kind}_{label}_door",
        sighting_id=sid, zones=("door",),
    ))
    assert did, "detector did not open an entity doubt"
    print(f"seeded owner={owner} entity_type={entity_kind} label={label} sighting={sid} doubt={did}")
    return did


def show(store: FileGraphStore, did: str) -> None:
    g = store.snapshot()
    d = g.doubts[did]
    print("\n=== graph after run ===")
    print(json.dumps({"doubt": {"status": d.status, "kind": d.kind, "rounds": d.rounds,
                                "question": d.question, "answer": d.answer,
                                "last_error": d.last_error,
                                "resolution": d.detail.get("resolution"),
                                "dismissed_reason": d.detail.get("dismissed_reason")},
                      "entities": [{"id": p.id, "name": p.name, "entity_type": p.entity_type,
                                    "label": p.label, "kind": p.kind,
                                    "persona": p.persona_summary} for p in g.people.values()],
                      "clusters": {cid: c["person_id"] for cid, c in g.clusters.items()},
                      "facts": [{"entity": f.person_id, "key": f.key, "value": f.value,
                                 "source": f.source, "confidence": f.confidence} for f in g._facts.values()],
                      "relations": g._rels}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reply", required=True)
    ap.add_argument("--worker", choices=("auto", "gemini", "claude"), default="auto")
    ap.add_argument("--entity-kind", choices=("person", "animal", "object"), default="person")
    ap.add_argument("--model", default=None)
    ap.add_argument("--claude", default="claude")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    path, snaps = os.path.join(a.dir, "graph.pkl"), os.path.join(a.dir, "snapshots")
    if os.path.exists(path):
        os.remove(path)
    store = FileGraphStore(path)
    did = seed(store, snaps, entity_kind=a.entity_kind)
    base_runner = make_worker_runner(a.worker, f"file:{path}", snaps, model=a.model, claude_bin=a.claude)
    if isinstance(base_runner, GeminiWorkerRunner):
        runner = TimedGeminiRunner(f"file:{path}", snaps, model=a.model)
    else:
        runner = TimedClaudeRunner(f"file:{path}", snaps, claude_bin=a.claude, model=a.model)
    notifier = FakeNotifier()
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
