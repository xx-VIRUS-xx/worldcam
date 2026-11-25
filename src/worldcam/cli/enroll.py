"""Tell worldcam which face is YOU (and link your account to your person).

  worldcam-enroll                          # capture ~15 s from the camera for the only owner
  worldcam-enroll --cluster fc_ab12...     # adopt a face group the camera already saw
  worldcam-enroll --user "Name" --capture 20
  worldcam-enroll --list                   # show face groups the camera has seen

What it does:
  1. makes sure your user has a graph Person (kind "owner") and links the two
     (so "my mother" resolves relative to you and the agent may ask you about yourself);
  2. labels your face cluster to that person, so you are never asked "who is this?" again;
  3. closes open questions about that face group.
Only an owner can do this. It runs on this machine, against the local graph.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from ..sources import DEFAULT_SOURCE


# ---- pure steps (unit-tested without a camera) ----------------------------------------
def link_owner(store, users, user) -> str:
    """Ensure `user` has a graph Person of kind 'owner'; link and return its id."""
    if user.person_id and store.get_person(user.person_id) is not None:
        return user.person_id
    pid = store.upsert_person(None, name=user.name, kind="owner")
    users.set_person(user.id, pid)
    return pid


def adopt_cluster(store, person_id: str, cluster_id: str) -> list[str]:
    """Label a face cluster as `person_id`; close open questions about it. Returns closed doubt ids."""
    store.label_face_cluster(cluster_id, person_id)
    closed = []
    for d in store.list_doubts(None, 10_000):
        if cluster_id in d.about_ids and d.status in ("open", "asked", "answered"):
            store.update_doubt(d.id, status="resolved", detail=dict(
                d.detail, resolved_by="enroll", resolved_person_id=person_id, resolved_at=time.time()))
            closed.append(d.id)
    return closed


def store_faces(store, person_id: str, faces) -> set[str]:
    """Add enrolment faces (FaceObservation-like: embedding, quality, ts, crop_path) and label their clusters."""
    clusters: set[str] = set()
    for f in faces:
        _, cid = store.add_face(f.embedding, quality=f.quality, ts=f.ts, snapshot_path=f.crop_path)
        clusters.add(cid)
    for cid in clusters:
        adopt_cluster(store, person_id, cid)
    return clusters


def pick_user(users, name: Optional[str]):
    owners = [u for u in users.list() if u.role == "owner"]
    if name:
        m = [u for u in users.list() if name in (u.id, u.name)]
        if not m:
            raise SystemExit(f"no user named {name!r} (see worldcam-users list)")
        user = m[0]
    elif len(owners) == 1:
        user = owners[0]
    else:
        names = ", ".join(f"{u.name!r} ({u.id})" for u in owners)
        raise SystemExit(f"more than one owner ({names}): pass --user NAME")
    if user.role != "owner":
        raise SystemExit("only an owner can enrol their own face with this tool")
    return user


# ---- camera capture ----------------------------------------------------------------------
def capture_faces(source: str, seconds: float, min_quality: float, want: int, snapshot_dir: str):
    import numpy as np  # noqa: F401  (cv2 needs it)

    from ..capture import LatestFrameGrabber
    from ..identity import FaceAnalyzer
    from ..pipeline import BestSnapshot, Track, TrackState
    from ..sources import make_source

    analyzer = FaceAnalyzer(snapshot_dir=snapshot_dir)
    analyzer.warmup()
    got = []
    with LatestFrameGrabber(make_source(source, 1280, 720, 30)) as g:
        print(f"Look at the camera for about {int(seconds)} s (move your head a little). Starting in 3 s ...")
        time.sleep(3)
        t_end, last = time.monotonic() + seconds, 0
        while time.monotonic() < t_end and len(got) < want * 3:
            f = g.get(after_seq=last, timeout=1.0)
            if f is None:
                continue
            last = f.seq
            full = (0.0, 0.0, 1.0, 1.0)
            tr = Track(0, "person", TrackState.ACTIVE, full, f.ts, f.ts,
                       best=BestSnapshot(seq=f.seq, ts=f.ts, box=full, score=1.0, crop=f.image, crop_box=full))
            faces = [x for x in analyzer.analyze(f.image, tr, f.ts) if x.quality >= min_quality]
            if faces:
                got.append(max(faces, key=lambda x: x.quality))
                print(f"  good faces: {len(got)}", end="\r", flush=True)
            time.sleep(0.15)
    print()
    got.sort(key=lambda x: -x.quality)
    return got[: max(want * 2, want)]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="user name or id (default: the only owner)")
    ap.add_argument("--cluster", help="adopt this existing face cluster instead of capturing")
    ap.add_argument("--list", action="store_true", help="list face clusters and exit")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--capture", type=float, default=15.0, help="seconds to capture")
    ap.add_argument("--faces", type=int, default=8, help="good faces wanted")
    ap.add_argument("--min-quality", type=float, default=0.25)
    ap.add_argument("--store", default="neo4j")
    ap.add_argument("--snapshot-dir", default="captures/snapshots")
    args = ap.parse_args(argv)

    import os

    from ..ambiguity.file_store import open_store
    from ..users.store import open_directory

    store = open_store(args.store)
    if args.list:
        rows = store._read_rows("MATCH (c:FaceCluster) RETURN c.id AS id, c.n_faces AS n, c.person_id AS p "
                                "ORDER BY c.n_faces DESC") if hasattr(store, "_read_rows") else []
        for r in rows:
            print(f"{r['id']}  faces={r['n']}  person={r['p'] or '-'}")
        return 0

    users = open_directory(os.path.join("captures", "users.db"))
    user = pick_user(users, args.user)
    pid = link_owner(store, users, user)
    print(f"linked user {user.name!r} -> person {pid}")

    if args.cluster:
        closed = adopt_cluster(store, pid, args.cluster)
        print(f"cluster {args.cluster} is now you; closed {len(closed)} open question(s) about it")
        return 0

    faces = capture_faces(args.source, args.capture, args.min_quality, args.faces,
                          os.path.abspath(args.snapshot_dir))
    if len(faces) < max(3, args.faces // 2):
        print(f"only {len(faces)} usable face(s) captured. Try better light, face the camera, or run again "
              f"(lower --min-quality). Nothing was stored.", file=sys.stderr)
        return 1
    clusters = store_faces(store, pid, faces)
    print(f"enrolled {len(faces)} faces into {len(clusters)} cluster(s) for {user.name!r}. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
