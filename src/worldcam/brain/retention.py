"""Retention + forget. Stores return snapshot paths; this module deletes the files.

`RetentionPolicy.apply` purges events older than a per-profile horizon. An event
tagged with several profiles lives as long as the LONGEST of its profiles'
horizons; events with no profile use `default_days`. Files are only removed when
no surviving event still references them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

DAY_S = 86400.0


def delete_files(paths, *, root: Optional[str] = None) -> list[str]:
    """Delete snapshot files; if `root` is given, refuse paths outside it. Returns deleted paths."""
    gone = []
    base = Path(root).resolve() if root else None
    for p in paths:
        try:
            q = Path(p).resolve()
            if base is not None and base not in q.parents:
                continue
            q.unlink()
            gone.append(str(p))
        except OSError:
            pass
    return gone


@dataclass
class RetentionPolicy:
    days_by_profile: Mapping[str, float] = field(default_factory=dict)
    default_days: float = 30.0

    def horizon_days(self, profiles) -> float:
        ds = [self.days_by_profile[p] for p in profiles if p in self.days_by_profile]
        return max(ds) if ds else self.default_days

    def apply(self, store, now: float, days_by_profile: Optional[Mapping[str, float]] = None, *,
              delete: bool = False, root: Optional[str] = None, limit: int = 10**7) -> list[str]:
        """Purge expired events. Returns snapshot paths that are no longer referenced
        (and deletes them from disk if `delete`)."""
        pol = RetentionPolicy(days_by_profile or self.days_by_profile, self.default_days)
        cutoff_max = now - min([pol.default_days, *pol.days_by_profile.values()]) * DAY_S
        victims = [e for e in store.query(until=cutoff_max, limit=limit, newest_first=False)
                   if e.ts < now - pol.horizon_days(e.profiles) * DAY_S]
        if not victims:
            return []
        paths: set[str] = set()
        victims.sort(key=lambda e: e.ts)
        if hasattr(store, "delete_event"):
            for e in victims:
                paths.update(store.delete_event(e.id))
        else:
            # EventStore has no per-event delete: only purge the leading run that is entirely expired
            vid = {e.id for e in victims}
            run_end = None
            for e in store.query(until=victims[-1].ts + 1e-6, limit=limit, newest_first=False):
                if e.id not in vid:
                    break
                run_end = e.ts
            if run_end is not None:
                paths.update(store.purge_before(run_end + 1e-6))
        if delete:
            delete_files(paths, root=root)
        return sorted(paths)


def forget_subject(store, subject_id: str, *, delete: bool = False, root: Optional[str] = None) -> list[str]:
    paths = store.forget_subject(subject_id)
    if delete:
        delete_files(paths, root=root)
    return paths
