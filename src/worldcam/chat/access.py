"""Role-aware visibility for chat memory tools.

One place decides what a user may see; every tool filters through it, so a
prompt-injected model cannot widen the view. Rules:

- guest: only data about themselves (their linked person_id). Nothing about others.
- caregiver (no chat_memory): only subjects/events/facts tagged `medical`, plus themselves.
- household: everything except `medical` data of others, unless granted
  (answer_medical or receive_medical_alerts).
- owner: everything.
- snapshot paths only with `view_images`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..brain.api import BrainEvent
from ..graph.api import Fact
from ..users.api import User, can

# activities that are medical even when an event forgot its profile tag
MEDICAL_ACTIVITIES = frozenset({"taking_medication", "medication", "taking_pills",
                                "fall_suspected", "fall", "fallen"})
_MEDICAL_KEY_RE = re.compile(
    r"(medic|health|pill|dose|allerg|diagnos|symptom|blood|doctor|illness|condition)", re.I)
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean(s: object, limit: int = 160) -> str:
    """Graph/event text is untrusted data: strip control chars, cap length."""
    return CTRL_RE.sub(" ", str(s)).replace("\n", " ").strip()[:limit]


def is_medical_key(key: str) -> bool:
    return bool(_MEDICAL_KEY_RE.search(key or ""))


def is_medical_event(e: BrainEvent) -> bool:
    return ("medical" in e.profiles or (e.activity or "").lower() in MEDICAL_ACTIVITIES
            or e.detail.get("medical") is True)


@dataclass(frozen=True)
class Access:
    user: User
    general: bool          # chat_memory: may ask about people/events beyond themselves
    medical: bool          # may read medical-tagged data of others
    images: bool           # view_images
    own_id: Optional[str]

    @classmethod
    def for_user(cls, user: User) -> "Access":
        ok = can(user, "chat")
        return cls(user=user, general=ok and can(user, "chat_memory"),
                   medical=ok and (can(user, "answer_medical") or can(user, "receive_medical_alerts")),
                   images=ok and can(user, "view_images"), own_id=user.person_id if ok else None)

    @property
    def scope(self) -> str:
        if self.general:
            return "full" if self.medical else "no_medical_of_others"
        if self.medical:
            return "medical_only"
        return "own_only"

    def is_own(self, subject_id: Optional[str]) -> bool:
        return bool(subject_id) and subject_id == self.own_id

    def event_visible(self, e: BrainEvent) -> bool:
        if self.is_own(e.subject_id):
            return True
        return self.medical if is_medical_event(e) else self.general

    def fact_visible(self, f: Fact) -> bool:
        if self.is_own(f.person_id):
            return True
        return self.medical if is_medical_key(f.key) else self.general

    def general_or_own(self, subject_id: Optional[str]) -> bool:
        return self.general or self.is_own(subject_id)

    def routine_visible(self, subject_id: str, profiles: tuple[str, ...]) -> bool:
        if self.is_own(subject_id):
            return True
        return self.medical if "medical" in profiles else self.general

    def snapshots(self, paths) -> list[str]:
        return [str(p) for p in paths] if self.images else []
