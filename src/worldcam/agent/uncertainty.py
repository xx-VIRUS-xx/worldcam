"""UncertaintyManager: what is worth asking, and whom?

Rank open doubts and knowledge gaps by value-of-answer / attention-cost, choose the
recipient by permission (not always the owner), and decide whether an answer may settle
a doubt. An answer from someone who may not confirm that topic is kept as an
UNCONFIRMED statement; the doubt is not settled and a review doubt is created for
someone who can.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..graph.api import Doubt, GraphStore, new_id
from ..users.api import User, UserDirectory, can, can_confirm
from .profiles import ProfileSet

log = logging.getLogger(__name__)

# permission a user needs in order to be ASKED about a topic
ASK_PERMISSION = {"identity": "receive_companion", "relations": "receive_companion",
                  "medical": "receive_medical_alerts", "own_facts": "chat"}
MEDICAL_WORDS = ("medic", "pill", "dose", "health", "allerg", "diagnos", "doctor", "illness")
KIND_VALUE = {"conflict": 3.0, "low_confidence": 2.0, "unknown_face": 2.0, "metadata_gap": 1.0}
ROLE_ORDER = {"household": 1, "caregiver": 1, "owner": 2, "guest": 3}

# The agent's curiosity about its owner: asked one at a time, gently spaced.
# (fact_key, question, profiles that care (None = all), topic, needs health-tracking consent)
OWNER_INTERVIEW = (
    ("areas", "What are the rooms or areas this camera can see (for example desk, bed, door)? "
              "I'll use them to understand where things happen.", None, "own_facts", False),
    ("household", "Who lives here with you? Names and how they're related to you, if you like.",
     ("companion", "cctv"), "own_facts", False),
    ("pets", "Do you have any pets? What are their names?", ("companion", "cctv"), "own_facts", False),
    ("daily_rhythm", "Roughly what does a normal weekday look like for you: when you wake up, work, "
                     "eat and sleep?", ("companion", "medical"), "own_facts", False),
    ("regular_visitors", "Who visits regularly (family, friends, delivery, help at home)?",
     ("cctv", "companion"), "own_facts", False),
    ("interests", "What do you enjoy? I'd like to be able to chat with you about it.",
     ("companion",), "own_facts", False),
    ("privacy_preferences", "Is there anything I should never ask about, or times and places where I "
                            "should stay quiet?", None, "own_facts", False),
    ("consent_health_tracking", "Would you like me to keep track of health routines such as medication "
                                "or meals, and tell a caregiver if something seems off? You can say no.",
     ("medical",), "own_facts", False),
    ("care_routines", "Which medication or care routines should I know about, and roughly when?",
     ("medical",), "medical", True),
    ("care_contacts", "Who should I tell if something seems wrong, and how?", ("medical",), "medical", True),
)


@dataclass
class Gap:
    key: str                       # stable id for dedupe: "unlabeled:p_1", "stale:p_1:works_at"
    kind: str                      # unlabeled_person | missing_relation | stale_fact | ambiguous_routine
    topic: str                     # identity | relations | medical | own_facts
    subject_id: str
    text: str                      # question hint for the worker
    value: float = 1.0
    snapshot_paths: tuple = ()
    profile: Optional[str] = None
    priority: int = 1
    extra: dict = field(default_factory=dict)     # merged into the doubt's detail (e.g. fact_key)


@dataclass
class Ranked:
    doubt: Doubt
    topic: str
    value: float
    cost: float

    @property
    def score(self) -> float:
        return self.value / self.cost


@dataclass
class AnswerVerdict:
    settled: bool
    topic: str
    confirmed_by: Optional[dict] = None
    review_doubt_id: Optional[str] = None


@dataclass
class UncertaintyConfig:
    min_sightings_unlabeled: int = 3
    min_sightings_relation: int = 5
    stale_days: float = 180.0
    gap_cooldown_days: float = 14.0
    max_open_curiosity: int = 2
    ambiguous_routine_band: tuple = (0.35, 0.7)
    self_interview_gap_s: float = 4 * 3600.0     # minimum time between questions about the owner


class UncertaintyManager:
    def __init__(self, graph: GraphStore, users: UserDirectory, *, budget=None, profiles: Optional[ProfileSet] = None,
                 routines=None, config: Optional[UncertaintyConfig] = None,
                 clock: Callable[[], float] = time.time):
        self.graph, self.users, self.budget = graph, users, budget
        self.profiles, self.routines = profiles, routines
        self.cfg = config or UncertaintyConfig()
        self.clock = clock

    # ---- topic ----------------------------------------------------------------
    def topic_of(self, d: Doubt) -> str:
        t = d.detail.get("topic")
        if t:
            return t
        text = " ".join(str(x) for x in (d.detail.get("gap_key", ""), d.question or "")).lower()
        if any(w in text for w in MEDICAL_WORDS):
            return "medical"
        if d.detail.get("about_self"):
            return "own_facts"
        if d.kind == "metadata_gap" and "relation" in text:
            return "relations"
        return "identity"

    def _subject_ids(self, d: Doubt) -> set:
        ids = set(d.about_ids)
        if d.detail.get("subject_id"):
            ids.add(d.detail["subject_id"])
        return ids

    def _is_self(self, user: User, d: Doubt) -> bool:
        return bool(user.person_id and user.person_id in self._subject_ids(d))

    # ---- value / cost ---------------------------------------------------------------
    def _sightings(self, d: Doubt) -> int:
        n = 0
        for a in d.about_ids:
            for kw in ({"cluster_id": a}, {"person_id": a}):
                try:
                    n = max(n, len(self.graph.recent_sightings(limit=50, **kw)))
                except Exception:
                    pass
        return n

    def rank(self, d: Doubt, now: Optional[float] = None) -> Ranked:
        now = self.clock() if now is None else now
        topic = self.topic_of(d)
        value = KIND_VALUE.get(d.kind, 1.0) + math.log1p(self._sightings(d))
        text = f"{d.question or ''} {d.detail.get('gap_key', '')}".lower()
        if self.profiles and any(w in text for g in self.profiles.goals for w in g.lower().split() if len(w) > 5):
            value += 1.0
        value += min(max(now - d.created, 0.0) / (7 * 86400.0), 1.0)          # staleness
        if topic == "medical":
            value += 3.0                                                       # safety
        value += float(d.detail.get("value_boost", 0.0))
        return Ranked(d, topic, value, 1.0 + 0.5 * d.rounds)

    def rank_doubts(self, doubts: Sequence[Doubt], now: Optional[float] = None) -> list[Ranked]:
        return sorted((self.rank(d, now) for d in doubts), key=lambda r: (-r.score, r.doubt.created))

    # ---- who to ask -------------------------------------------------------------------
    def candidates(self, d: Doubt, now: Optional[float] = None, *, ignore_budget: bool = False) -> list[User]:
        """Users who may confirm this doubt's topic, best first. Owner is the fallback."""
        now = self.clock() if now is None else now
        topic = self.topic_of(d)
        perm = ASK_PERMISSION.get(topic, "chat")
        profile = d.detail.get("profile")
        prio = int(d.detail.get("priority", 2 if topic == "medical" else 1))
        excluded = set(d.detail.get("asked_exclude", ()))
        pool = []
        for u in self.users.list():
            if u.id in excluded or not can(u, perm):
                continue
            if not can_confirm(u, topic, about_self=self._is_self(u, d)):
                continue
            if self.budget is not None and not ignore_budget and not self.budget.allow(
                    u.id, priority=prio, profile=profile, now=now):
                continue
            pool.append(u)

        def key(u: User):
            load = self.budget.remaining(u.id, now)["day"] if self.budget is not None else 0
            return (0 if self._is_self(u, d) else 1,
                    0 if u.id == d.detail.get("prefer_user") else 1,
                    0 if profile and profile in u.profiles else 1,
                    ROLE_ORDER.get(u.role, 9), -load, u.id)
        return sorted(pool, key=key)

    def ask_target(self, d: Doubt, now: Optional[float] = None) -> Optional[User]:
        c = self.candidates(d, now)
        return c[0] if c else None

    # ---- answers ----------------------------------------------------------------------------
    def record_answer(self, d: Doubt, user: User, text: str, now: Optional[float] = None,
                      max_chars: int = 1000) -> AnswerVerdict:
        now = self.clock() if now is None else now
        topic = self.topic_of(d)
        text = text.strip()[:max_chars]
        if can_confirm(user, topic, about_self=self._is_self(user, d)):
            confirmed = {"user_id": user.id, "role": user.role, "ts": now}
            if d.status == "answered" and d.answer:          # second message before the resolve run
                text = f"{d.answer}\n{text}"[:max_chars]
            self.graph.update_doubt(d.id, status="answered", answer=text, detail=dict(
                d.detail, answered_at=now, confirmed_by=confirmed, answered_by=user.id))
            return AnswerVerdict(True, topic, confirmed)
        return self._unconfirmed(d, user, text, topic, now)

    def _unconfirmed(self, d: Doubt, user: User, text: str, topic: str, now: float) -> AnswerVerdict:
        stmt = {"user_id": user.id, "role": user.role, "text": text, "ts": now}
        # keep it as a low-confidence inferred fact on the person, never as source="owner"
        for a in d.about_ids:
            try:
                if self.graph.get_person(a) is not None:
                    self.graph.add_fact(a, f"unconfirmed_{topic}", f"{text} (said by {user.name}, {user.role})",
                                        source="inferred", confidence=0.3, ts=now)
            except Exception as e:
                log.warning("could not store unconfirmed statement: %s", e)
        review_id = new_id("db")
        self.graph.create_doubt(Doubt(
            id=review_id, kind="conflict", about_ids=d.about_ids, snapshot_paths=d.snapshot_paths,
            question=(f"{user.name} ({user.role}) says: \"{text}\". Is that right? "
                      f"Original question: {d.question or d.kind}"),
            created=now, updated=now,
            detail={"origin": "unconfirmed_answer", "review_of": d.id, "topic": topic, "statement": stmt,
                    "profile": d.detail.get("profile"), "priority": d.detail.get("priority", 1),
                    "asked_exclude": [user.id]}))
        self.graph.update_doubt(d.id, status="dismissed", detail=dict(
            d.detail, dismissed_reason="unconfirmed answer", dismissed_at=now, review_doubt=review_id,
            unconfirmed_statements=list(d.detail.get("unconfirmed_statements", ())) + [stmt]))
        return AnswerVerdict(False, topic, None, review_id)

    # ---- knowledge gaps ----------------------------------------------------------------------
    def _existing_keys(self, now: float) -> set:
        keys = set()
        cool = self.cfg.gap_cooldown_days * 86400.0
        for d in self.graph.list_doubts(None, 10_000):
            k = d.detail.get("gap_key")
            if not k:
                continue
            if d.status in ("open", "asked", "answered") or now - max(d.updated, d.created) < cool:
                keys.add(k)
        return keys

    def knowledge_gaps(self, now: Optional[float] = None, limit: int = 20) -> list[Gap]:
        now = self.clock() if now is None else now
        taken = self._existing_keys(now)
        gaps: list[Gap] = []
        try:
            people = self.graph.list_people(1000)
        except Exception:
            people = []
        for p in people:
            if p.kind == "owner":
                g = self._owner_gap(p, now, taken)
                if g is not None:
                    gaps.append(g)
                continue
            try:
                sights = self.graph.recent_sightings(person_id=p.id, limit=50)
            except Exception:
                sights = []
            n = len(sights)
            snaps = tuple(s.snapshot_path for s in sights if s.snapshot_path)[:1]
            if p.kind == "unknown" and not p.name and n >= self.cfg.min_sightings_unlabeled:
                gaps.append(Gap(f"unlabeled:{p.id}", "unlabeled_person", "identity", p.id,
                                f"Who is this person (seen {n} times)?", 2.0 + math.log1p(n), snaps))
            facts = self.graph.facts(p.id)
            if p.name and n >= self.cfg.min_sightings_relation and not any(
                    f.key == "relation_to_owner" for f in facts) and not self.graph.relations(p.id):
                gaps.append(Gap(f"relation:{p.id}", "missing_relation", "relations", p.id,
                                f"How is {p.name} related to you?", 1.0 + math.log1p(n), snaps))
            for f in facts:
                if f.source == "owner" and now - f.ts > self.cfg.stale_days * 86400.0:
                    medical = any(w in f"{f.key} {f.value}".lower() for w in MEDICAL_WORDS)
                    gaps.append(Gap(f"stale:{p.id}:{f.key}", "stale_fact", "medical" if medical else "relations",
                                    p.id, f"Is this still true for {p.name or 'them'}: {f.key} = {f.value}?",
                                    0.8 + min((now - f.ts) / (365 * 86400.0), 1.0),
                                    priority=2 if medical else 1))
        if self.routines is not None:
            lo, hi = self.cfg.ambiguous_routine_band
            try:
                for r in self.routines.list_routines():
                    if r.source == "inferred" and lo <= r.confidence < hi and r.support >= 3:
                        medical = "medical" in r.profiles or any(w in r.description.lower() for w in MEDICAL_WORDS)
                        gaps.append(Gap(f"routine:{r.id}", "ambiguous_routine", "medical" if medical else "own_facts",
                                        r.subject_id, f"Is this a real routine: {r.description}?",
                                        1.0 + r.support / 10.0, profile=(r.profiles[0] if r.profiles else None),
                                        priority=2 if medical else 1))
            except Exception as e:
                log.warning("routine gaps failed: %s", e)
        gaps = [g for g in gaps if g.key not in taken]
        return sorted(gaps, key=lambda g: -g.value)[:limit]

    def _owner_gap(self, owner, now: float, taken: set) -> Optional[Gap]:
        """The next unanswered question of the owner interview, or None.
        One at a time: nothing while an earlier one is still open, and a pause between them.
        Only meaningful once a user is linked to this person (see worldcam-enroll)."""
        if not any(u.person_id == owner.id for u in self.users.list()):
            return None
        mine = [d for d in self.graph.list_doubts(None, 10_000) if d.detail.get("gap_kind") == "self_profile"
                and owner.id in d.about_ids]
        if any(d.status in ("open", "asked", "answered") for d in mine):
            return None
        if mine and now - max(d.created for d in mine) < self.cfg.self_interview_gap_s:
            return None
        active = set(self.profiles.names()) if self.profiles else set()
        facts = {f.key: (f.value or "").strip().lower() for f in self.graph.facts(owner.id)}
        consent = facts.get("consent_health_tracking", "").startswith(("y", "ok", "sure", "please", "of course"))
        for i, (key, question, profiles, topic, needs_consent) in enumerate(OWNER_INTERVIEW):
            if profiles and active and not (set(profiles) & active):
                continue
            if key in facts or f"self:{owner.id}:{key}" in taken:
                continue
            if needs_consent and not consent:
                continue
            return Gap(f"self:{owner.id}:{key}", "self_profile", topic, owner.id, question,
                       value=2.5 - 0.05 * i, profile=(profiles[0] if profiles and len(profiles) == 1 else None),
                       extra={"fact_key": key})
        return None

    def best_gap(self, now: Optional[float] = None) -> Optional[Gap]:
        """Most valuable gap that somebody can currently be asked about."""
        now = self.clock() if now is None else now
        for g in self.knowledge_gaps(now):
            probe = self.gap_to_doubt(g, now)
            if self.ask_target(probe, now) is not None:
                return g
        return None

    def gap_to_doubt(self, g: Gap, now: Optional[float] = None) -> Doubt:
        now = self.clock() if now is None else now
        return Doubt(id=new_id("db"), kind="metadata_gap", about_ids=(g.subject_id,),
                     snapshot_paths=tuple(g.snapshot_paths), question=g.text, created=now, updated=now,
                     detail={"origin": "curiosity", "gap_key": g.key, "gap_kind": g.kind, "topic": g.topic,
                             "priority": g.priority, "profile": g.profile, "value_boost": g.value,
                             **g.extra})

    def open_curiosity_count(self) -> int:
        return sum(1 for st in ("open", "asked", "answered") for d in self.graph.list_doubts(st, 10_000)
                   if d.detail.get("origin") == "curiosity")
