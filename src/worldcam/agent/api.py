"""Agent layer contract: profiles, triggers, attention budget, memory context, chat.

The agent is a long-running loop. It reacts to events and chat messages and, on
its own schedule (including randomly jittered times), looks around, reviews its
memory for gaps, consolidates and checks in. Everything proactive spends the same
per-user attention budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from ..brain.api import EventStore, RoutineStore
from ..channels.api import InboundMessage, OutboundMessage
from ..graph.api import GraphStore
from ..models.api import ModelRouter
from ..users.api import User, UserDirectory

TRIGGER_KINDS = ("event", "chat", "doubt", "schedule", "random")
LOOP_TASKS = ("curiosity", "look_around", "consolidate", "checkin", "alert_review")


@dataclass(frozen=True)
class Profile:
    """One application intent. Data, not code: loaded from JSON/dicts."""
    name: str                                    # "cctv" | "companion" | "medical"
    description: str
    goals: tuple[str, ...] = ()                  # what the agent tries to learn / keep an eye on
    notice_activities: tuple[str, ...] = ()      # vocabulary for scene/activity detection
    notice_labels: tuple[str, ...] = ()          # YOLO labels this profile cares about
    ask_topics: tuple[str, ...] = ()             # kinds of questions worth asking
    alert_rules: tuple[dict, ...] = ()           # {"when": {...}, "to_permission": "...", "priority": 0-3, "text": "..."}
    checkins: tuple[dict, ...] = ()              # {"every_hours": 6, "jitter": 0.3, "prompt": "...", "quiet": true}
    tone: str = "neutral"
    allowed_roles: tuple[str, ...] = ("owner",)
    retention_days: int = 30
    send_images_to_cloud: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class Trigger:
    kind: str                                    # one of TRIGGER_KINDS
    ts: float
    payload: dict = field(default_factory=dict)  # event id / doubt id / chat ref / loop task
    profile: Optional[str] = None
    priority: int = 1                            # 0 low .. 3 urgent (urgent may bypass quiet hours)


class AttentionBudget(Protocol):
    def allow(self, user_id: str, *, priority: int = 1, profile: Optional[str] = None,
              now: Optional[float] = None) -> bool:
        """May we interrupt this user now? Per-hour/day limits by role, quiet hours,
        priority 3 may bypass quiet hours but not the daily cap's hard ceiling."""
    def record(self, user_id: str, kind: str, now: Optional[float] = None) -> None: ...
    def remaining(self, user_id: str, now: Optional[float] = None) -> dict: ...


@dataclass
class MemoryContext:
    """Everything a tool or loop may read; passed to build_memory_toolset()."""
    graph: GraphStore
    events: EventStore
    routines: RoutineStore
    users: UserDirectory
    user: Optional[User] = None                  # who is asking (None = the agent itself)
    profile: Optional[str] = None
    look_now: Optional[Callable[[str], Optional[str]]] = None   # camera name -> fresh snapshot path
    clock: Callable[[], float] = field(default=lambda: __import__("time").time())
    extras: dict = field(default_factory=dict)


class ChatHandler(Protocol):
    def handle(self, user: User, msg: InboundMessage) -> Optional[OutboundMessage]:
        """None = nothing to say (e.g. the message was a reply consumed by the doubt engine)."""


class AgentLoop(Protocol):
    def start(self) -> "AgentLoop": ...
    def stop(self) -> None: ...
    def submit(self, trigger: Trigger) -> bool:
        """Bounded queue; False = dropped (queue full)."""
