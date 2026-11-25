"""Users, roles and permissions. The owner whitelists users and assigns roles.

A user can be reached on one or more channels (`channel_ids`), may be linked to a
graph Person (`person_id`, so "my mother" resolves relative to that user), and
subscribes to profiles (which alerts/questions they get).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, Sequence

ROLES = ("owner", "household", "caregiver", "guest")
PROFILES = ("cctv", "companion", "medical")

PERMISSIONS = (
    "chat",                    # may talk to the agent at all
    "chat_memory",             # may ask about people/events beyond themselves
    "answer_identity",         # answers settle who a person/animal/object is
    "answer_own_facts",        # answers about themselves settle facts about themselves
    "answer_medical",          # answers settle medication / health-routine facts
    "receive_cctv_alerts",
    "receive_companion",       # proactive check-ins / curiosity questions
    "receive_medical_alerts",
    "view_images",             # may be sent crops / snapshots
    "manage_users",
    "forget",                  # may hard-delete people/data
    "configure",               # profiles, cameras, zones, models
)

DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "owner": frozenset(PERMISSIONS),
    "household": frozenset({"chat", "chat_memory", "answer_identity", "answer_own_facts",
                            "receive_cctv_alerts", "receive_companion", "view_images"}),
    "caregiver": frozenset({"chat", "answer_own_facts", "answer_medical",
                            "receive_medical_alerts", "view_images"}),
    "guest": frozenset({"chat", "answer_own_facts"}),
}

# which permission settles which kind of statement (see can_confirm)
TOPIC_PERMISSION = {
    "identity": "answer_identity",    # who is this person / animal / object
    "own_facts": "answer_own_facts",  # facts about the speaker
    "medical": "answer_medical",
    "relations": "answer_identity",
}


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: str
    channel_ids: dict = field(default_factory=dict)   # {"telegram": "123456"}
    person_id: Optional[str] = None
    profiles: tuple[str, ...] = ()                    # profiles they subscribe to
    permissions_extra: frozenset = frozenset()        # owner-granted on top of the role
    permissions_denied: frozenset = frozenset()       # owner-revoked from the role
    active: bool = True
    created: float = 0.0


class UserDirectory(Protocol):
    def add(self, name: str, role: str, channel: str, sender_id: str, *,
            profiles: Sequence[str] = (), person_id: Optional[str] = None) -> User: ...
    def get(self, user_id: str) -> Optional[User]: ...
    def by_channel(self, channel: str, sender_id: str) -> Optional[User]:
        """Active users only. None = not whitelisted."""
    def list(self, include_inactive: bool = False) -> list[User]: ...
    def set_role(self, user_id: str, role: str) -> User: ...
    def set_profiles(self, user_id: str, profiles: Sequence[str]) -> User: ...
    def set_person(self, user_id: str, person_id: Optional[str]) -> User:
        """Link the user to the graph Person that is them (enables "my mother", self-facts)."""
    def grant(self, user_id: str, permission: str) -> User: ...
    def revoke(self, user_id: str, permission: str) -> User: ...
    def deactivate(self, user_id: str) -> None: ...
    def with_permission(self, permission: str, profile: Optional[str] = None) -> list[User]:
        """Active users who hold `permission` (and subscribe to `profile` if given)."""


def can(user: User, permission: str) -> bool:
    if not user.active:
        return False
    if permission in user.permissions_denied:
        return False
    return permission in DEFAULT_ROLE_PERMISSIONS.get(user.role, frozenset()) or permission in user.permissions_extra


def can_confirm(user: User, topic: str, *, about_self: bool = False) -> bool:
    """May this user's statement settle a fact of this topic?
    identity / relations: owner + household. medical: owner + caregiver.
    own_facts: anyone about themselves. A guest can never settle identity."""
    perm = TOPIC_PERMISSION.get(topic)
    if perm is None:
        return False
    if topic == "own_facts":
        return about_self and can(user, perm)
    return can(user, perm)


class InMemoryUserDirectory:
    def __init__(self):
        self._users: dict[str, User] = {}

    def add(self, name, role, channel, sender_id, *, profiles=(), person_id=None):
        if role not in ROLES:
            raise ValueError(f"bad role {role!r}")
        if self.by_channel(channel, str(sender_id)) is not None:
            raise ValueError("already whitelisted")
        u = User(uuid.uuid4().hex[:10], name, role, {channel: str(sender_id)}, person_id,
                 tuple(profiles), created=time.time())
        self._users[u.id] = u
        return u

    def get(self, user_id):
        return self._users.get(user_id)

    def by_channel(self, channel, sender_id):
        return next((u for u in self._users.values()
                     if u.active and u.channel_ids.get(channel) == str(sender_id)), None)

    def list(self, include_inactive=False):
        return [u for u in self._users.values() if include_inactive or u.active]

    def _upd(self, user_id, **kw):
        self._users[user_id] = replace(self._users[user_id], **kw)
        return self._users[user_id]

    def set_role(self, user_id, role):
        if role not in ROLES:
            raise ValueError(f"bad role {role!r}")
        return self._upd(user_id, role=role)

    def set_profiles(self, user_id, profiles):
        return self._upd(user_id, profiles=tuple(profiles))

    def set_person(self, user_id, person_id):
        return self._upd(user_id, person_id=person_id)

    def grant(self, user_id, permission):
        u = self._users[user_id]
        return self._upd(user_id, permissions_extra=u.permissions_extra | {permission},
                         permissions_denied=u.permissions_denied - {permission})

    def revoke(self, user_id, permission):
        u = self._users[user_id]
        return self._upd(user_id, permissions_denied=u.permissions_denied | {permission},
                         permissions_extra=u.permissions_extra - {permission})

    def deactivate(self, user_id):
        self._upd(user_id, active=False)

    def with_permission(self, permission, profile=None):
        return [u for u in self._users.values()
                if can(u, permission) and (profile is None or profile in u.profiles)]
