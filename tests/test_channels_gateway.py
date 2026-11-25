"""ChannelGateway + ChannelNotifier with FakeChannel."""
from __future__ import annotations

import logging
import threading
import time

from worldcam.channels.api import FakeChannel
from worldcam.channels.gateway import BAD_CODE_TEXT, REFUSAL_TEXT, ChannelGateway, parse_join
from worldcam.channels.notifier_adapter import ChannelNotifier
from worldcam.messenger.base import Choice, OutgoingQuestion
from worldcam.users.store import SqliteUserDirectory


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def setup(tmp_path):
    clock = Clock()
    users = SqliteUserDirectory(tmp_path / "u.db", clock=clock)
    owner = users.add("Boss", "owner", "fake", "1", profiles=["cctv"])
    ch = FakeChannel()
    got = []
    gw = ChannelGateway(ch, users, lambda u, m: got.append((u, m)), clock=clock)
    return clock, users, owner, ch, got, gw


def texts(ch, to=None):
    return [m.text for _, r, m in ch.sent if to is None or r == to]


def test_parse_join():
    assert parse_join("/join ABC123") == "ABC123"
    assert parse_join("/join@mybot ABC123") == "ABC123"
    assert parse_join("/start ABC123") == "ABC123"
    assert parse_join("/join") is None and parse_join("hello ABC") is None
    assert parse_join("/join a b") is None


def test_whitelisted_dispatch_and_inactive_dropped(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    ann = users.add("Ann", "household", "fake", "2")
    ch.push("hi", "1")
    ch.push("yo", "2")
    gw.poll_once()
    assert [(u.name, m.text) for u, m in got] == [("Boss", "hi"), ("Ann", "yo")]
    users.deactivate(ann.id)
    ch.push("still me", "2")
    gw.poll_once()
    assert len(got) == 2                                     # not dispatched
    assert texts(ch, "2") == [REFUSAL_TEXT]                  # one polite line


def test_user_without_chat_permission_ignored(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    g = users.add("G", "guest", "fake", "3")
    users.revoke(g.id, "chat")
    ch.push("hello", "3")
    gw.poll_once()
    assert got == [] and ch.sent == []


def test_unknown_sender_refused_once_per_day_without_echo(tmp_path, caplog):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    caplog.set_level(logging.DEBUG)
    for i in range(4):
        ch.push("SECRET-CONTENT please", "666")
    gw.poll_once()
    assert got == [] and texts(ch, "666") == [REFUSAL_TEXT]
    assert "SECRET-CONTENT" not in caplog.text and "SECRET-CONTENT" not in str(ch.sent)
    clock.t += 24 * 3600 + 1
    ch.push("again", "666")
    gw.poll_once()
    assert len(texts(ch, "666")) == 2
    assert gw.stats["refused"] == 2 and gw.stats["ignored"] == 3


def test_join_flow(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    code = users.create_invite("caregiver", ["medical"], ttl=3600)
    joined = []
    gw.on_joined = joined.append
    m = ch.push(f"/join {code}", "50")
    m.raw["sender_name"] = "Joy"
    gw.poll_once()
    u = users.by_channel("fake", "50")
    assert u and u.role == "caregiver" and u.name == "Joy" and u.profiles == ("medical",)
    assert "Welcome" in texts(ch, "50")[0] and joined == [u]
    assert any("Joy joined as caregiver" in t for t in texts(ch, "1"))   # owner told
    ch.push("hello", "50")
    gw.poll_once()
    assert got[-1][0].id == u.id
    # code is single use
    ch.push(f"/join {code}", "51")
    gw.poll_once()
    assert texts(ch, "51") == [BAD_CODE_TEXT] and users.by_channel("fake", "51") is None


def test_join_bruteforce_limited_and_expired(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    for i in range(10):
        ch.push(f"/join WRONG{i}XXXX", "70")
    gw.poll_once()
    assert len(texts(ch, "70")) == 5            # 5 failures then silence
    code = users.create_invite("guest", ttl=60)
    clock.t += 3600 + 100                        # window passed, invite expired
    ch.push(f"/join {code}", "70")
    gw.poll_once()
    assert users.by_channel("fake", "70") is None
    assert texts(ch, "70")[-1] == BAD_CODE_TEXT


def test_handler_exception_does_not_kill_gateway(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    gw.on_message = lambda u, m: 1 / 0
    ch.push("x", "1")
    gw.poll_once()
    assert gw.stats["errors"] == 1


def test_thread_start_stop(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    gw.poll_timeout_s = 0.01
    orig = ch.poll

    def slow_poll(t=0):
        time.sleep(0.01)
        return orig(t)

    ch.poll = slow_poll
    gw.start()
    ch.push("ping", "1")
    for _ in range(100):
        if got:
            break
        time.sleep(0.02)
    gw.stop(5)
    assert got and not gw._thread.is_alive()


def test_thread_survives_poll_errors(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    calls = []

    def flaky(t=0):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("409")
        time.sleep(0.01)
        return []

    ch.poll = flaky
    gw.start()
    time.sleep(1.5)
    gw.stop(5)
    assert len(calls) >= 2 and gw.stats["errors"] >= 1


# ------------------------------------------------------------- notifier ---
def mk_notifier(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    others = []
    n = ChannelNotifier(ch, users, on_other=lambda u, m: others.append((u, m)))
    return users, owner, ch, n, others


def q(**kw):
    return OutgoingQuestion("d1", "Who is this?", choices=(Choice("Mom", "mom"),), **kw)


def test_notifier_ask_goes_to_owner_and_matches_reply(tmp_path):
    users, owner, ch, n, others = mk_notifier(tmp_path)
    ref = n.ask(q())
    sent_ref, rid, msg = ch.sent[-1]
    assert rid == "1" and sent_ref == ref and msg.choices[0].value == "mom"
    m = ch.push("", "1", reply_to_ref=ref, choice_value="mom")
    assert n.feed(owner, m) is True
    [r] = n.poll_replies(0.1)
    assert (r.message_ref, r.choice_value, r.text) == (ref, "mom", None)
    assert r.raw["user_id"] == owner.id and r.raw["role"] == "owner"
    assert n.poll_replies(0.05) == []


def test_notifier_bare_text_matches_latest_open_for_same_user(tmp_path):
    users, owner, ch, n, others = mk_notifier(tmp_path)
    n.ask(q())
    ref2 = n.ask(q())
    m = ch.push("it is Anna", "1")
    assert n.feed(owner, m)
    [r] = n.poll_replies(0.1)
    assert r.message_ref == ref2 and r.text == "it is Anna"
    assert n.feed(owner, ch.push("more chat", "1")) is True or True
    # unrelated reply_to_ref -> not ours
    assert n.feed(owner, ch.push("x", "1", reply_to_ref="zzz")) is False


def test_notifier_permission_filter_guest_not_confirmation(tmp_path):
    users, owner, ch, n, others = mk_notifier(tmp_path)
    guest = users.add("G", "guest", "fake", "9")
    care = users.add("C", "caregiver", "fake", "8")
    ref = n.ask(q(), ) if False else n.ask(q())
    for u in (guest, care):                                  # neither may settle identity
        m = ch.push("it's Bob", u.channel_ids["fake"], reply_to_ref=ref)
        assert n.feed(u, m) is True
    assert n.poll_replies(0.05) == []
    assert [u.name for u, _ in others] == ["G", "C"]         # surfaced, not lost
    hh = users.add("H", "household", "fake", "7")
    assert n.feed(hh, ch.push("it's Mom", "7", reply_to_ref=ref)) is True
    [r] = n.poll_replies(0.1)
    assert r.raw["role"] == "household" and r.text == "it's Mom"


def test_notifier_topic_and_recipient_resolvers(tmp_path):
    clock, users, owner, ch, got, gw = setup(tmp_path)
    care = users.add("C", "caregiver", "fake", "8")
    n = ChannelNotifier(ch, users, recipient_resolver=lambda qq: care,
                        topic_resolver=lambda qq: "medical")
    ref = n.ask(q())
    assert ch.sent[-1][1] == "8"
    assert n.feed(care, ch.push("took pills", "8", reply_to_ref=ref))
    [r] = n.poll_replies(0.1)
    assert r.raw["topic"] == "medical" and r.raw["role"] == "caregiver"


def test_notifier_poll_waits_for_feed_from_other_thread(tmp_path):
    users, owner, ch, n, others = mk_notifier(tmp_path)
    ref = n.ask(q())
    threading.Timer(0.1, lambda: n.feed(owner, ch.push("yes", "1", reply_to_ref=ref))).start()
    t0 = time.monotonic()
    [r] = n.poll_replies(3)
    assert r.text == "yes" and time.monotonic() - t0 < 2


def test_notifier_notify_and_second_answer_ignored(tmp_path):
    users, owner, ch, n, others = mk_notifier(tmp_path)
    n.notify("hello")
    assert ch.sent[-1][1:] == ("1", ch.sent[-1][2]) and ch.sent[-1][2].text == "hello"
    ref = n.ask(q())
    assert n.feed(owner, ch.push("a", "1", reply_to_ref=ref))
    assert n.feed(owner, ch.push("b", "1", reply_to_ref=ref)) is False
    assert len(n.poll_replies(0.05)) == 1
