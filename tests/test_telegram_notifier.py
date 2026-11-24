"""TelegramNotifier tests. No network: a fake Transport records every request."""
from __future__ import annotations

import email.parser
import email.policy
import json
import logging
import os
from collections import defaultdict, deque

import pytest

from worldcam.messenger.base import Choice, OutgoingQuestion
from worldcam.messenger.telegram import (
    BotAPI, TelegramError, TelegramNetworkError, TelegramNotifier, cap_text, load_env_file, redact,
)

TOKEN = "123456789:AAHfakeTokenForTestsOnly_abcdefghijklmn"
OWNER = 4242
STRANGER = 999


def decode_body(body: bytes, ctype: str) -> tuple[dict, dict]:
    """-> (fields, files{name: (filename, bytes, ctype)}). JSON fields are decoded."""
    if ctype == "application/json":
        return json.loads(body), {}
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {ctype}\r\n\r\n".encode() + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fn = part.get_filename()
        data = part.get_payload(decode=True)
        if fn:
            files[name] = (fn, data, part.get_content_type())
        else:
            v = data.decode()
            try:
                v = json.loads(v)
            except ValueError:
                pass
            fields[name] = v
    return fields, files


class FakeTransport:
    """Scripted Bot API. Queue responses per method; otherwise sensible defaults."""

    def __init__(self):
        self.calls: list[tuple[str, dict, dict]] = []
        self.urls: list[str] = []
        self.script: dict[str, deque] = defaultdict(deque)
        self.next_id = 100

    def queue(self, method, *responses):
        self.script[method].extend(responses)

    def post(self, url, body, content_type, timeout):
        method = url.rsplit("/", 1)[1]
        self.urls.append(url)
        fields, files = decode_body(body, content_type)
        self.calls.append((method, fields, files))
        if self.script[method]:
            r = self.script[method].popleft()
            if isinstance(r, BaseException):
                raise r
            status, payload = r
            return status, json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        return 200, json.dumps({"ok": True, "result": self._default(method, fields)}).encode()

    def _msg(self):
        self.next_id += 1
        return {"message_id": self.next_id, "chat": {"id": OWNER, "type": "private"}, "date": 1}

    def _default(self, method, fields):
        if method in ("sendMessage", "sendPhoto"):
            return self._msg()
        if method == "sendMediaGroup":
            return [self._msg() for _ in fields["media"]]
        if method == "getUpdates":
            return []
        return True

    def methods(self):
        return [c[0] for c in self.calls]

    def last(self, method):
        return [c for c in self.calls if c[0] == method][-1]


@pytest.fixture
def snap(tmp_path):
    d = tmp_path / "snapshots"
    d.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (d / name).write_bytes(b"\xff\xd8\xff fake jpeg " + name.encode())
    return d


@pytest.fixture
def tp():
    return FakeTransport()


def make(tp, tmp_path, snap, **kw):
    sleeps = kw.pop("sleeps", [])
    return TelegramNotifier(TOKEN, OWNER, state_path=tmp_path / "captures" / "tg.json",
                            snapshot_dir=snap, transport=tp, sleep=sleeps.append, **kw)


def upd(uid, **body):
    return {"update_id": uid, **body}


def text_msg(text, chat=OWNER, reply_to=None, mid=500):
    m = {"message_id": mid, "date": 7, "chat": {"id": chat, "type": "private"},
         "from": {"id": chat}, "text": text}
    if reply_to is not None:
        m["reply_to_message"] = {"message_id": reply_to, "chat": {"id": chat}}
    return m


def callback(data, message_id, chat=OWNER, keyboard=None, text="Q?", caption=None):
    msg = {"message_id": message_id, "chat": {"id": chat, "type": "private"}, "date": 1}
    if caption is not None:
        msg["caption"] = caption
    else:
        msg["text"] = text
    if keyboard:
        msg["reply_markup"] = keyboard
    return {"id": f"cb{message_id}", "from": {"id": chat}, "message": msg, "data": data}


def ok(result):
    return 200, {"ok": True, "result": result}


# ---------------------------------------------------------- ask variants ---
def test_ask_text_only_with_buttons(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d1", "Is this Prabhat?",
                                 choices=(Choice("Yes", "yes"), Choice("No", "no"))))
    assert tp.methods() == ["sendMessage"]
    _, f, files = tp.calls[0]
    assert ref == "101" and not files
    assert f["chat_id"] == OWNER and f["text"].startswith("Is this Prabhat?")
    assert "reply to this message" in f["text"]            # free text still allowed
    assert f["reply_markup"] == {"inline_keyboard": [[
        {"text": "Yes", "callback_data": "v:yes"}, {"text": "No", "callback_data": "v:no"}]]}
    assert "parse_mode" not in f                           # plain text, no markup injection


def test_ask_free_text_uses_force_reply(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    n.ask(OutgoingQuestion("d1", "How do you know Rahul?"))
    f = tp.last("sendMessage")[1]
    assert f["reply_markup"]["force_reply"] is True
    assert f["text"] == "How do you know Rahul?"


def test_ask_no_buttons_no_free_text_has_no_markup(tp, tmp_path, snap):
    make(tp, tmp_path, snap).ask(OutgoingQuestion("d", "FYI", allow_free_text=False))
    assert "reply_markup" not in tp.last("sendMessage")[1]


def test_ask_one_image_uses_send_photo_caption(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d1", "Who is this?", image_paths=(str(snap / "a.jpg"),),
                                 choices=(Choice("New person", "new"),)))
    assert tp.methods() == ["sendPhoto"]
    _, f, files = tp.calls[0]
    assert f["chat_id"] == OWNER and f["caption"].startswith("Who is this?")
    assert f["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "v:new"
    assert f["protect_content"] is True
    assert files["photo"][0] == "a.jpg" and files["photo"][1].startswith(b"\xff\xd8")
    assert files["photo"][2] == "image/jpeg"
    assert ref == "101"


def test_ask_one_image_long_text_splits(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "x" * 2000, image_paths=(str(snap / "a.jpg"),),
                                 choices=(Choice("Y", "y"),)))
    assert tp.methods() == ["sendPhoto", "sendMessage"]
    assert "caption" not in tp.calls[0][1] and "reply_markup" not in tp.calls[0][1]
    f = tp.calls[1][1]
    assert f["reply_parameters"]["message_id"] == 101 and "reply_markup" in f
    assert ref == "102"


def test_ask_two_images_media_group_then_keyboard_message(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d1", "Are these the same person?",
                                 image_paths=(str(snap / "a.jpg"), str(snap / "b.jpg")),
                                 choices=(Choice("Same", "same"), Choice("Different", "diff"))))
    assert tp.methods() == ["sendMediaGroup", "sendMessage"]
    mg, files = tp.calls[0][1], tp.calls[0][2]
    assert mg["media"] == [{"type": "photo", "media": "attach://photo0"},
                           {"type": "photo", "media": "attach://photo1"}]
    assert set(files) == {"photo0", "photo1"} and files["photo1"][0] == "b.jpg"
    assert "reply_markup" not in mg                        # albums cannot carry keyboards
    sm = tp.calls[1][1]
    assert sm["text"].startswith("Are these the same person?")
    assert sm["reply_parameters"] == {"message_id": 101, "allow_sending_without_reply": True}
    assert [b["callback_data"] for b in sm["reply_markup"]["inline_keyboard"][0]] == ["v:same", "v:diff"]
    assert ref == "103"                                    # the keyboard message, not the album


def test_ask_rejects_more_than_two_images(tp, tmp_path, snap):
    with pytest.raises(ValueError):
        make(tp, tmp_path, snap).ask(OutgoingQuestion("d", "q", image_paths=(str(snap / "a.jpg"),) * 3))
    assert tp.calls == []


def test_long_text_is_capped(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    n.notify("y" * 5000)
    t = tp.last("sendMessage")[1]["text"]
    assert len(t) == 4096 and t.endswith("…")
    assert cap_text("\U0001F600" * 3000, 4096).encode("utf-16-le").__len__() // 2 <= 4096


# -------------------------------------------------- keyboard / id table ---
def test_long_and_non_ascii_values_use_id_table(tp, tmp_path, snap):
    long_value = "person:" + "x" * 80
    utf8_value = "é" * 31                # 62 bytes + 'v:' = 64 -> still fits
    utf8_over = "é" * 32                 # 66 bytes -> table
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Who?", choices=(Choice("Long", long_value),
                                                       Choice("Fits", utf8_value),
                                                       Choice("Over", utf8_over))))
    kb = tp.last("sendMessage")[1]["reply_markup"]
    datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
    assert all(1 <= len(d.encode()) <= 64 for d in datas)
    assert datas[1] == "v:" + utf8_value
    assert datas[0].startswith("k:") and datas[2].startswith("k:")
    state = json.loads((tmp_path / "captures" / "tg.json").read_text())
    assert state["questions"][ref]["long_values"][datas[0][2:]] == long_value

    # A fresh instance (restart) can still decode the long value.
    n2 = make(tp, tmp_path, snap)
    tp.queue("getUpdates", ok([upd(1, callback_query=callback(datas[0], int(ref), keyboard=kb))]))
    [r] = n2.poll_replies(0)
    assert r.message_ref == ref and r.choice_value == long_value and r.text is None


# ------------------------------------------------------------ matching ---
def test_button_press_answers_and_edits(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Is this Prabhat?",
                                 choices=(Choice("Yes", "yes"), Choice("No", "no"))))
    kb = tp.last("sendMessage")[1]["reply_markup"]
    tp.queue("getUpdates", ok([upd(10, callback_query=callback("v:no", int(ref), keyboard=kb,
                                                               text="Is this Prabhat?"))]))
    [r] = n.poll_replies(5)
    assert (r.message_ref, r.choice_value, r.text) == (ref, "no", None)
    gu = tp.last("getUpdates")[1]
    assert gu["timeout"] == 5 and gu["allowed_updates"] == ["message", "callback_query"]
    assert tp.methods()[-2:] == ["answerCallbackQuery", "editMessageText"]
    assert tp.last("answerCallbackQuery")[1] == {"callback_query_id": f"cb{ref}", "text": "Got it: No"}
    e = tp.last("editMessageText")[1]
    assert e["message_id"] == int(ref) and e["chat_id"] == OWNER
    assert e["text"].startswith("Is this Prabhat?") and e["text"].endswith("Answer: No")
    assert e["reply_markup"] == {"inline_keyboard": []}

    # A second tap on the same (now answered) message is not returned again.
    tp.queue("getUpdates", ok([upd(11, callback_query=callback("v:yes", int(ref), keyboard=kb))]))
    assert n.poll_replies(0) == []
    assert tp.last("answerCallbackQuery")[1]["text"] == "Already answered."


def test_button_press_on_photo_edits_caption(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Who?", image_paths=(str(snap / "a.jpg"),),
                                 choices=(Choice("Ignore", "ignore"),)))
    tp.queue("getUpdates", ok([upd(1, callback_query=callback("v:ignore", int(ref), caption="Who?"))]))
    [r] = n.poll_replies(0)
    assert r.choice_value == "ignore"
    e = tp.last("editMessageCaption")[1]
    assert e["caption"].startswith("Who?") and "Answer: ignore" in e["caption"]


def test_text_reply_to_question_matches_its_ref(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    r1 = n.ask(OutgoingQuestion("d1", "Who is A?"))
    r2 = n.ask(OutgoingQuestion("d2", "Who is B?"))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("That's my sister Asha", reply_to=int(r1)))]))
    [r] = n.poll_replies(0)
    assert (r.message_ref, r.text, r.choice_value) == (r1, "That's my sister Asha", None)
    assert r.raw["matched"] == "reply" and r2 != r1


def test_reply_to_album_photo_maps_to_keyboard_message(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Same?", image_paths=(str(snap / "a.jpg"), str(snap / "b.jpg")),
                                 choices=(Choice("Yes", "y"),)))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("yes same guy", reply_to=102))]))
    [r] = n.poll_replies(0)
    assert r.message_ref == ref
    assert tp.last("editMessageReplyMarkup")[1]["message_id"] == int(ref)   # buttons removed


def test_bare_text_goes_to_most_recent_open_question(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    r1 = n.ask(OutgoingQuestion("d1", "Q1"))
    r2 = n.ask(OutgoingQuestion("d2", "Q2"))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("answer to latest"))]))
    [a] = n.poll_replies(0)
    assert a.message_ref == r2 and a.raw["matched"] == "latest_open"
    tp.queue("getUpdates", ok([upd(2, message=text_msg("next bare text"))]))
    [b] = n.poll_replies(0)
    assert b.message_ref == r1                             # r2 is answered now
    tp.queue("getUpdates", ok([upd(3, message=text_msg("nothing open"))]))
    assert n.poll_replies(0) == []
    assert "No open question" in tp.last("sendMessage")[1]["text"]


def test_bare_command_is_ignored(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    n.ask(OutgoingQuestion("d1", "Q1"))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("/start"))]))
    assert n.poll_replies(0) == []


# --------------------------------------------------------------- filter ---
def test_non_owner_updates_are_dropped_and_logged_without_content(tp, tmp_path, snap, caplog):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Q", choices=(Choice("Yes", "yes"),)))
    secret_text = "SECRET-STRANGER-TEXT"
    spoof = callback("v:yes", int(ref))
    spoof["from"] = {"id": STRANGER}                        # stranger pressing owner's button
    group = text_msg(secret_text, chat=-100123)
    group["from"] = {"id": OWNER}                           # owner, but in a group chat
    tp.queue("getUpdates", ok([
        upd(1, message=text_msg(secret_text, chat=STRANGER, reply_to=int(ref))),
        upd(2, callback_query=callback("v:yes", int(ref), chat=STRANGER)),
        upd(3, callback_query=spoof),
        upd(4, message=group),
        upd(5, edited_message=text_msg("edit")),
    ]))
    before = len(tp.calls)
    with caplog.at_level(logging.INFO, logger="worldcam.messenger.telegram"):
        assert n.poll_replies(0) == []
    assert len(tp.calls) == before + 1                      # only getUpdates; no reply to strangers
    logs = caplog.text
    assert secret_text not in logs and str(STRANGER) not in logs
    assert logs.count("non-owner") == 4
    # The owner can still answer afterwards (question was not closed by the stranger).
    tp.queue("getUpdates", ok([upd(6, callback_query=callback("v:yes", int(ref)))]))
    [r] = n.poll_replies(0)
    assert r.choice_value == "yes"


# --------------------------------------------------------------- offset ---
def test_offset_persists_across_restarts(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    n.ask(OutgoingQuestion("d1", "Q1"))
    n.ask(OutgoingQuestion("d2", "Q2"))
    tp.queue("getUpdates", ok([upd(41, message=text_msg("a")), upd(42, message=text_msg("b"))]))
    assert len(n.poll_replies(0)) == 2
    assert "offset" not in tp.last("getUpdates")[1]         # first ever poll: no offset
    n.poll_replies(0)                                       # commits 43 before fetching
    assert tp.last("getUpdates")[1]["offset"] == 43
    state_file = tmp_path / "captures" / "tg.json"
    st = json.loads(state_file.read_text())
    assert st["offset"] == 43
    assert TOKEN not in state_file.read_text() and "AAHfake" not in state_file.read_text()
    assert oct(os.stat(state_file).st_mode & 0o777) == "0o600"
    assert not [p for p in state_file.parent.iterdir() if p.name.startswith(".tg_state.")]

    n2 = make(tp, tmp_path, snap)                           # restart
    n2.poll_replies(0)
    assert tp.last("getUpdates")[1]["offset"] == 43


def test_crash_before_next_poll_replays_batch(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Q", choices=(Choice("Yes", "yes"),)))
    batch = [upd(7, callback_query=callback("v:yes", int(ref)))]
    tp.queue("getUpdates", ok(batch))
    assert len(n.poll_replies(0)) == 1
    # Process dies before calling poll_replies again: offset was not committed.
    n2 = make(tp, tmp_path, snap)
    tp.queue("getUpdates", ok(batch))                       # Telegram redelivers
    [r] = n2.poll_replies(0)
    assert "offset" not in tp.last("getUpdates")[1]
    assert r.choice_value == "yes"                          # same update id: not "already answered"


# ------------------------------------------------------------ robustness ---
def test_429_honours_retry_after(tp, tmp_path, snap):
    sleeps = []
    n = make(tp, tmp_path, snap, sleeps=sleeps)
    tp.queue("sendMessage", (429, {"ok": False, "error_code": 429,
                                   "description": "Too Many Requests: retry after 7",
                                   "parameters": {"retry_after": 7}}))
    n.notify("hi")
    assert sleeps == [7]
    assert tp.methods() == ["sendMessage", "sendMessage"]


def test_429_beyond_limit_raises(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    tp.queue("sendMessage", (429, {"ok": False, "error_code": 429, "description": "flood",
                                   "parameters": {"retry_after": 10_000}}))
    with pytest.raises(TelegramError) as ei:
        n.notify("hi")
    assert ei.value.retry_after == 10_000


def test_network_errors_and_5xx_retry_with_backoff(tp, tmp_path, snap):
    sleeps = []
    n = make(tp, tmp_path, snap, sleeps=sleeps)
    tp.queue("sendMessage", ConnectionResetError("reset"), TimeoutError("timed out"),
             (502, b"<html>bad gateway</html>"))
    n.notify("hi")
    assert len(tp.calls) == 4 and len(sleeps) == 3
    assert 1 <= sleeps[0] < 1.3 and 2 <= sleeps[1] < 2.3 and 4 <= sleeps[2] < 4.3


def test_gives_up_after_max_retries(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap, max_retries=2)
    tp.queue("sendMessage", *[ConnectionError("down")] * 3)
    with pytest.raises(TelegramNetworkError):
        n.notify("hi")
    assert len(tp.calls) == 3


def test_4xx_is_not_retried(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    tp.queue("sendMessage", (400, {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}))
    with pytest.raises(TelegramError, match="chat not found"):
        n.notify("hi")
    assert len(tp.calls) == 1


def test_side_effect_failure_does_not_lose_reply(tp, tmp_path, snap):
    n = make(tp, tmp_path, snap)
    ref = n.ask(OutgoingQuestion("d", "Q", choices=(Choice("Yes", "yes"),)))
    tp.queue("answerCallbackQuery", (400, {"ok": False, "description": "query is too old"}))
    tp.queue("getUpdates", ok([upd(1, callback_query=callback("v:yes", int(ref)))]))
    [r] = n.poll_replies(0)
    assert r.choice_value == "yes"


# -------------------------------------------------------------- security ---
def test_token_redacted_from_exceptions_and_repr(tmp_path, snap):
    class Leaky:
        def post(self, url, body, ctype, timeout):
            raise ConnectionError(f"failed to reach {url}")
    n = TelegramNotifier(TOKEN, OWNER, state_path=tmp_path / "s.json", snapshot_dir=snap,
                         transport=Leaky(), sleep=lambda s: None, max_retries=1)
    with pytest.raises(TelegramNetworkError) as ei:
        n.notify("hi")
    secret = TOKEN.split(":")[1]
    assert secret not in str(ei.value) and secret not in repr(ei.value)
    assert ei.value.__cause__ is None and ei.value.__context__ is None   # no chained urllib error
    assert "<redacted>" in str(ei.value)
    assert secret not in repr(n) and secret not in repr(n.api) and secret not in str(vars(n.api))


def test_token_redacted_from_api_error_description(tmp_path, snap, tp):
    n = make(tp, tmp_path, snap)
    tp.queue("sendMessage", (401, {"ok": False, "error_code": 401,
                                   "description": f"Unauthorized bot{TOKEN}"}))
    with pytest.raises(TelegramError) as ei:
        n.notify("x")
    assert TOKEN.split(":")[1] not in str(ei.value)


def test_token_never_logged(tp, tmp_path, snap, caplog):
    n = make(tp, tmp_path, snap)
    tp.queue("sendMessage", ConnectionError("x"))
    with caplog.at_level(logging.DEBUG):
        n.notify("hi")
        n.ask(OutgoingQuestion("d", "Q"))
    assert TOKEN.split(":")[1] not in caplog.text


def test_invalid_token_message_does_not_echo_it(tmp_path):
    bad = "not-a-token-but-a-secret-value"
    with pytest.raises(ValueError) as ei:
        BotAPI(bad)
    assert bad not in str(ei.value)


def test_redact_helper():
    assert redact(f"https://api.telegram.org/bot{TOKEN}/getMe") == "https://api.telegram.org/bot<redacted>/getMe"


@pytest.mark.parametrize("bad", ["outside.jpg", "../outside.jpg", "link.jpg", "a.txt", "missing.jpg", ""])
def test_image_path_confinement(tp, tmp_path, snap, bad):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8 secret")
    (snap / "link.jpg").symlink_to(outside)                 # symlink escaping the snapshot dir
    (snap / "a.txt").write_text("x")
    n = make(tp, tmp_path, snap)
    path = {"outside.jpg": str(outside), "../outside.jpg": str(snap / ".." / "outside.jpg"),
            "": str(snap)}.get(bad, str(snap / bad))
    with pytest.raises(ValueError):
        n.ask(OutgoingQuestion("d", "Q", image_paths=(path,)))
    assert tp.calls == []                                   # nothing was sent


def test_image_in_subdir_is_allowed(tp, tmp_path, snap):
    (snap / "2026-11-24").mkdir()
    p = snap / "2026-11-24" / "c.png"
    p.write_bytes(b"\x89PNG fake")
    make(tp, tmp_path, snap).ask(OutgoingQuestion("d", "Q", image_paths=(str(p),)))
    assert tp.last("sendPhoto")[2]["photo"][2] == "image/png"


def test_state_shared_between_instances(tp, tmp_path, snap):
    """The MCP process asks, the engine process polls: both see the same questions."""
    asker, poller = make(tp, tmp_path, snap), make(tp, tmp_path, snap)
    ref = asker.ask(OutgoingQuestion("d", "Q"))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("bare"))]))
    [r] = poller.poll_replies(0)
    assert r.message_ref == ref


def test_load_env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# c\nexport TELEGRAM_BOT_TOKEN="abc"\nTELEGRAM_OWNER_CHAT_ID=42 # me\nEMPTY=\n')
    assert load_env_file(p) == {"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_OWNER_CHAT_ID": "42", "EMPTY": ""}


def test_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_OWNER_CHAT_ID", raising=False)
    p = tmp_path / ".env"
    p.write_text(f"TELEGRAM_BOT_TOKEN={TOKEN}\nTELEGRAM_OWNER_CHAT_ID=77\n")
    n = TelegramNotifier.from_env(p, state_path=tmp_path / "s.json", transport=FakeTransport())
    assert n.owner_chat_id == 77
