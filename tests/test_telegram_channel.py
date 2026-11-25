"""TelegramChannel against a fake HTTP transport (no network, no token)."""
from __future__ import annotations

import json
import logging
import os
import stat

import pytest

from worldcam.channels.api import OutboundMessage
from worldcam.channels.gateway import ChannelGateway
from worldcam.channels.telegram_channel import TelegramChannel, sniff_image
from worldcam.messenger.base import Choice
from worldcam.messenger.telegram import TelegramError
from worldcam.users.store import SqliteUserDirectory

TOKEN = "123456789:AAHfakeTokenForTestsOnly_abcdefghijklmn"
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


class FakeTransport:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.updates: list[dict] = []
        self.files: dict[str, bytes] = {"photos/a.jpg": JPEG}
        self.urls: list[str] = []
        self.n = 100
        self.fail: dict[str, tuple[int, dict]] = {}

    def post(self, url, body, ctype, timeout):
        self.urls.append(url)
        method = url.rsplit("/", 1)[1]
        params = json.loads(body) if ctype == "application/json" else {"_multipart": True}
        self.calls.append((method, params))
        if method in self.fail:
            st, js = self.fail[method]
            return st, json.dumps(js).encode()
        if method == "getUpdates":
            ups, self.updates = self.updates, []
            res = ups
        elif method == "getFile":
            res = {"file_id": params["file_id"], "file_path": "photos/a.jpg"}
        elif method == "sendMediaGroup":
            self.n += 2
            res = [{"message_id": self.n - 1}, {"message_id": self.n}]
        elif method in ("sendMessage", "sendPhoto"):
            self.n += 1
            res = {"message_id": self.n}
        else:
            res = True
        return 200, json.dumps({"ok": True, "result": res}).encode()

    def get(self, url, timeout, max_bytes):
        self.urls.append(url)
        path = url.split("/file/bot" + TOKEN + "/")[1]
        data = self.files.get(path)
        return (200, data) if data is not None else (404, b"")

    def of(self, method):
        return [p for m, p in self.calls if m == method]


def msg(uid, chat, text=None, mid=1, frm=None, ctype="private", **kw):
    m = {"message_id": mid, "date": 1700000000, "chat": {"id": chat, "type": ctype},
         "from": {"id": frm if frm is not None else chat, "first_name": "Zed"}, **kw}
    if text is not None:
        m["text"] = text
    return {"update_id": uid, "message": m}


@pytest.fixture
def env(tmp_path):
    tr = FakeTransport()
    ch = TelegramChannel(TOKEN, state_path=tmp_path / "st.json", inbox_dir=tmp_path / "inbox",
                         snapshot_dir=tmp_path / "snaps", transport=tr, sleep=lambda s: None)
    (tmp_path / "snaps").mkdir()
    return tr, ch, tmp_path


def test_poll_returns_any_sender_text_and_reply(env):
    tr, ch, _ = env
    tr.updates = [msg(1, 111, "hello"), msg(2, 222, "/join ABC", mid=5,
                                            reply_to_message={"message_id": 77})]
    out = ch.poll(0)
    assert [(m.sender_id, m.text, m.reply_to_ref, m.ref) for m in out] == [
        ("111", "hello", None, "1"), ("222", "/join ABC", "77", "5")]
    assert out[1].raw["sender_name"] == "Zed" and out[0].channel == "telegram"


def test_offset_committed_on_next_poll(env):
    tr, ch, _ = env
    tr.updates = [msg(10, 1, "a")]
    ch.poll(0)
    ch.poll(0)
    gets = tr.of("getUpdates")
    assert gets[0].get("offset") is None and gets[1]["offset"] == 11
    assert json.loads(open(ch.state_path).read())["offset"] == 11
    assert stat.S_IMODE(os.stat(ch.state_path).st_mode) == 0o600


def test_groups_bots_and_mismatched_sender_dropped(env, caplog):
    tr, ch, _ = env
    caplog.set_level(logging.DEBUG)
    tr.updates = [msg(1, -100, "grp secret", ctype="group"),
                  msg(2, 5, "spoof", frm=6),
                  {"update_id": 3, "edited_message": {"text": "edit secret"}}]
    assert ch.poll(0) == []
    assert "secret" not in caplog.text and "spoof" not in caplog.text


def test_callback_becomes_choice_with_reply_to_ref(env):
    tr, ch, _ = env
    ref = ch.send("42", OutboundMessage("Who?", choices=(Choice("Mom", "mom"),
                                                          Choice("Long", "x" * 80))))
    kb = tr.of("sendMessage")[-1]["reply_markup"]["inline_keyboard"]
    data = [b["callback_data"] for row in kb for b in row]
    assert data[0] == "v:mom" and data[1].startswith("k:")
    cb = lambda uid, d: {"update_id": uid, "callback_query": {
        "id": "c1", "from": {"id": 42}, "data": d,
        "message": {"message_id": int(ref), "chat": {"id": 42, "type": "private"},
                    "text": "Who?"}}}
    tr.updates = [cb(1, data[0]), cb(2, data[1]), cb(3, "k:deadbeef")]
    out = ch.poll(0)
    assert [(m.choice_value, m.reply_to_ref, m.sender_id) for m in out] == [
        ("mom", ref, "42"), ("x" * 80, ref, "42")]
    assert tr.of("answerCallbackQuery") and tr.of("editMessageReplyMarkup")
    # someone else's tap on a message in another chat is dropped
    tr.updates = [{"update_id": 9, "callback_query": {
        "id": "c", "from": {"id": 99}, "data": "v:mom",
        "message": {"message_id": 1, "chat": {"id": 42, "type": "private"}}}}]
    assert ch.poll(0) == []


def test_send_text_images_reply_silent(env):
    tr, ch, tmp = env
    img = tmp / "snaps" / "a.jpg"
    img.write_bytes(JPEG)
    img2 = tmp / "snaps" / "b.png"
    img2.write_bytes(PNG)
    r1 = ch.send(7, OutboundMessage("plain", reply_to_ref="55", silent=True))
    p = tr.of("sendMessage")[-1]
    assert p["chat_id"] == 7 and p["disable_notification"] is True
    assert p["reply_parameters"]["message_id"] == 55 and p["protect_content"] is True
    r2 = ch.send(7, OutboundMessage("cap", images=(str(img),)))
    assert tr.calls[-1][0] == "sendPhoto" and r2 != r1
    r3 = ch.send(7, OutboundMessage("two", images=(str(img), str(img2))))
    assert tr.calls[-2][0] == "sendMediaGroup" and tr.calls[-1][0] == "sendMessage"
    assert r3 == str(tr.n)
    with pytest.raises(ValueError):
        ch.send(7, OutboundMessage("x", images=(str(img),) * 3))
    with pytest.raises(ValueError):
        ch.send("abc", OutboundMessage("x"))


def test_send_rejects_images_outside_snapshot_dir(env):
    tr, ch, tmp = env
    evil = tmp / "secret.jpg"
    evil.write_bytes(JPEG)
    with pytest.raises(ValueError):
        ch.send(7, OutboundMessage("x", images=(str(evil),)))
    link = tmp / "snaps" / "link.jpg"
    link.symlink_to(evil)
    with pytest.raises(ValueError):
        ch.send(7, OutboundMessage("x", images=(str(link),)))
    assert tr.of("sendPhoto") == []


def test_long_text_is_chunked_buttons_on_last(env):
    tr, ch, _ = env
    ch.send(7, OutboundMessage("para\n" * 2500, choices=(Choice("ok", "ok"),)))
    sends = tr.of("sendMessage")
    assert len(sends) >= 3
    assert all(len(s["text"].encode("utf-16-le")) // 2 <= 4096 for s in sends)
    assert "reply_markup" not in sends[0] and "reply_markup" in sends[-1]


def photo_update(uid, chat, sizes=None, caption=None):
    sizes = sizes or [{"file_id": "small", "width": 10, "height": 10, "file_size": 100},
                      {"file_id": "big", "width": 100, "height": 100, "file_size": 200}]
    return msg(uid, chat, None, photo=sizes, **({"caption": caption} if caption else {}))


def test_photo_saved_confined_and_private(env):
    tr, ch, tmp = env
    tr.updates = [photo_update(1, 5, caption="look")]
    [m] = ch.poll(0)
    assert m.text == "look" and len(m.attachments) == 1
    p = m.attachments[0]
    assert os.path.realpath(p).startswith(os.path.realpath(tmp / "inbox" / "telegram") + os.sep)
    assert p.endswith(".jpg") and open(p, "rb").read() == JPEG
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert tr.of("getFile")[0]["file_id"] == "big"
    assert all(TOKEN not in str(m.raw) for m in [m])


def test_photo_rejections(env):
    tr, ch, tmp = env
    tr.files["photos/a.jpg"] = b"<html>not an image</html>"          # wrong magic
    tr.updates = [photo_update(1, 5, caption="c")]
    [m] = ch.poll(0)
    assert m.attachments == () and m.raw["attachment_rejected"] == "not a jpeg/png"
    tr.updates = [photo_update(2, 5, sizes=[{"file_id": "h", "width": 9, "height": 9,
                                              "file_size": 50 * 1024 * 1024}], caption="c")]
    [m] = ch.poll(0)
    assert m.raw["attachment_rejected"] == "too large"
    tr.files["photos/a.jpg"] = JPEG + b"0" * (9 * 1024 * 1024)       # liar about file_size
    tr.updates = [photo_update(3, 5, caption="c")]
    [m] = ch.poll(0)
    assert m.attachments == () and m.raw["attachment_rejected"] == "download failed"
    tr.updates = [photo_update(4, 5)]                                  # no caption, no photo -> dropped
    assert ch.poll(0) == []
    assert not (tmp / "inbox" / "telegram").exists() or not list((tmp / "inbox" / "telegram").iterdir())


def test_sniff_and_inbox_pruning(env):
    assert sniff_image(JPEG) == ".jpg" and sniff_image(PNG) == ".png"
    assert sniff_image(b"GIF89a") is None and sniff_image(b"RIFF....WEBP") is None
    tr, ch, tmp = env
    ch.max_inbox_files = 3
    for i in range(6):
        ch._store_inbound(JPEG, ".jpg")
    assert len(list((tmp / "inbox" / "telegram").iterdir())) == 3


def test_gateway_blocks_stranger_downloads(env):
    tr, ch, tmp = env
    users = SqliteUserDirectory(tmp / "u.db")
    users.add("Ann", "household", "telegram", "5")
    got = []
    gw = ChannelGateway(ch, users, lambda u, m: got.append(m))
    tr.updates = [photo_update(1, 5, caption="mine"), photo_update(2, 6, caption="stranger")]
    gw.poll_once(0)
    assert len(got) == 1 and got[0].attachments
    assert len(tr.of("getFile")) == 1                                   # nothing fetched for 6
    assert any(p.get("chat_id") == 6 for p in tr.of("sendMessage"))      # refusal line


def test_errors_are_redacted_and_409_surfaces(env):
    tr, ch, _ = env
    tr.fail["getUpdates"] = (409, {"ok": False, "error_code": 409,
                                   "description": f"Conflict {TOKEN}"})
    with pytest.raises(TelegramError) as e:
        ch.poll(0)
    assert e.value.error_code == 409 and TOKEN not in str(e.value)
    assert TOKEN not in repr(ch)


def test_second_poller_on_same_state_refused(env, tmp_path):
    tr, ch, tmp = env
    ch.poll(0)
    other = TelegramChannel(TOKEN, state_path=ch.state_path, transport=FakeTransport())
    with pytest.raises(RuntimeError):
        other.poll(0)
    ch.close()
    other.poll(0)


def test_notifier_still_backward_compatible():
    from worldcam.messenger.telegram import TelegramNotifier, UrllibTransport
    assert hasattr(TelegramNotifier, "ask") and hasattr(UrllibTransport, "get")
