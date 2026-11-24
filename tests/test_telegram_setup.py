"""telegram_setup CLI + UrllibTransport tests. No network."""
from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

import worldcam.messenger.telegram as tg
from worldcam.cli import telegram_setup as setup
from worldcam.messenger.telegram import BotAPI, UrllibTransport

from test_telegram_notifier import OWNER, TOKEN, FakeTransport, ok, upd, text_msg


def api(tp):
    return BotAPI(TOKEN, transport=tp, sleep=lambda s: None)


def test_write_env_var_replaces_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=x\nexport TELEGRAM_OWNER_CHAT_ID=1\n# keep\n")
    setup.write_env_var(env, "TELEGRAM_OWNER_CHAT_ID", "42")
    assert env.read_text() == "TELEGRAM_BOT_TOKEN=x\nTELEGRAM_OWNER_CHAT_ID=42\n# keep\n"
    setup.write_env_var(env, "NEW", "v")
    assert env.read_text().endswith("NEW=v\n")
    assert oct(env.stat().st_mode & 0o777) == "0o600"


def test_wait_for_start_collects_private_chats_and_confirms(tmp_path):
    tp = FakeTransport()
    group = text_msg("hi", chat=-100)
    group["chat"]["type"] = "group"
    start = text_msg("/start", chat=OWNER)
    start["chat"].update(first_name="Owner", username="owner")
    tp.queue("getUpdates", ok([upd(5, message=group)]), ok([upd(6, message=start)]))
    lines = []
    chats = setup.wait_for_start(api(tp), 60, out=lines.append)
    assert chats == {OWNER: {"name": "Owner", "username": "owner", "start": True}}
    assert tp.calls[-1][1]["offset"] == 7                 # consumed updates confirmed
    assert any(str(OWNER) in l for l in lines)


def test_run_setup_writes_chat_id_after_confirmation(tmp_path):
    tp = FakeTransport()
    tp.queue("getMe", ok({"id": 1, "is_bot": True, "username": "worldcam_test_bot"}))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("/start"))]))
    env = tmp_path / ".env"
    env.write_text(f"TELEGRAM_BOT_TOKEN={TOKEN}\n")
    args = argparse.Namespace(env=str(env), wait=30, yes=False)
    out = []
    assert setup.run_setup(args, api(tp), out=out.append, ask=lambda p: "y") == 0
    assert f"TELEGRAM_OWNER_CHAT_ID={OWNER}" in env.read_text()
    assert TOKEN.split(":")[1] not in "\n".join(out)
    assert any("t.me/worldcam_test_bot" in l for l in out)


def test_run_setup_declined_writes_nothing(tmp_path):
    tp = FakeTransport()
    tp.queue("getMe", ok({"id": 1, "username": "b"}))
    tp.queue("getUpdates", ok([upd(1, message=text_msg("/start"))]))
    env = tmp_path / ".env"
    args = argparse.Namespace(env=str(env), wait=30, yes=False)
    assert setup.run_setup(args, api(tp), out=lambda s: None, ask=lambda p: "n") == 0
    assert not env.exists()


def test_run_test_sends_question_and_prints_reply(tmp_path, monkeypatch):
    tp = FakeTransport()
    env = tmp_path / ".env"
    env.write_text(f"TELEGRAM_OWNER_CHAT_ID={OWNER}\n")
    monkeypatch.delenv("TELEGRAM_OWNER_CHAT_ID", raising=False)
    # skip-pending (offset -1) returns nothing; the first real poll gets the button press.
    tp.queue("getUpdates", ok([]))
    tp.queue("getUpdates", ok([upd(9, callback_query={
        "id": "cb", "from": {"id": OWNER}, "data": "v:yes",
        "message": {"message_id": 101, "chat": {"id": OWNER}, "caption": "q"}})]))
    args = argparse.Namespace(env=str(env), wait=10, images=1, snapshot_dir=str(tmp_path / "snap"))
    out = []
    assert setup.run_test(args, TOKEN, out=out.append, transport=tp, sleep=lambda s: None) == 0
    photo = [c for c in tp.calls if c[0] == "sendPhoto"][0]
    assert photo[2]["photo"][1][:2] == b"\xff\xd8"          # real JPEG from cv2
    kb = photo[1]["reply_markup"]["inline_keyboard"]
    assert [b["callback_data"] for row in kb for b in row] == ["v:yes", "v:no"]
    assert any("choice_value='yes'" in l and "matched_question=True" in l for l in out)
    assert tp.calls[-1][0] == "sendMessage"                # completion notice


def test_main_without_token(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert setup.main(["--env", str(tmp_path / "none.env")]) == 2
    assert "BotFather" in capsys.readouterr().err


# --------------------------------------------------- urllib transport ---
def test_urllib_transport_returns_http_errors_without_url(monkeypatch):
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"

    def fake_urlopen(req, timeout, context):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"ok":false,"error_code":401,"description":"Unauthorized"}'))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    status, body = UrllibTransport().post(url, b"{}", "application/json", 5)
    assert status == 401 and json.loads(body)["description"] == "Unauthorized"


def test_urllib_transport_network_error_hides_url(monkeypatch):
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"

    def fake_urlopen(req, timeout, context):
        raise urllib.error.URLError(OSError(f"cannot reach {url}"))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ConnectionError) as ei:
        UrllibTransport().post(url, b"{}", "application/json", 5)
    assert TOKEN not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__   # raised 'from None'
    n = BotAPI(TOKEN, sleep=lambda s: None, max_retries=0)
    with pytest.raises(tg.TelegramNetworkError) as e2:
        n.call("getMe")
    assert TOKEN.split(":")[1] not in str(e2.value)
