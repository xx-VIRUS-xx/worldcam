"""Telegram bot setup: find the owner's chat id and test the bot end to end.

  python -m worldcam.cli.telegram_setup            # getMe, wait for /start, save chat id
  python -m worldcam.cli.telegram_setup --test     # send a sample question + image

Reads TELEGRAM_BOT_TOKEN from the environment or .env. The token is never
printed. Stop any running worldcam poller first: Telegram allows only one
getUpdates consumer per bot (a second one gets HTTP 409).
See docs/telegram.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from ..messenger.base import Choice, OutgoingQuestion
from ..messenger.telegram import BotAPI, TelegramError, TelegramNotifier, get_setting


def _chats_from_updates(updates: list[dict]) -> dict[int, dict]:
    """Private chats that sent a message, keyed by chat id."""
    seen: dict[int, dict] = {}
    for u in updates:
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("type") != "private" or "id" not in chat:
            continue
        seen[int(chat["id"])] = {
            "name": " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x),
            "username": chat.get("username") or "",
            "start": str(msg.get("text", "")).startswith("/start"),
        }
    return seen


def wait_for_start(api: BotAPI, wait_s: float, *, out=print,
                   clock: Callable[[], float] = time.monotonic) -> dict[int, dict]:
    """Poll getUpdates until a private chat sends /start (or wait_s passes)."""
    deadline = clock() + wait_s
    offset: Optional[int] = None
    chats: dict[int, dict] = {}
    while clock() < deadline:
        t = int(max(1, min(20, deadline - clock())))
        ups = api.call("getUpdates", {"offset": offset, "timeout": t,
                                      "allowed_updates": ["message"]},
                       timeout=t + api.request_timeout) or []
        for u in ups:
            offset = int(u["update_id"]) + 1
        for cid, info in _chats_from_updates(ups).items():
            if cid not in chats:
                out(f"  seen chat id {cid}  name={info['name']!r} username=@{info['username']}"
                    f"{'  (/start)' if info['start'] else ''}")
            chats[cid] = info
        if any(c["start"] for c in chats.values()):
            break
    if offset is not None:   # confirm the consumed updates so the bot won't see them again
        api.call("getUpdates", {"offset": offset, "timeout": 0, "limit": 1})
    return chats


def write_env_var(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file (replace or append), atomically, mode 0600."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("export "):
            s = s[7:].lstrip()
        if s.split("=", 1)[0].strip() == key and "=" in s:
            if not done:
                out.append(f"{key}={value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _git_ignored(path: Path) -> Optional[bool]:
    try:
        path = path.resolve()
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=path.parent,
                           capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(r.returncode)


def _make_test_image(path: Path, label: str) -> None:
    import cv2
    import numpy as np
    img = np.zeros((360, 480, 3), np.uint8)
    img[:] = np.linspace(40, 200, 480, dtype=np.uint8)[None, :, None]
    cv2.rectangle(img, (150, 70), (330, 290), (0, 200, 255), 3)
    cv2.circle(img, (240, 160), 45, (255, 255, 255), -1)
    cv2.putText(img, label, (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"could not write {path}")


def run_setup(args, api: BotAPI, out=print, ask=input) -> int:
    me = api.call("getMe")
    user = me.get("username", "?")
    out(f"Bot OK: @{user} (id {me.get('id')})")
    out(f"Now open https://t.me/{user} in Telegram (on the owner's account) and send /start")
    out(f"Waiting up to {args.wait:.0f}s ...")
    chats = wait_for_start(api, args.wait, out=out)
    if not chats:
        out("No private chat seen. Send /start to the bot and run this again.")
        return 1
    ids = list(chats)
    starters = [c for c in ids if chats[c]["start"]]
    chosen = starters[0] if len(starters) == 1 else (ids[0] if len(ids) == 1 else None)
    if chosen is None:
        ans = ask(f"Several chats seen {ids}. Which one is the owner? ").strip()
        if not ans.lstrip("-").isdigit() or int(ans) not in chats:
            out("No valid chat id chosen; nothing written.")
            return 1
        chosen = int(ans)
    out(f"Owner chat id: {chosen}")
    env = Path(args.env)
    if _git_ignored(env) is False:
        out(f"WARNING: {env} is not git-ignored. Add it to .gitignore before storing secrets.")
    if args.yes or ask(f"Write TELEGRAM_OWNER_CHAT_ID={chosen} to {env}? [y/N] ").strip().lower() in ("y", "yes"):
        write_env_var(env, "TELEGRAM_OWNER_CHAT_ID", str(chosen))
        out(f"Saved to {env}.")
    else:
        out(f"Not written. Set TELEGRAM_OWNER_CHAT_ID={chosen} yourself.")
    return 0


def _skip_pending(api: BotAPI) -> None:
    last = api.call("getUpdates", {"offset": -1, "timeout": 0}) or []
    if last:
        api.call("getUpdates", {"offset": int(last[-1]["update_id"]) + 1, "timeout": 0, "limit": 1})


def run_test(args, token: str, out=print, transport=None, sleep=time.sleep) -> int:
    chat = get_setting("TELEGRAM_OWNER_CHAT_ID", args.env)
    if not chat:
        out("TELEGRAM_OWNER_CHAT_ID is not set; run without --test first.")
        return 2
    snap = Path(args.snapshot_dir)
    imgs = []
    for i in range(args.images):
        p = snap / f"telegram_setup_test_{i + 1}.jpg"
        _make_test_image(p, f"worldcam test {i + 1}/{args.images}")
        imgs.append(str(p))
    with tempfile.TemporaryDirectory() as td:
        n = TelegramNotifier(token, chat, state_path=Path(td) / "state.json", snapshot_dir=snap,
                             transport=transport, sleep=sleep)
        # Skip anything already queued so an old message isn't mistaken for the answer.
        _skip_pending(n.api)
        q = OutgoingQuestion(doubt_id="setup-test",
                             text="worldcam setup test: can you see the test image?",
                             image_paths=tuple(imgs),
                             choices=(Choice("Yes", "yes"), Choice("No", "no")))
        ref = n.ask(q)
        out(f"Question sent (message_ref={ref}). Tap a button or reply in Telegram ...")
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            for r in n.poll_replies(timeout_s=min(20, max(1, deadline - time.monotonic()))):
                out(f"Reply: message_ref={r.message_ref} choice_value={r.choice_value!r} "
                    f"text={r.text!r} matched_question={r.message_ref == ref}")
                n.poll_replies(timeout_s=0)       # confirm the offset with Telegram
                n.notify("worldcam setup test complete.")
                return 0
        out("No reply received before the timeout.")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env", help="path of the .env file (default: .env)")
    ap.add_argument("--wait", type=float, default=180.0, help="seconds to wait for /start or a reply")
    ap.add_argument("--yes", action="store_true", help="write the chat id without asking")
    ap.add_argument("--test", action="store_true",
                    help="send a sample question with 2 buttons and a generated image")
    ap.add_argument("--images", type=int, choices=(1, 2), default=1, help="images in the --test question")
    ap.add_argument("--snapshot-dir", default=os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots"))
    args = ap.parse_args(argv)

    token = get_setting("TELEGRAM_BOT_TOKEN", args.env)
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set (environment or .env). Create a bot with "
              "@BotFather first; see docs/telegram.md.", file=sys.stderr)
        return 2
    try:
        if args.test:
            return run_test(args, token)
        return run_setup(args, BotAPI(token))
    except (TelegramError, ValueError) as e:   # messages are already token-free
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
