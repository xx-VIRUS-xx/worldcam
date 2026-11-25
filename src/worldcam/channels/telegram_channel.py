"""Multi-user Telegram channel (implements ``channels.api.Channel``).

Unlike ``messenger.telegram.TelegramNotifier`` (owner-only, drops everyone else),
this channel returns messages from ANY private chat; authorization is the caller's
job (see ``ChannelGateway``). It reuses the Bot API plumbing of
``messenger.telegram`` (BotAPI: retries, 429, token redaction; Transport; cap_text).

* ``send``: text / one or two images / inline buttons to any chat id. Long text is
  split in chunks (buttons on the last). Image paths must live under ``snapshot_dir``.
* ``poll``: long-poll getUpdates. Text, photos (saved to ``inbox_dir``) and button
  taps (``choice_value`` + ``reply_to_ref`` = the message holding the keyboard).
  Only private chats are accepted; group/channel updates are dropped (type only logged).
* Inbound photos: only the largest size under ``max_photo_bytes``, only real JPEG/PNG
  (magic bytes, not the extension), saved under a generated name in ``inbox_dir``
  (mode 0600, dir 0700), oldest files pruned past ``max_inbox_files``. Download is
  skipped for senders rejected by ``attachment_filter`` (the gateway sets it to the
  whitelist so strangers can never make us write files).
* Delivery is at-least-once: the offset is committed at the start of the next poll.
* ONE poller per bot token: Telegram answers 409 Conflict to a second getUpdates
  consumer. Do not run this next to a ``TelegramNotifier`` that polls, or a second
  worldcam process. A non-blocking file lock on ``<state>.poller`` catches the
  same-machine case with a clear error.

Logs never contain message text, names or tokens: update id + kind only.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..messenger.base import Choice
from ..messenger.telegram import (
    ALLOWED_IMAGE_EXT, API_BASE, CAPTION_LIMIT, CALLBACK_ANSWER_LIMIT, CALLBACK_DATA_LIMIT,
    MAX_IMAGES, PHOTO_MAX_BYTES, TEXT_LIMIT, BotAPI, TelegramError, Transport, _utf16_len,
    cap_text, get_setting,
)
from .api import InboundMessage, OutboundMessage

log = logging.getLogger("worldcam.channels.telegram")

MAX_INBOUND_PHOTO_BYTES = 8 * 1024 * 1024
MAX_LONG_VALUE_REFS = 500
_CTYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
          ".webp": "image/webp"}


def sniff_image(data: bytes) -> Optional[str]:
    """'.jpg' / '.png' from magic bytes, else None (webp/gif/svg/html all rejected)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return None


class TelegramChannel:
    name = "telegram"
    ALLOWED_UPDATES = ["message", "callback_query"]

    def __init__(self, token: str, *,
                 state_path: str | os.PathLike = "captures/telegram_channel_state.json",
                 inbox_dir: str | os.PathLike = "captures/inbox",
                 snapshot_dir: Optional[str | os.PathLike] = None,
                 transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 4, request_timeout: float = 30.0,
                 protect_content: bool = True, clock: Callable[[], float] = time.time,
                 api_base: str = API_BASE,
                 max_photo_bytes: int = MAX_INBOUND_PHOTO_BYTES, max_inbox_files: int = 500):
        self.api = BotAPI(token, transport=transport, sleep=sleep, max_retries=max_retries,
                          request_timeout=request_timeout, api_base=api_base)
        self.state_path = Path(state_path)
        self.inbox_dir = Path(inbox_dir) / "telegram"
        snap = snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR") or "captures/snapshots"
        self.snapshot_dir = Path(snap)
        self.protect_content = protect_content
        self.max_photo_bytes = min(max_photo_bytes, PHOTO_MAX_BYTES * 2)
        self.max_inbox_files = max_inbox_files
        self.attachment_filter: Optional[Callable[[str], bool]] = None
        self._clock = clock
        self._lock = threading.RLock()
        self._uncommitted_offset: Optional[int] = None
        self._poll_lock_fd: Optional[int] = None

    @classmethod
    def from_env(cls, env_path: str | os.PathLike = ".env", **kw) -> "TelegramChannel":
        token = get_setting("TELEGRAM_BOT_TOKEN", env_path)
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN must be set (environment or .env)")
        return cls(token, **kw)

    def __repr__(self) -> str:
        return f"TelegramChannel(token=<secret>, state_path={str(self.state_path)!r})"

    def close(self) -> None:
        with self._lock:
            if self._poll_lock_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._poll_lock_fd)
                self._poll_lock_fd = None

    # ---------------------------------------------------------------- state ---
    def _read_state(self) -> dict:
        st: dict = {}
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            log.error("telegram channel state unreadable; starting fresh")
        if not isinstance(st, dict):
            st = {}
        st.setdefault("version", 1)
        st.setdefault("offset", None)
        st.setdefault("long_values", {})   # {message_ref: {key: value}}
        return st

    def _write_state(self, st: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tgc_state.", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.state_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _state(self):
        with self._lock:
            st = self._read_state()
            before = json.dumps(st, sort_keys=True)
            yield st
            lv = st["long_values"]
            if len(lv) > MAX_LONG_VALUE_REFS:
                for r in sorted(lv, key=lambda x: int(x) if x.isdigit() else 0)[:-MAX_LONG_VALUE_REFS]:
                    lv.pop(r, None)
            if json.dumps(st, sort_keys=True) != before:
                self._write_state(st)

    def _acquire_poller_lock(self) -> None:
        if self._poll_lock_fd is not None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.state_path) + ".poller", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError("another process is polling this Telegram channel "
                               "(only one getUpdates consumer per bot token)") from None
        self._poll_lock_fd = fd

    # ------------------------------------------------------------- sending ---
    def _load_image(self, path: str) -> tuple[str, bytes, str]:
        root = os.path.realpath(self.snapshot_dir)
        real = os.path.realpath(path)
        if os.path.commonpath([root, real]) != root or real == root:
            raise ValueError("image path is outside the snapshot directory")
        ext = os.path.splitext(real)[1].lower()
        if ext not in ALLOWED_IMAGE_EXT:
            raise ValueError(f"unsupported image type {ext!r}")
        if not os.path.isfile(real):
            raise ValueError("image file not found")
        size = os.path.getsize(real)
        if size == 0 or size > PHOTO_MAX_BYTES:
            raise ValueError(f"image size {size} bytes is outside 1..{PHOTO_MAX_BYTES}")
        with open(real, "rb") as f:
            return os.path.basename(real), f.read(), _CTYPE[ext]

    @staticmethod
    def _keyboard(choices: Sequence[Choice]) -> tuple[dict, dict[str, str]]:
        table: dict[str, str] = {}
        buttons = []
        for c in choices:
            data = "v:" + c.value
            if not c.value or len(data.encode("utf-8")) > CALLBACK_DATA_LIMIT:
                key = secrets.token_hex(8)
                table[key] = c.value
                data = "k:" + key
            buttons.append({"text": cap_text(c.label, 64) or "?", "callback_data": data})
        per_row = 2 if buttons and all(_utf16_len(b["text"]) <= 18 for b in buttons) else 1
        return {"inline_keyboard": [buttons[i:i + per_row]
                                    for i in range(0, len(buttons), per_row)]}, table

    @staticmethod
    def _chunks(text: str) -> list[str]:
        text = text.strip() or "?"
        out: list[str] = []
        while _utf16_len(text) > TEXT_LIMIT:
            cut = TEXT_LIMIT // 2   # chars; <= TEXT_LIMIT utf-16 units even for astral chars
            window = text[:cut]
            nl = window.rfind("\n")
            if nl > cut // 2:
                cut = nl
            out.append(text[:cut].rstrip() or text[:cut])
            text = text[cut:].lstrip("\n")
        out.append(text)
        return out

    def send(self, recipient_id: str, msg: OutboundMessage) -> str:
        try:
            chat = int(str(recipient_id).strip())
        except ValueError:
            raise ValueError("recipient_id must be an integer chat id") from None
        if len(msg.images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per message")
        images = [self._load_image(p) for p in msg.images]   # validate before sending
        markup, table = (None, {})
        if msg.choices:
            markup, table = self._keyboard(msg.choices)
        common: dict = {"chat_id": chat, "protect_content": self.protect_content or None,
                        "disable_notification": True if msg.silent else None}
        reply = None
        if msg.reply_to_ref and str(msg.reply_to_ref).isdigit():
            reply = {"message_id": int(msg.reply_to_ref), "allow_sending_without_reply": True}
        text = msg.text.strip() or ("?" if not images else "")
        ref_msg: dict

        if len(images) == 1 and _utf16_len(text) <= CAPTION_LIMIT:
            name, data, ctype = images[0]
            ref_msg = self.api.call(
                "sendPhoto", {**common, "caption": text or None, "reply_markup": markup,
                              "reply_parameters": reply},
                files=[("photo", name, data, ctype)])
        else:
            first_reply = reply
            if len(images) == 1:
                name, data, ctype = images[0]
                m = self.api.call("sendPhoto", {**common, "reply_parameters": reply},
                                  files=[("photo", name, data, ctype)])
                first_reply = {"message_id": int(m["message_id"]),
                               "allow_sending_without_reply": True}
            elif len(images) >= 2:
                media = [{"type": "photo", "media": f"attach://photo{i}"}
                         for i in range(len(images))]
                files = [(f"photo{i}", n, d, c) for i, (n, d, c) in enumerate(images)]
                ms = self.api.call("sendMediaGroup",
                                   {**common, "media": media, "reply_parameters": reply},
                                   files=files)
                first_reply = {"message_id": int(ms[0]["message_id"]),
                               "allow_sending_without_reply": True}
            parts = self._chunks(text) if text else []
            ref_msg = {}
            for i, part in enumerate(parts):
                last = i == len(parts) - 1
                ref_msg = self.api.call("sendMessage", {
                    **common, "text": part, "reply_markup": markup if last else None,
                    "link_preview_options": {"is_disabled": True},
                    "reply_parameters": first_reply if i == 0 else None})
            if not parts:
                ref_msg = m if len(images) == 1 else ms[0]
        ref = str(int(ref_msg["message_id"]))
        if table:
            with self._state() as st:
                st["long_values"][ref] = table
        return ref

    # ------------------------------------------------------------- polling ---
    def poll(self, timeout_s: float = 25.0) -> Sequence[InboundMessage]:
        self._acquire_poller_lock()
        t = max(0, int(timeout_s))
        with self._state() as st:
            if self._uncommitted_offset is not None:
                st["offset"] = max(st["offset"] or 0, self._uncommitted_offset)
            offset = st["offset"]
        updates = self.api.call("getUpdates", {"offset": offset, "timeout": t, "limit": 100,
                                               "allowed_updates": self.ALLOWED_UPDATES},
                                timeout=t + self.api.request_timeout) or []
        out: list[InboundMessage] = []
        for u in updates:
            uid = int(u.get("update_id", 0))
            self._uncommitted_offset = max(self._uncommitted_offset or 0, uid + 1)
            try:
                m = self._handle_update(uid, u)
            except TelegramError as e:
                log.warning("update %d: %s", uid, e)
                m = None
            except Exception as e:   # one bad update must not poison the batch
                log.warning("update %d failed: %s", uid, type(e).__name__)
                m = None
            if m is not None:
                out.append(m)
        return out

    @staticmethod
    def _private_chat_id(chat) -> Optional[str]:
        if isinstance(chat, dict) and chat.get("type") == "private" and isinstance(chat.get("id"), int):
            return str(chat["id"])
        return None

    def _handle_update(self, uid: int, u: dict) -> Optional[InboundMessage]:
        if "callback_query" in u:
            return self._handle_callback(uid, u["callback_query"] or {})
        if "message" in u:
            return self._handle_message(uid, u["message"] or {})
        log.info("ignored update %d of type %s", uid,
                 next((k for k in u if k != "update_id"), "unknown"))
        return None

    def _handle_message(self, uid: int, msg: dict) -> Optional[InboundMessage]:
        sender = self._private_chat_id(msg.get("chat"))
        frm = (msg.get("from") or {}).get("id")
        if sender is None or (frm is not None and str(frm) != sender) \
                or (msg.get("from") or {}).get("is_bot"):
            log.info("dropped update %d (message): not a private human chat", uid)
            return None
        text = msg.get("text") if isinstance(msg.get("text"), str) else msg.get("caption")
        text = text if isinstance(text, str) else ""
        attachments: list[str] = []
        raw: dict = {"update_id": uid, "type": "message"}
        frm_d = msg.get("from") or {}
        nm = " ".join(x for x in (frm_d.get("first_name"), frm_d.get("last_name"))
                      if isinstance(x, str))
        if nm.strip():
            raw["sender_name"] = nm.strip()[:80]   # used by /join; never logged
        if msg.get("photo"):
            allowed = self.attachment_filter is None or self.attachment_filter(sender)
            if allowed:
                path, why = self._save_photo(msg["photo"])
                if path:
                    attachments.append(path)
                else:
                    raw["attachment_rejected"] = why
            else:
                raw["attachment_rejected"] = "sender not allowed"
        rt = msg.get("reply_to_message")
        reply_ref = str(rt["message_id"]) if isinstance(rt, dict) and "message_id" in rt else None
        if not text.strip() and not attachments:
            log.info("ignored empty/unsupported message in update %d", uid)
            return None
        return InboundMessage(self.name, sender, text, float(msg.get("date", 0) or 0),
                              ref=str(msg.get("message_id")), reply_to_ref=reply_ref,
                              attachments=tuple(attachments), raw=raw)

    def _handle_callback(self, uid: int, cq: dict) -> Optional[InboundMessage]:
        msg = cq.get("message") or {}
        sender = self._private_chat_id(msg.get("chat"))
        if sender is None or str((cq.get("from") or {}).get("id")) != sender:
            log.info("dropped update %d (callback_query): not a private human chat", uid)
            return None
        data = cq.get("data") or ""
        ref = str(msg.get("message_id"))
        value = None
        if data.startswith("v:"):
            value = data[2:]
        elif data.startswith("k:"):
            with self._state() as st:
                value = st["long_values"].get(ref, {}).get(data[2:])
        cq_id = str(cq.get("id", ""))
        if value is None:
            self._best_effort("answerCallbackQuery", {"callback_query_id": cq_id,
                                                      "text": "This question has expired."})
            return None
        label = value
        for row in (msg.get("reply_markup") or {}).get("inline_keyboard", []):
            for b in row:
                if b.get("callback_data") == data:
                    label = b.get("text", value)
        self._best_effort("answerCallbackQuery", {
            "callback_query_id": cq_id,
            "text": cap_text(f"Got it: {label}", CALLBACK_ANSWER_LIMIT)})
        self._best_effort("editMessageReplyMarkup", {
            "chat_id": int(sender), "message_id": int(ref),
            "reply_markup": {"inline_keyboard": []}})
        return InboundMessage(self.name, sender, "", self._clock(), ref=f"cb{uid}",
                              reply_to_ref=ref, choice_value=value,
                              raw={"update_id": uid, "type": "callback_query"})

    def _best_effort(self, method: str, params: dict) -> None:
        try:
            self.api.call(method, params)
        except TelegramError as e:
            log.warning("telegram %s failed: %s", method, e)

    # -------------------------------------------------------------- photos ---
    def _save_photo(self, sizes) -> tuple[Optional[str], str]:
        """Pick the largest size within the cap, download, verify magic, save. -> (path, why)"""
        cands = [s for s in sizes if isinstance(s, dict) and isinstance(s.get("file_id"), str)]
        ok = [s for s in cands if (s.get("file_size") or 0) <= self.max_photo_bytes]
        if not ok:
            return None, "too large"
        best = max(ok, key=lambda s: (s.get("width", 0) or 0) * (s.get("height", 0) or 0))
        try:
            info = self.api.call("getFile", {"file_id": best["file_id"]}) or {}
            fpath = info.get("file_path")
            if not isinstance(fpath, str):
                return None, "no file path"
            data = self.api.download(fpath, self.max_photo_bytes)
        except TelegramError as e:
            log.warning("photo download failed: %s", e)
            return None, "download failed"
        ext = sniff_image(data)
        if ext is None:
            return None, "not a jpeg/png"
        return self._store_inbound(data, ext), ""

    def _store_inbound(self, data: bytes, ext: str) -> str:
        self.inbox_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.inbox_dir, 0o700)
        root = os.path.realpath(self.inbox_dir)
        target = os.path.join(root, f"{int(self._clock())}_{uuid.uuid4().hex[:12]}{ext}")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        files = sorted((p for p in Path(root).iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime)
        for old in files[:-self.max_inbox_files] if len(files) > self.max_inbox_files else []:
            with contextlib.suppress(OSError):
                old.unlink()
        return target
