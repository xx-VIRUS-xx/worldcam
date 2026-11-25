"""Telegram adapter for the Notifier protocol (owner-only bot, long polling).

Plain HTTPS to the Bot API (https://core.telegram.org/bots/api) with stdlib
``urllib``; no bot framework. Every HTTP request goes through an injectable
``Transport`` so tests run without network.

Message layout per ``OutgoingQuestion``:
  * 0 images: ``sendMessage`` (question text + keyboard).
  * 1 image:  ``sendPhoto`` with the text as caption + keyboard. If the text is
    longer than a caption may be (1024), the photo goes alone and the text
    follows in a ``sendMessage`` that carries the keyboard.
  * 2 images: ``sendMediaGroup`` (an album cannot carry a keyboard), then a
    ``sendMessage`` with the text + keyboard, sent as a reply to the album.
The returned ``message_ref`` is ``str(message_id)`` of the message that carries
the keyboard / that the owner is asked to reply to. Album photo ids are kept as
aliases, so a reply to one of the photos maps to the same ``message_ref``.

Keyboard: ``Choice``s become inline buttons. ``callback_data`` is ``"v:" + value``
when that fits in 64 bytes, else ``"k:" + <random id>`` and the value is kept in a
local id table in the state file. With no choices and ``allow_free_text`` the
message uses ``force_reply`` so the owner's text arrives as a reply to it.

Reply matching in ``poll_replies``:
  * a button press -> ``message_ref`` of the message holding the button;
    the spinner is stopped (``answerCallbackQuery``) and the message is edited
    to show the chosen answer and drop the keyboard.
  * a text that replies to a known question (or one of its album photos) ->
    that question's ``message_ref``.
  * a text with no reply (or replying to an unknown message) -> the most
    recent *open* question (highest message id not yet answered). With no open
    question the owner gets a short "reply to a question" note and nothing is
    returned. Bare ``/commands`` are ignored.
  * every update not from ``owner_chat_id`` is dropped and logged as
    ``update_id`` + type only (no content, no sender).

Delivery is at-least-once: the getUpdates offset is persisted (atomically, under
a file lock) at the start of the *next* ``poll_replies`` call, i.e. after the
caller has had the previous batch. A crash in between replays that batch.

State file (default ``captures/telegram_state.json``): offset, question message
ids, album aliases and the long-callback-value table. No token, no question text.
The file is shared safely between processes (the MCP ``ask_owner`` process and
the poller) via ``fcntl.flock`` + read-modify-write.

Security: the token lives only in the request URL. It is never logged, never in
``repr``, and is redacted from every exception message; exceptions raised here
never chain the original urllib error (which holds the URL).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import random
import re
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from .base import Choice, OutgoingQuestion, OwnerReply

log = logging.getLogger("worldcam.messenger.telegram")

API_BASE = "https://api.telegram.org"
TEXT_LIMIT = 4096            # sendMessage / editMessageText: 1-4096 characters
CAPTION_LIMIT = 1024         # sendPhoto / editMessageCaption: 0-1024 characters
CALLBACK_DATA_LIMIT = 64     # InlineKeyboardButton.callback_data: 1-64 bytes
CALLBACK_ANSWER_LIMIT = 200  # answerCallbackQuery.text: 0-200 characters
PHOTO_MAX_BYTES = 10 * 1024 * 1024   # multipart photo upload: 10 MB max
MAX_IMAGES = 2               # contract: 0-2 images per question
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_QUESTIONS_KEPT = 500
QUESTION_TTL_S = 30 * 24 * 3600

_TOKEN_RE = re.compile(r"^\d{3,}:[A-Za-z0-9_-]{30,}$")
_ANY_TOKEN_RE = re.compile(r"\d{3,}:[A-Za-z0-9_-]{30,}")


# ---------------------------------------------------------------- errors ---
class TelegramError(RuntimeError):
    """Bot API call failed. The message never contains the token."""

    def __init__(self, method: str, description: str, error_code: Optional[int] = None,
                 retry_after: Optional[int] = None):
        self.method = method
        self.description = description
        self.error_code = error_code
        self.retry_after = retry_after
        code = f" [{error_code}]" if error_code is not None else ""
        super().__init__(f"Telegram {method}{code}: {description}")


class TelegramNetworkError(TelegramError):
    """Network failure (or 5xx) that survived all retries."""


def redact(text: str, token: Optional[str] = None) -> str:
    """Remove a bot token (the given one, and anything token-shaped) from text."""
    if token:
        text = text.replace(token, "<redacted>")
    return _ANY_TOKEN_RE.sub("<redacted>", text)


class _Secret:
    """Holds the token so it never shows up in repr/str/vars dumps."""
    __slots__ = ("_v",)

    def __init__(self, v: str):
        self._v = v

    def get(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__


# ------------------------------------------------------------- transport ---
class Transport(Protocol):
    def post(self, url: str, body: bytes, content_type: str,
             timeout: float) -> tuple[int, bytes]:
        """POST and return (http_status, body). HTTP error statuses are returned,
        not raised. Network failures raise OSError (or a subclass)."""


class UrllibTransport:
    """Default HTTPS transport (stdlib urllib, certificate verification on)."""

    def __init__(self):
        import ssl
        self._ctx = ssl.create_default_context()

    def post(self, url: str, body: bytes, content_type: str,
             timeout: float) -> tuple[int, bytes]:
        import http.client
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": content_type,
                                              "User-Agent": "worldcam"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:   # carries the URL: never let it escape
            try:
                data = e.read()
            except Exception:
                data = b""
            return e.code, data
        except urllib.error.URLError as e:
            raise ConnectionError(f"URLError: {type(e.reason).__name__}") from None
        except http.client.HTTPException as e:
            raise ConnectionError(f"HTTPException: {type(e).__name__}") from None


    def get(self, url: str, timeout: float, max_bytes: int) -> tuple[int, bytes]:
        """GET (file downloads). Reads at most max_bytes + 1 so callers can detect overflow."""
        import http.client
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "worldcam"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as r:
                return r.status, r.read(max_bytes + 1)
        except urllib.error.HTTPError as e:
            return e.code, b""
        except urllib.error.URLError as e:
            raise ConnectionError(f"URLError: {type(e.reason).__name__}") from None
        except http.client.HTTPException as e:
            raise ConnectionError(f"HTTPException: {type(e).__name__}") from None


def _encode_multipart(fields: dict[str, str],
                      files: Sequence[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "worldcam-" + uuid.uuid4().hex
    out: list[bytes] = []
    for k, v in fields.items():
        out += [f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
                v.encode("utf-8"), b"\r\n"]
    for name, filename, data, ctype in files:
        out += [f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {ctype}\r\n\r\n".encode(), data, b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


# ------------------------------------------------------------- API client ---
class BotAPI:
    """Minimal Bot API client: retries, 429 handling, token redaction."""

    def __init__(self, token: str, *, transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep, max_retries: int = 4,
                 request_timeout: float = 30.0, max_retry_after: float = 300.0,
                 api_base: str = API_BASE):
        token = (token or "").strip()
        if not _TOKEN_RE.match(token):
            raise ValueError("Telegram bot token has an invalid format "
                             "(expected '<digits>:<secret>' from @BotFather)")
        self._token = _Secret(token)
        self._transport = transport or UrllibTransport()
        self._sleep = sleep
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_retry_after = max_retry_after
        self._api_base = api_base.rstrip("/")

    def __repr__(self) -> str:
        return f"BotAPI(token=<secret>, api_base={self._api_base!r})"

    def _url(self, method: str) -> str:
        return f"{self._api_base}/bot{self._token.get()}/{method}"

    def _redact(self, s: str) -> str:
        return redact(s, self._token.get())

    def call(self, method: str, params: Optional[dict] = None, *,
             files: Sequence[tuple[str, str, bytes, str]] = (),
             timeout: Optional[float] = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if files:
            fields = {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                      for k, v in params.items()}
            body, ctype = _encode_multipart(fields, files)
        else:
            body, ctype = json.dumps(params, ensure_ascii=False).encode(), "application/json"
        timeout = timeout or self.request_timeout
        attempt = 0
        while True:
            err: Optional[str] = None
            try:
                status, data = self._transport.post(self._url(method), body, ctype, timeout)
            except OSError as e:   # ConnectionError, TimeoutError, ssl errors, ...
                err = f"{type(e).__name__}: {self._redact(str(e))}"
                status, data = 0, b""
            if err is None:
                try:
                    resp = json.loads(data.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    resp = None
                if isinstance(resp, dict) and resp.get("ok") is True:
                    return resp.get("result")
                desc = self._redact(str(resp.get("description", ""))) if isinstance(resp, dict) \
                    else "non-JSON response"
                code = resp.get("error_code", status) if isinstance(resp, dict) else status
                if status == 429 or code == 429:
                    ra = ((resp or {}).get("parameters") or {}).get("retry_after")
                    ra = int(ra) if isinstance(ra, (int, float)) else 1
                    if attempt >= self.max_retries or ra > self.max_retry_after:
                        raise TelegramError(method, desc or "Too Many Requests", 429, ra)
                    log.warning("telegram %s rate-limited; retry in %ss", method, ra)
                    self._sleep(ra)
                    attempt += 1
                    continue
                if status >= 500:
                    err = f"HTTP {status}"
                else:
                    raise TelegramError(method, desc or f"HTTP {status}", code)
            if attempt >= self.max_retries:
                raise TelegramNetworkError(method, f"giving up after {attempt + 1} attempts: {err}")
            delay = min(30.0, 2.0 ** attempt) + random.uniform(0, 0.25)
            log.warning("telegram %s failed (%s); retry %d in %.1fs", method, err, attempt + 1, delay)
            self._sleep(delay)
            attempt += 1


    def download(self, file_path: str, max_bytes: int) -> bytes:
        """Download a file by Bot API ``file_path`` (from getFile). Needs a transport
        with ``get``. Raises TelegramError (token redacted) on failure or overflow."""
        getter = getattr(self._transport, "get", None)
        if getter is None:
            raise TelegramError("download", "transport cannot download files")
        if not re.fullmatch(r"[A-Za-z0-9_./-]{1,200}", file_path) or ".." in file_path:
            raise TelegramError("download", "unexpected file path")
        url = f"{self._api_base}/file/bot{self._token.get()}/{file_path}"
        try:
            status, data = getter(url, self.request_timeout, max_bytes)
        except OSError as e:
            raise TelegramNetworkError("download", f"{type(e).__name__}: {self._redact(str(e))}") from None
        if status != 200:
            raise TelegramError("download", f"HTTP {status}", status)
        if len(data) > max_bytes:
            raise TelegramError("download", "file too large")
        return data


# ---------------------------------------------------------------- helpers ---
def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def cap_text(s: str, limit: int) -> str:
    """Truncate to Telegram's limit (counted in UTF-16 units, conservatively)."""
    if _utf16_len(s) <= limit:
        return s
    out, n = [], 0
    for ch in s:
        w = 2 if ord(ch) > 0xFFFF else 1
        if n + w > limit - 1:
            break
        out.append(ch)
        n += w
    return "".join(out) + "…"


def load_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Tiny .env parser: KEY=VALUE lines, '#' comments, optional 'export ' and quotes."""
    env: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        env[k] = v
    return env


def get_setting(name: str, env_path: str | os.PathLike = ".env") -> Optional[str]:
    """Environment first, then the .env file."""
    v = os.environ.get(name)
    return v if v else load_env_file(env_path).get(name)


@dataclass
class _Pending:
    """Side effects computed under the state lock, performed after it."""
    answer_cb: list[tuple[str, str]]
    edits: list[tuple[str, int, str]]       # (kind, message_id, new_text)
    strip_keyboard: list[int]
    notes: list[str]


# --------------------------------------------------------------- notifier ---
class TelegramNotifier:
    """Notifier over the Telegram Bot API. Serves exactly one chat: the owner's."""

    ALLOWED_UPDATES = ["message", "callback_query"]

    def __init__(self, token: str, owner_chat_id: int | str,
                 state_path: str | os.PathLike = "captures/telegram_state.json", *,
                 snapshot_dir: Optional[str | os.PathLike] = None,
                 transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 4, request_timeout: float = 30.0,
                 protect_content: bool = True, clock: Callable[[], float] = time.time,
                 api_base: str = API_BASE):
        self.api = BotAPI(token, transport=transport, sleep=sleep, max_retries=max_retries,
                          request_timeout=request_timeout, api_base=api_base)
        try:
            self.owner_chat_id = int(str(owner_chat_id).strip())
        except ValueError:
            raise ValueError("owner_chat_id must be an integer chat id") from None
        self.state_path = Path(state_path)
        snap = snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR") or "captures/snapshots"
        self.snapshot_dir = Path(snap)
        self.protect_content = protect_content
        self._clock = clock
        self._tlock = threading.RLock()
        self._uncommitted_offset: Optional[int] = None

    @classmethod
    def from_env(cls, env_path: str | os.PathLike = ".env", **kw) -> "TelegramNotifier":
        token = get_setting("TELEGRAM_BOT_TOKEN", env_path)
        chat = get_setting("TELEGRAM_OWNER_CHAT_ID", env_path)
        if not token or not chat:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_CHAT_ID must be set "
                               "(environment or .env); run python -m worldcam.cli.telegram_setup")
        return cls(token, chat, **kw)

    def __repr__(self) -> str:
        return (f"TelegramNotifier(token=<secret>, owner_chat_id={self.owner_chat_id}, "
                f"state_path={str(self.state_path)!r})")

    # ---------------------------------------------------------- state ---
    @contextlib.contextmanager
    def _state(self):
        """Exclusive read-modify-write of the state file (threads + processes)."""
        with self._tlock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.state_path.with_name(self.state_path.name + ".lock")
            with open(lock_path, "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    st = self._read_state()
                    before = json.dumps(st, sort_keys=True)
                    yield st
                    if json.dumps(st, sort_keys=True) != before:
                        self._write_state(st)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def _read_state(self) -> dict:
        st: dict = {}
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            log.error("telegram state file unreadable; starting fresh")
        if not isinstance(st, dict):
            st = {}
        st.setdefault("version", 1)
        st.setdefault("offset", None)
        st.setdefault("questions", {})
        st.setdefault("aliases", {})
        return st

    def _write_state(self, st: dict) -> None:
        d = self.state_path.parent
        fd, tmp = tempfile.mkstemp(prefix=".tg_state.", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.state_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _prune(self, st: dict) -> None:
        qs = st["questions"]
        now = self._clock()
        drop = {r for r, q in qs.items() if now - q.get("ts", now) > QUESTION_TTL_S}
        keep = sorted((r for r in qs if r not in drop), key=int)
        drop |= set(keep[:-MAX_QUESTIONS_KEPT]) if len(keep) > MAX_QUESTIONS_KEPT else set()
        for r in drop:
            qs.pop(r, None)
        st["aliases"] = {a: r for a, r in st["aliases"].items() if r in qs}

    # --------------------------------------------------------- images ---
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
            data = f.read()
        ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".webp": "image/webp"}[ext]
        return os.path.basename(real), data, ctype

    # ------------------------------------------------------- keyboard ---
    @staticmethod
    def _keyboard(choices: Iterable[Choice]) -> tuple[dict, dict[str, str]]:
        """InlineKeyboardMarkup + {id: value} for values too long for callback_data."""
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
        rows = [buttons[i:i + per_row] for i in range(0, len(buttons), per_row)]
        return {"inline_keyboard": rows}, table

    # ----------------------------------------------------- Notifier API ---
    def ask(self, q: OutgoingQuestion) -> str:
        if len(q.image_paths) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per question")
        images = [self._load_image(p) for p in q.image_paths]   # validate before sending
        text = q.text.strip() or "?"
        markup, table = (None, {})
        if q.choices:
            markup, table = self._keyboard(q.choices)
            if q.allow_free_text:
                text += "\n\nTap a button, or reply to this message with your own answer."
        elif q.allow_free_text:
            markup = {"force_reply": True, "input_field_placeholder": "Your answer"}
        chat = self.owner_chat_id
        common = {"chat_id": chat, "protect_content": self.protect_content or None}
        aliases: list[int] = []
        kind = "text"

        if len(images) == 1 and _utf16_len(text) <= CAPTION_LIMIT:
            name, data, ctype = images[0]
            msg = self.api.call("sendPhoto", {**common, "caption": text, "reply_markup": markup},
                                files=[("photo", name, data, ctype)])
            kind = "caption"
        else:
            reply_to = None
            if len(images) == 1:
                name, data, ctype = images[0]
                m = self.api.call("sendPhoto", common, files=[("photo", name, data, ctype)])
                aliases.append(int(m["message_id"]))
                reply_to = aliases[0]
            elif len(images) >= 2:
                media = [{"type": "photo", "media": f"attach://photo{i}"} for i in range(len(images))]
                files = [(f"photo{i}", n, d, c) for i, (n, d, c) in enumerate(images)]
                msgs = self.api.call("sendMediaGroup", {**common, "media": media}, files=files)
                aliases += [int(m["message_id"]) for m in msgs]
                reply_to = aliases[0]
            params = {**common, "text": cap_text(text, TEXT_LIMIT), "reply_markup": markup,
                      "link_preview_options": {"is_disabled": True}}
            if reply_to is not None:
                params["reply_parameters"] = {"message_id": reply_to,
                                              "allow_sending_without_reply": True}
            msg = self.api.call("sendMessage", params)

        ref = str(int(msg["message_id"]))
        with self._state() as st:
            st["questions"][ref] = {"ts": self._clock(), "kind": kind,
                                    "has_keyboard": bool(q.choices), "closed_by": None,
                                    "long_values": table}
            for a in aliases:
                st["aliases"][str(a)] = ref
            self._prune(st)
        log.info("telegram question sent (doubt=%s, ref=%s, images=%d, buttons=%d)",
                 q.doubt_id, ref, len(images), len(q.choices))
        return ref

    def notify(self, text: str) -> None:
        self.api.call("sendMessage", {"chat_id": self.owner_chat_id,
                                      "text": cap_text(text.strip() or "?", TEXT_LIMIT),
                                      "link_preview_options": {"is_disabled": True}})

    def poll_replies(self, timeout_s: float = 25.0) -> Sequence[OwnerReply]:
        t = max(0, int(timeout_s))
        with self._state() as st:
            if self._uncommitted_offset is not None:
                st["offset"] = max(st["offset"] or 0, self._uncommitted_offset)
            offset = st["offset"]
        updates = self.api.call("getUpdates",
                                {"offset": offset, "timeout": t, "limit": 100,
                                 "allowed_updates": self.ALLOWED_UPDATES},
                                timeout=t + self.api.request_timeout) or []
        if not updates:
            return []
        replies: list[OwnerReply] = []
        todo = _Pending([], [], [], [])
        with self._state() as st:
            for u in updates:
                uid = int(u.get("update_id", 0))
                self._uncommitted_offset = max(self._uncommitted_offset or 0, uid + 1)
                r = self._handle_update(st, uid, u, todo)
                if r is not None:
                    replies.append(r)
        self._side_effects(todo)
        return replies

    # -------------------------------------------------------- updates ---
    def _is_owner_chat(self, chat: Any) -> bool:
        return isinstance(chat, dict) and chat.get("id") == self.owner_chat_id

    def _handle_update(self, st: dict, uid: int, u: dict, todo: _Pending) -> Optional[OwnerReply]:
        if "callback_query" in u:
            cq = u["callback_query"] or {}
            msg = cq.get("message") or {}
            sender = (cq.get("from") or {}).get("id")
            if sender != self.owner_chat_id or not self._is_owner_chat(msg.get("chat")):
                log.warning("dropped update %d (callback_query) from a non-owner chat", uid)
                return None
            return self._handle_callback(st, uid, cq, msg, todo)
        if "message" in u:
            msg = u["message"] or {}
            sender = (msg.get("from") or {}).get("id")
            if not self._is_owner_chat(msg.get("chat")) or (sender is not None
                                                             and sender != self.owner_chat_id):
                log.warning("dropped update %d (message) from a non-owner chat", uid)
                return None
            return self._handle_message(st, uid, msg, todo)
        kind = next((k for k in u if k != "update_id"), "unknown")
        log.info("ignored update %d of type %s", uid, kind)
        return None

    def _is_open(self, q: dict, uid: int) -> bool:
        return q.get("closed_by") in (None, uid)   # same update replayed after a crash

    def _ref_for(self, st: dict, message_id: Any) -> Optional[str]:
        key = str(message_id)
        if key in st["questions"]:
            return key
        return st["aliases"].get(key)

    def _handle_callback(self, st, uid, cq, msg, todo) -> Optional[OwnerReply]:
        data = cq.get("data") or ""
        cq_id = str(cq.get("id", ""))
        ref = self._ref_for(st, msg.get("message_id"))
        q = st["questions"].get(ref) if ref else None
        value: Optional[str] = None
        if data.startswith("v:"):
            value = data[2:]
        elif data.startswith("k:") and q is not None:
            value = q.get("long_values", {}).get(data[2:])
        if ref is None or q is None or value is None:
            log.warning("callback in update %d does not match a known question", uid)
            todo.answer_cb.append((cq_id, "This question has expired."))
            return None
        if not self._is_open(q, uid):
            todo.answer_cb.append((cq_id, "Already answered."))
            return None
        q["closed_by"] = uid
        label = value
        for row in (msg.get("reply_markup") or {}).get("inline_keyboard", []):
            for b in row:
                if b.get("callback_data") == data:
                    label = b.get("text", value)
        todo.answer_cb.append((cq_id, cap_text(f"Got it: {label}", CALLBACK_ANSWER_LIMIT)))
        kind = q.get("kind", "text")
        original = msg.get("caption" if kind == "caption" else "text") or ""
        new = f"{original}\n\n✔ Answer: {label}".strip()
        todo.edits.append((kind, int(msg["message_id"]),
                           cap_text(new, CAPTION_LIMIT if kind == "caption" else TEXT_LIMIT)))
        return OwnerReply(message_ref=ref, choice_value=value, ts=self._clock(),
                          raw={"update_id": uid, "type": "callback_query"})

    def _handle_message(self, st, uid, msg, todo) -> Optional[OwnerReply]:
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            log.info("ignored non-text owner message in update %d", uid)
            return None
        ref = None
        rt = msg.get("reply_to_message")
        if isinstance(rt, dict):
            ref = self._ref_for(st, rt.get("message_id"))
        if ref is None:
            if text.startswith("/"):
                log.info("ignored owner command in update %d", uid)
                return None
            open_refs = [r for r, q in st["questions"].items() if self._is_open(q, uid)]
            if not open_refs:
                todo.notes.append("No open question right now. To answer an older one, "
                                  "reply to that message.")
                return None
            ref = max(open_refs, key=int)
        q = st["questions"][ref]
        if self._is_open(q, uid):
            q["closed_by"] = uid
            if q.get("has_keyboard"):
                todo.strip_keyboard.append(int(ref))
        return OwnerReply(message_ref=ref, text=text, ts=float(msg.get("date", 0) or 0),
                          raw={"update_id": uid, "type": "message",
                               "matched": "reply" if isinstance(rt, dict) else "latest_open"})

    def _side_effects(self, todo: _Pending) -> None:
        """Best effort: a failure here must not lose the replies already parsed."""
        chat = self.owner_chat_id
        calls: list[tuple[str, dict]] = []
        calls += [("answerCallbackQuery", {"callback_query_id": i, "text": t}) for i, t in todo.answer_cb]
        for kind, mid, text in todo.edits:
            if kind == "caption":
                calls.append(("editMessageCaption", {"chat_id": chat, "message_id": mid,
                                                     "caption": text,
                                                     "reply_markup": {"inline_keyboard": []}}))
            else:
                calls.append(("editMessageText", {"chat_id": chat, "message_id": mid, "text": text,
                                                  "reply_markup": {"inline_keyboard": []}}))
        calls += [("editMessageReplyMarkup", {"chat_id": chat, "message_id": mid,
                                              "reply_markup": {"inline_keyboard": []}})
                  for mid in todo.strip_keyboard]
        calls += [("sendMessage", {"chat_id": chat, "text": n}) for n in todo.notes]
        for method, params in calls:
            try:
                self.api.call(method, params)
            except TelegramError as e:
                log.warning("telegram %s failed: %s", method, e)
