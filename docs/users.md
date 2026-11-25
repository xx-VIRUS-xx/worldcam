# Users, roles and channels

Who may talk to the agent, what they may do, and how new people get in. Contract:
`users/api.py` (roles, permissions, `can`, `can_confirm`) and `channels/api.py`.

## Roles and permissions

| Permission | owner | household | caregiver | guest |
|---|:-:|:-:|:-:|:-:|
| chat | x | x | x | x |
| chat_memory (ask about others/events) | x | x | | |
| answer_identity (settles who someone is) | x | x | | |
| answer_own_facts (about themselves) | x | x | x | x |
| answer_medical | x | | x | |
| receive_cctv_alerts | x | x | | |
| receive_companion | x | x | | |
| receive_medical_alerts | x | | x | |
| view_images | x | x | x | |
| manage_users, forget, configure | x | | | |

Per user the owner can `grant` extra permissions or `revoke` role permissions
(`users_admin grant|revoke`). `can_confirm(user, topic)` decides whether a reply may settle a
fact: identity/relations -> `answer_identity`, medical -> `answer_medical`,
own_facts -> only about themselves. A guest can never settle identity.

## Storage

`captures/users.db` (SQLite, WAL, file mode 0600, schema version in `meta`). Users, their channel
addresses and invites. Invite codes are stored as SHA-256 hashes only. Open it with
`open_directory()`; if the directory is empty and `TELEGRAM_OWNER_CHAT_ID` is set, the owner is
created from it (subscribed to every profile).

Owner protection (enforced in `SqliteUserDirectory`): the last active owner cannot be
deactivated or demoted, and `manage_users`/`chat` cannot be revoked from an owner
(`LastOwnerError`). Invites can never create an owner. The in-memory fake (`api.py`) does not
enforce this; production code must use the SQLite directory.

## Onboarding

1. Owner: `python -m worldcam.cli.users_admin invite --role caregiver --profiles medical --ttl-hours 24`
   prints a one-time code (shown once) and the instruction.
2. New person opens the bot in Telegram and sends `/join CODE` (also `/start CODE`, so a
   `t.me/<bot>?start=CODE` link works).
3. The gateway redeems the code (single use, expiring), whitelists the sender, sends a welcome and
   tells every user with `manage_users` who joined and in which role.
4. If you already know a Telegram id: `users_admin add --name Anna --role household --telegram-id 123`.
   Change later with `set-role`, `profiles`, `grant`, `revoke`, `deactivate`.

## Running it

```python
users = open_directory()                       # captures/users.db + owner bootstrap
channel = TelegramChannel.from_env()
notifier = ChannelNotifier(channel, users, on_other=chat.handle)   # for AmbiguityEngine
def on_message(user, msg):
    if not notifier.feed(user, msg):           # answer to one of our questions?
        chat.handle(user, msg)                 # else ordinary chat
gateway = ChannelGateway(channel, users, on_message); gateway.start()
```

* **One poller per bot token.** Telegram returns `409 Conflict` to a second `getUpdates`
  consumer. Run exactly one `ChannelGateway`; never next to a polling `TelegramNotifier` or a
  second worldcam process (a lock file `<state>.poller` catches the same-machine case). The
  adapter never polls; it is fed by the gateway.
* `ChannelNotifier.ask()` goes to the owner by default; pass a `recipient_resolver`, or an
  object with `.recipient` / `.recipient_user_id` / `.topic` attributes. A reply settles a doubt
  only from a user who `can_confirm(user, topic)`; `OwnerReply.raw` carries `user_id`, `role`,
  `topic`. Replies from users who may not confirm go to `on_other`, never to the engine.
  Callers that write facts should record `confirmed_by = raw["user_id"]`.

## Security notes

- Whitelist by channel id; the gateway is the only door. Unknown senders get one neutral line per
  24 h (nothing echoed, content never logged), then silence. Deactivated users are treated as unknown.
- `/join` failures: 5 per sender per hour, then ignored. Codes are 10 chars from a 31-symbol
  alphabet (about 50 bits), single use, max 14 days.
- Only private chats are accepted (groups, channels and spoofed senders dropped).
- Inbound photos: only for whitelisted senders, largest size under 8 MB, accepted only if the
  bytes are really JPEG/PNG, saved under a generated name in `captures/inbox/telegram` (dir 0700,
  files 0600, oldest pruned past 500). Outbound images must live under the snapshot dir
  (symlinks resolved).
- Tokens never appear in logs, `repr`, exceptions or argv; logs carry update ids and counts only.
- `protect_content` is on by default for sent messages (no forwarding/saving).

## Voice (v2)

Nothing above `Channel` knows about Telegram. A voice channel implements `send()` (text-to-speech,
honouring `OutboundMessage.modality_hint="voice"`) and `poll()` (wake word + speech-to-text), and
returns `InboundMessage(modality="voice", text=<transcript>)` with `sender_id` from speaker
identification. The same `ChannelGateway`, `ChannelNotifier` and users/permissions apply; a voice
address is just another entry in `User.channel_ids` (`{"voice": "<speaker id>"}`). Because a
microphone is more sensitive than a camera, require an explicit grant before a spoken reply may
settle `answer_identity`/`answer_medical`.
