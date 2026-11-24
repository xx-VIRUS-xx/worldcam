# Telegram messenger

worldcam asks its owner questions ("Who is this?") through a private Telegram bot.
The Mac long-polls the Bot API over HTTPS (`getUpdates`); no inbound port is opened.
Code: `src/worldcam/messenger/telegram.py` (`TelegramNotifier`), setup tool:
`src/worldcam/cli/telegram_setup.py`.

## 1. Create the bot (BotFather)
1. In Telegram, open a chat with **@BotFather** (check the blue verified tick).
2. Send `/newbot`. Pick a display name (e.g. "worldcam home") and a username that
   ends in `bot`. Prefer something not guessable (e.g. `wc_7f3k2_bot`); anyone who
   finds the username can message the bot, although worldcam ignores them.
3. BotFather replies with the token (`123456789:AA...`). Treat it like a password.
4. Lock the bot down, still in @BotFather:
   - `/setjoingroups` -> choose the bot -> **Disable**. It can't be added to groups.
   - `/setprivacy` -> **Enable** (the default). In groups it would see only commands
     and replies. With joining disabled this is a second safeguard.
   - Optional: `/setdescription`, `/setuserpic`. Don't add commands or inline mode.

## 2. Store the token
Put it in `.env` at the repo root. `.env` is git-ignored; check with
`git check-ignore -v .env`.
```
TELEGRAM_BOT_TOKEN=123456789:AA...
```
Variables set in the environment take precedence over `.env`.

## 3. Find your chat id
Stop any running worldcam process first. Telegram allows only one `getUpdates`
consumer per bot; a second one gets HTTP 409.
```
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.telegram_setup
```
The tool calls `getMe` and prints `https://t.me/<your_bot>`. Open that link **from the
owner's Telegram account** and send `/start`. The tool prints each private chat id it
sees, then asks before writing `TELEGRAM_OWNER_CHAT_ID=<id>` into `.env`
(mode 0600). Use `--yes` to skip the prompt and `--wait N` to wait longer.

## 4. Test end to end
```
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.telegram_setup --test            # 1 image
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.telegram_setup --test --images 2 # album + buttons
```
The tool generates a test image with cv2 under `captures/snapshots/`, then sends a
question with **Yes / No** buttons. Tap a button or reply with text. The tool prints
the `OwnerReply` it received (`message_ref`, `choice_value`, `text`) and sends a
"test complete" message.

## How it behaves
- **Owner only.** Updates whose chat is not `TELEGRAM_OWNER_CHAT_ID`, including button
  presses by anyone else, are dropped. The log records only the update id and its type,
  never the content or the sender.
- **Questions.** With no image the bot uses `sendMessage`; with one image, `sendPhoto`
  with the question as caption. With two images it sends a `sendMediaGroup` album and
  then a `sendMessage` carrying the buttons, as a reply to the album. Buttons are inline
  keyboards. A question with no buttons uses `force_reply`, so your answer is tied to it.
- **Answering.** Tap a button, or *reply* (swipe or long-press, then Reply) to the
  question. A plain message that isn't a reply goes to the **most recent unanswered
  question**. If none is open, the bot tells you to reply to a specific question.
  Commands such as `/start` are ignored. After a tap, the message is edited to show
  "Answer: ..." and the buttons are removed.
- **State.** `captures/telegram_state.json` holds the `getUpdates` offset, question
  message ids and a lookup table for button values longer than 64 bytes. It holds no
  token and no question text. It is written atomically (mode 0600) under a file lock,
  so the MCP `ask_owner` process and the poller can share it. Replies are delivered at
  least once: if the process dies right after a poll, that batch is delivered again.
- **Robustness.** Each request has a timeout. Network errors and 5xx are retried with
  exponential backoff. On 429 the bot waits `retry_after` seconds. Text is capped at
  4096 characters and captions at 1024.
- **Photos.** Only files inside `WORLDCAM_SNAPSHOT_DIR` (default `captures/snapshots`)
  can be sent, after symlinks are resolved. The file must be .jpg, .jpeg, .png or .webp
  and 10 MB or smaller. Questions are sent with `protect_content` on, so Telegram
  blocks forwarding and saving them in the app; screenshots are still possible.

## What data goes to Telegram
- The question text, the crop(s) being asked about (0-2 small images) and your replies.
- Nothing else: no full frames, no video, no graph data, no continuous upload.
- Bot chats are **not end-to-end encrypted**. They use Telegram's client-server
  encryption and are stored on Telegram's servers, so Telegram can read them. Your
  messages and the bot's stay in the chat history until you delete them
  (long-press -> Delete, or clear the chat history).
- The token sits only in `.env` and in the request URL. worldcam never logs it, and it
  is redacted from errors.

## Revoke or rotate the token
If the token may have leaked (a pasted log, a committed `.env`, a lost laptop):
1. @BotFather -> `/revoke` -> choose the bot. The old token stops working at once.
   (`/token` shows a new one if needed.)
2. Put the new token in `.env` and restart worldcam. The chat id stays the same.
To shut the bot down completely: @BotFather -> `/deletebot`, then delete the
`TELEGRAM_*` lines from `.env` and `captures/telegram_state.json`.

## Troubleshooting
| Symptom | Cause |
|---|---|
| `Telegram getUpdates [409]: Conflict` | Another poller (or a webhook) is running. Stop it; for a webhook call `deleteWebhook`. |
| `[401] Unauthorized` | Wrong or revoked token. |
| `[400] Bad Request: chat not found` | Wrong `TELEGRAM_OWNER_CHAT_ID`, or `/start` was never sent to the bot. |
| `[403] Forbidden: bot was blocked by the user` | Unblock the bot in Telegram. |
| `image path is outside the snapshot directory` | The crop isn't under `WORLDCAM_SNAPSHOT_DIR`. |
