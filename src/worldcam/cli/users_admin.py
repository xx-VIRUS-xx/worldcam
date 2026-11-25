"""Manage who may talk to the agent (the whitelist, roles, profiles, invites).

  python -m worldcam.cli.users_admin list
  python -m worldcam.cli.users_admin add --name Anna --role household --telegram-id 123456789
  python -m worldcam.cli.users_admin invite --role caregiver --profiles medical --ttl-hours 24
  python -m worldcam.cli.users_admin set-role USER_ID guest
  python -m worldcam.cli.users_admin profiles USER_ID cctv companion
  python -m worldcam.cli.users_admin grant USER_ID view_images
  python -m worldcam.cli.users_admin revoke USER_ID view_images
  python -m worldcam.cli.users_admin deactivate USER_ID

The database is captures/users.db (override with --db). If it is empty and
TELEGRAM_OWNER_CHAT_ID is set (environment or .env) the owner is created first.
Nothing secret is printed: invite codes are shown once, at creation.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from ..users.api import PERMISSIONS, PROFILES, ROLES, User
from ..users.store import DEFAULT_DB, InviteError, LastOwnerError, open_directory


def _fmt(u: User) -> str:
    chans = ",".join(f"{c}:{i}" for c, i in sorted(u.channel_ids.items())) or "-"
    extra = ("+" + ",".join(sorted(u.permissions_extra))) if u.permissions_extra else ""
    denied = ("-" + ",".join(sorted(u.permissions_denied))) if u.permissions_denied else ""
    flags = " ".join(x for x in (extra, denied) if x)
    return (f"{u.id:<10} {u.name[:20]:<20} {u.role:<9} {'active' if u.active else 'INACTIVE':<8} "
            f"profiles={','.join(u.profiles) or '-'} {chans} {flags}").rstrip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m worldcam.cli.users_admin",
        description="Manage the worldcam user whitelist: roles, profiles, permissions, invites.",
        epilog="Roles: " + ", ".join(ROLES) + ". Profiles: " + ", ".join(PROFILES) + ".")
    p.add_argument("--db", default=DEFAULT_DB, help=f"users database (default {DEFAULT_DB})")
    p.add_argument("--env", default=".env", help="env file for TELEGRAM_OWNER_CHAT_ID bootstrap")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="show users")
    s.add_argument("--all", action="store_true", help="include deactivated users")
    s = sub.add_parser("add", help="whitelist a user directly (you know their Telegram id)")
    s.add_argument("--name", required=True)
    s.add_argument("--role", required=True, choices=ROLES)
    s.add_argument("--telegram-id", required=True, help="numeric Telegram chat id")
    s.add_argument("--profiles", nargs="*", default=[], choices=PROFILES)
    s = sub.add_parser("invite", help="create a single-use invite code")
    s.add_argument("--role", required=True, choices=[r for r in ROLES if r != "owner"])
    s.add_argument("--profiles", nargs="*", default=[], choices=PROFILES)
    s.add_argument("--ttl-hours", type=float, default=24.0, help="validity (default 24, max 336)")
    s = sub.add_parser("set-role", help="change a user's role")
    s.add_argument("user_id")
    s.add_argument("role", choices=ROLES)
    s = sub.add_parser("profiles", help="set the profiles a user subscribes to (replaces)")
    s.add_argument("user_id")
    s.add_argument("profiles", nargs="*", choices=PROFILES)
    for name, h in (("grant", "grant a permission on top of the role"),
                    ("revoke", "revoke a permission from the role")):
        s = sub.add_parser(name, help=h)
        s.add_argument("user_id")
        s.add_argument("permission", choices=PERMISSIONS)
    s = sub.add_parser("deactivate", help="remove access (the last owner cannot be removed)")
    s.add_argument("user_id")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    d = open_directory(a.db, env_path=a.env)
    try:
        if a.cmd == "list":
            users = d.list(include_inactive=a.all)
            print("\n".join(_fmt(u) for u in users) if users else "(no users)")
        elif a.cmd == "add":
            if not a.telegram_id.lstrip("-").isdigit():
                print("error: --telegram-id must be numeric", file=sys.stderr)
                return 2
            print(_fmt(d.add(a.name, a.role, "telegram", a.telegram_id, profiles=a.profiles)))
        elif a.cmd == "invite":
            code = d.create_invite(a.role, a.profiles, ttl=a.ttl_hours * 3600)
            print(f"Invite code: {code}   (single use, valid {a.ttl_hours:g} h, role {a.role})\n"
                  "Tell the new user to open your bot in Telegram and send:\n"
                  f"    /join {code}\n"
                  "The code is not stored in clear text and cannot be shown again.")
        elif a.cmd == "set-role":
            print(_fmt(d.set_role(a.user_id, a.role)))
        elif a.cmd == "profiles":
            print(_fmt(d.set_profiles(a.user_id, a.profiles)))
        elif a.cmd == "grant":
            print(_fmt(d.grant(a.user_id, a.permission)))
        elif a.cmd == "revoke":
            print(_fmt(d.revoke(a.user_id, a.permission)))
        elif a.cmd == "deactivate":
            d.deactivate(a.user_id)
            print("deactivated")
    except KeyError:
        print("error: unknown user id", file=sys.stderr)
        return 1
    except (LastOwnerError, InviteError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
