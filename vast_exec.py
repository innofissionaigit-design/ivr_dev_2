#!/usr/bin/env python3
"""Run a shell command on the Vast instance from this laptop.

Reads VAST_BASE / VAST_TOKEN from the environment, else from vast.env beside
this file.

    python vast_exec.py 'nvidia-smi'
    python vast_exec.py - < some_script.sh

HOW IT WORKS

The contents API moves files but cannot execute. Jupyter's /api/terminals can:
POST creates a pty, then wss://.../terminals/websocket/<name> carries
["stdin", txt] in and ["stdout", txt] out. The command is bracketed with two
random sentinels so we can tell when it finished (and with what status) instead
of guessing at a timeout, and so the pty's own echo and prompt noise can be cut
away from the command's real output.
"""
import json
import os
import re
import ssl
import sys
import uuid

import requests
import urllib3
import websocket

urllib3.disable_warnings()

# The pod's output is UTF-8 (this project's logs carry Bengali), while a
# Windows console defaults to cp1252 and raises UnicodeEncodeError on the
# first non-latin byte -- losing the whole command's output over a print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _load_env_file():
    """Fall back to vast.env beside this script when nothing is exported."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vast.env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env_file()

BASE = os.environ["VAST_BASE"].rstrip("/")
TOKEN = os.environ["VAST_TOKEN"]
WS = "wss://" + BASE.split("://", 1)[1]

S = requests.Session()
S.verify = False
S.headers["Authorization"] = "token " + TOKEN

ANSI = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"    # CSI: colours, cursor moves, bracketed paste
    r"|\x1b\][^\x07]*\x07"       # OSC: window title
    r"|\x1b[=>]"                 # keypad mode
    r"|\r"                       # bare CR from the pty
)


def clean(raw, start, sent):
    """Keep only what the command itself printed, between the two sentinels."""
    txt = ANSI.sub("", raw)
    if txt.count(start) >= 2:
        txt = txt.split(start, 2)[2]
    if sent in txt:
        txt = txt.split(sent, 1)[0]
    # Bracketed-paste mode can split the echoed command so the START marker is
    # seen once rather than twice, leaving it in after the split above -- drop
    # any line still carrying a sentinel instead of relying on the count.
    keep = [ln for ln in txt.split("\n")
            if not ln.startswith(("printf ", "> "))
            and "root@C." not in ln
            and start not in ln and sent not in ln]
    return "\n".join(keep).strip("\n")


def run(cmd, timeout=300, quiet=False):
    r = S.post(BASE + "/api/terminals", timeout=60)
    r.raise_for_status()
    name = r.json()["name"]
    tag = uuid.uuid4().hex[:12]
    start, sent = "___GO_" + tag + "___", "___DONE_" + tag + "___"
    ws = websocket.create_connection(
        "%s/terminals/websocket/%s" % (WS, name),
        header=["Authorization: token " + TOKEN],
        sslopt={"cert_reqs": ssl.CERT_NONE},
        timeout=timeout,
    )
    try:
        ws.send(json.dumps(["stdin",
                            "printf '%%s\\n' %s\n%s\nprintf '%%s %%s\\n' %s \"$?\"\n"
                            % (start, cmd, sent)]))
        buf, rc = [], None
        while True:
            try:
                msg = ws.recv()
            except Exception:
                break
            if not msg:
                continue
            try:
                parts = json.loads(msg)
                kind, payload = parts[0], parts[1]
            except Exception:
                continue
            if kind not in ("stdout", "stderr"):
                continue
            buf.append(payload)
            joined = "".join(buf)
            # The sentinel echoes once in the command line we sent and once as
            # real output; only the second occurrence carries the exit status.
            if joined.count(sent) >= 2:
                tail = ANSI.sub("", joined).rsplit(sent, 1)[1].strip().split()
                rc = int(tail[0]) if tail and tail[0].lstrip("-").isdigit() else None
                break
        out = clean("".join(buf), start, sent)
    finally:
        ws.close()
        try:
            S.delete(BASE + "/api/terminals/" + name, timeout=30)
        except Exception:
            pass
    if not quiet:
        print(out)
    return rc, out


if __name__ == "__main__":
    cmd = sys.stdin.read() if sys.argv[1:2] == ["-"] else " ".join(sys.argv[1:])
    rc, _ = run(cmd, timeout=int(os.environ.get("REXEC_TIMEOUT", "300")))
    print("[exit %s]" % rc)
    raise SystemExit(0 if rc in (0, None) else rc)
