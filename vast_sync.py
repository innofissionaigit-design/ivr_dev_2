#!/usr/bin/env python3
"""Keep merged_code/ on the Vast instance in step with this local tree.

WHY THIS EXISTS ON TOP OF deploy/vast_push.py
---------------------------------------------
vast_push.py already does the hard part -- an incremental, md5-manifest push
over the Jupyter contents API. Two things are missing for day-to-day editing:

  1. It needs VAST_BASE / VAST_TOKEN in the environment every single time, and
     retyping a 64-char token is how you end up pushing to yesterday's pod.
     Those live in vast.env next to this file, NOT inside merged_code/ --
     vast_push.py uploads every file under the repo root, so a token kept in
     there would be shipped to the instance and into any future git checkout.

  2. It is one-shot. "Edit locally and have it reflected" means not having to
     remember to run anything, so --watch polls the tree and pushes within a
     second or two of a save.

USAGE
    python vast_sync.py                # push whatever changed, once
    python vast_sync.py --watch        # push on every save until Ctrl-C
    python vast_sync.py --restart      # push, then restart the agent services
    python vast_sync.py --watch --restart
    python vast_sync.py -- --dry-run   # anything after -- goes to vast_push.py

WATCH MODE AND CHANGE DETECTION
Watch mode polls mtime+size locally (cheap, no dependency on a filesystem
notification API that behaves differently on every OS) and only shells out to
vast_push.py once something actually moved. vast_push.py then re-checks md5s
against the manifest ON THE INSTANCE, so a file that was touched but not
changed still uploads nothing -- the poll is a trigger, not the source of truth.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "merged_code", "merged_code")
ENV_FILE = os.path.join(HERE, "vast.env")
PUSH = os.path.join(REPO, "deploy", "vast_push.py")

# Mirrors vast_push.py's own exclusions. Kept deliberately loose: a false
# positive here just means one wasted push, while missing a directory would
# mean silently never syncing it.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "node_modules", ".idea", ".vscode", "venv", ".venv", "env",
             ".ipynb_checkpoints", "official_doc"}
SKIP_EXTS = (".pyc", ".pyo", ".pyd", ".db", ".db-journal", ".log", ".zip",
             ".gz", ".tar", ".whl", ".so", ".wav.tmp")

RESTART = ("bash /workspace/kolkata-care-voice-agent/deploy/start_vast.sh restart")


def load_env() -> dict[str, str]:
    """vast.env -> environment for the child push, without clobbering a real
    override the caller already exported."""
    env = dict(os.environ)
    if not os.path.exists(ENV_FILE):
        sys.exit("missing %s -- copy the instance's BASE/TOKEN in there first" % ENV_FILE)
    with open(ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    for k in ("VAST_BASE", "VAST_TOKEN"):
        if not env.get(k):
            sys.exit("%s is not set in %s" % (k, ENV_FILE))
    return env


def fingerprint() -> dict[str, tuple[float, int]]:
    out = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(SKIP_EXTS) or name.startswith(".~"):
                continue
            p = os.path.join(root, name)
            try:
                st = os.stat(p)
            except OSError:      # deleted between walk and stat
                continue
            out[p] = (st.st_mtime, st.st_size)
    return out


def push(env, extra, restart) -> int:
    rc = subprocess.call([sys.executable, PUSH] + extra, cwd=REPO, env=env)
    if rc == 0 and restart:
        rexec = os.path.join(HERE, "vast_exec.py")
        if os.path.exists(rexec):
            subprocess.call([sys.executable, rexec, RESTART], env=env)
        else:
            print("restart skipped: %s not found" % rexec)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="push on every save")
    ap.add_argument("--restart", action="store_true", help="restart services after a push")
    ap.add_argument("--interval", type=float, default=1.5, help="poll seconds (watch)")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    env = load_env()
    print("repo   : %s" % REPO)
    print("target : %s" % env["VAST_BASE"])

    if not args.watch:
        return push(env, extra, args.restart)

    print("watching for changes -- Ctrl-C to stop")
    prev = fingerprint()
    push(env, extra, args.restart)          # start from a known-synced state
    while True:
        try:
            time.sleep(args.interval)
            cur = fingerprint()
            if cur == prev:
                continue
            # Let a burst of saves (a formatter rewriting 20 files, a git
            # checkout) settle before pushing, so one edit session is one push.
            while True:
                time.sleep(args.interval)
                nxt = fingerprint()
                if nxt == cur:
                    break
                cur = nxt
            n = len(set(cur) ^ set(prev)) + sum(
                1 for k in set(cur) & set(prev) if cur[k] != prev[k])
            print("\n[%s] %d file(s) touched" % (time.strftime("%H:%M:%S"), n))
            push(env, extra, args.restart)
            prev = cur
        except KeyboardInterrupt:
            print("\nstopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
