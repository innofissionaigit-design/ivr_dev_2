#!/usr/bin/env python3
"""Push this working tree to a Vast.ai instance over the Jupyter contents API.

WHY NOT A TARBALL, AND WHY NOT GIT
----------------------------------
A tar.gz is a snapshot: building, uploading and unpacking one is a manual
round trip, and the instance silently keeps running whatever was in the
LAST tarball until someone remembers to repeat all three steps. Git has the
opposite problem here -- it only moves what has been committed and pushed,
so an uncommitted local experiment does not exist as far as the pod is
concerned, which is exactly the code you most want to try on a GPU.

This pushes the working tree as it is on disk, file by file, and only the
files whose contents actually changed. Edit locally, re-run this, restart
the services -- that is the whole loop, and it costs a couple of seconds
because an unchanged tree uploads nothing.

Change detection is an MD5 manifest stored ON THE INSTANCE
(/workspace/.vast_push_manifest.json), not locally: the instance is the one
thing that knows what it is actually running, so pushing from a second
machine -- or after this checkout is deleted -- still does the right thing
instead of re-uploading everything or, worse, skipping a file it never sent.

USAGE
    python deploy/vast_push.py                 # push changed files
    python deploy/vast_push.py --all           # ignore the manifest, push everything
    python deploy/vast_push.py --delete        # also remove remote files deleted locally
    python deploy/vast_push.py --dry-run       # show what would move, touch nothing

    VAST_BASE / VAST_TOKEN / VAST_DEST override the defaults below.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import sys

try:
    import requests
    import urllib3
except ImportError:  # pragma: no cover
    sys.exit("pip install requests urllib3")

urllib3.disable_warnings()

# The Vast "Jupyter Terminal" service. BASE is https://<public ip>:<mapped 8080>,
# TOKEN is the instance's $OPEN_BUTTON_TOKEN (same token the portal URL carries).
BASE = os.environ.get("VAST_BASE", "https://137.175.22.196:37759").rstrip("/")
TOKEN = os.environ.get(
    "VAST_TOKEN", "9f6b4d67cb59fc37063d37b6e564ae2174c07d1df40a9144c3b03857282617ea"
)
# Matches deploy/env.sh's PYTHONPATH and deploy/start_all.sh -- do not rename
# one without the others.
DEST = os.environ.get("VAST_DEST", "workspace/kolkata-care-voice-agent")

MANIFEST = "workspace/.vast_push_manifest.json"

# Never worth a GPU pod's disk: caches, local venvs, editor state, the stale
# nested copy of an older checkout, and the archives this script exists to
# replace. .db is excluded on purpose -- clinic.db is SEEDED on the instance
# and is live state there; uploading a local one would silently overwrite
# bookings made through the agent.
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".idea", ".vscode", "venv", ".venv", "env",
    ".ipynb_checkpoints", "official_doc",
}
EXCLUDE_PREFIXES = ("ivr_stagging-9e0dc8a1d713cf921073e59547b16332453ac5a6",)
EXCLUDE_EXTS = (".pyc", ".pyo", ".pyd", ".db", ".db-journal", ".log",
                ".zip", ".gz", ".tar", ".whl", ".so", ".wav.tmp")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

S = requests.Session()
S.verify = False
S.headers["Authorization"] = "token " + TOKEN


def url(path: str) -> str:
    return "%s/api/contents/%s" % (BASE, path.lstrip("/"))


def local_files() -> dict[str, str]:
    """Map of repo-relative posix path -> md5 of the file's bytes."""
    out = {}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        rel_dir = os.path.relpath(root, REPO_ROOT).replace(os.sep, "/")
        if rel_dir != "." and rel_dir.split("/")[0] in EXCLUDE_DIRS:
            continue
        if rel_dir.startswith(EXCLUDE_PREFIXES):
            dirs[:] = []
            continue
        for name in sorted(files):
            if name.endswith(EXCLUDE_EXTS) or name.startswith(".~"):
                continue
            abs_p = os.path.join(root, name)
            rel = os.path.relpath(abs_p, REPO_ROOT).replace(os.sep, "/")
            if rel.startswith(EXCLUDE_PREFIXES):
                continue
            with open(abs_p, "rb") as fh:
                out[rel] = hashlib.md5(fh.read()).hexdigest()
    return out


def read_manifest() -> dict[str, str]:
    """Read the remote manifest, tolerating either serialisation.

    The contents API decides the wire format from the FILE EXTENSION, not from
    how it was written: this is uploaded as base64 but read back as plain text
    because it ends in .json. Assuming base64 here silently yielded an empty
    manifest, which looks exactly like "first run" -- so every push re-uploaded
    the entire tree and the incremental path was never actually exercised.
    """
    try:
        r = S.get(url(MANIFEST), timeout=60)
        if r.status_code != 200:
            return {}
        body = r.json()
        content = body.get("content")
        if body.get("format") == "base64":
            content = base64.b64decode(content).decode()
        if isinstance(content, (dict, list)):  # format == "json"
            return content
        return json.loads(content)
    except Exception:
        return {}


def write_manifest(man: dict[str, str]) -> None:
    put_bytes(MANIFEST, json.dumps(man, indent=0, sort_keys=True).encode())


_made: set[str] = set()


def ensure_dir(path: str) -> None:
    """mkdir -p on the remote. Jupyter has no recursive create, so walk down."""
    parts = [p for p in path.split("/") if p]
    cur = ""
    for p in parts:
        cur = cur + "/" + p if cur else p
        if cur in _made:
            continue
        r = S.get(url(cur), params={"content": "0"}, timeout=60)
        if r.status_code != 200:
            S.put(url(cur), json={"type": "directory"}, timeout=120).raise_for_status()
        _made.add(cur)


def normalize_newlines(rel: str, data: bytes) -> bytes:
    """Convert CRLF to LF for files the SHELL has to parse.

    Editing on Windows means every .sh in this repo is CRLF on disk, and the
    upload is byte-exact, so bash on the instance sees a stray \\r at the end
    of every line. Blank lines become `$'\\r': command not found`, which is
    noisy but harmless -- the damaging half is silent: `export HF_HOME=/path`
    assigns "/path\\r", so mkdir creates a directory whose name ends in a
    carriage return and every later lookup misses it.

    Only shell scripts and shebang files are touched. Python reads CRLF fine,
    and rewriting everything would corrupt anything genuinely binary. The
    manifest records the md5 of the LOCAL bytes either way, so this does not
    make an unchanged file look changed on the next run.
    """
    if not (rel.endswith((".sh", ".bash")) or data[:2] == b"#!"):
        return data
    return data.replace(b"\r\n", b"\n")


def put_bytes(remote: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(remote))
    body = {
        "type": "file",
        "format": "base64",  # base64 for EVERY file: 'text' round-trips through
        "content": base64.b64encode(data).decode(),  # the server's newline and
    }                                                # encoding handling, which
    r = S.put(url(remote), json=body, timeout=300)   # corrupts .py on Windows.
    r.raise_for_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="push every file, ignore manifest")
    ap.add_argument("--delete", action="store_true", help="delete remote files gone locally")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    print("local  : %s" % REPO_ROOT)
    print("remote : %s/%s" % (BASE, DEST))

    local = local_files()
    remote_man = {} if args.all else read_manifest()

    changed = sorted(p for p, h in local.items() if remote_man.get(p) != h)
    removed = sorted(p for p in remote_man if p not in local)

    total_mb = sum(os.path.getsize(os.path.join(REPO_ROOT, p)) for p in changed) / 1e6
    print("files  : %d total, %d to upload (%.2f MB), %d deleted locally"
          % (len(local), len(changed), total_mb, len(removed)))

    if args.dry_run:
        for p in changed:
            print("  PUT    ", p)
        for p in removed:
            print("  DELETE ", p)
        return 0

    if not changed and not (removed and args.delete):
        print("nothing to do -- instance already matches this tree")
        return 0

    # Pre-create directories serially: concurrent mkdir of the same parent
    # races and returns 409 from the contents API.
    for p in changed:
        ensure_dir(os.path.dirname("%s/%s" % (DEST, p)))

    failures: list[tuple[str, str]] = []
    done = [0]

    def upload(rel: str) -> None:
        try:
            with open(os.path.join(REPO_ROOT, rel), "rb") as fh:
                put_bytes("%s/%s" % (DEST, rel), normalize_newlines(rel, fh.read()))
            done[0] += 1
            print("  [%3d/%3d] %s" % (done[0], len(changed), rel))
        except Exception as exc:  # keep going; report every failure at the end
            failures.append((rel, repr(exc)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(upload, changed))

    if args.delete and removed:
        for rel in removed:
            try:
                S.delete(url("%s/%s" % (DEST, rel)), timeout=60)
                print("  DELETED %s" % rel)
            except Exception as exc:
                failures.append((rel, repr(exc)))

    if failures:
        print("\n%d FAILED:" % len(failures))
        for rel, err in failures:
            print("  %s -- %s" % (rel, err))
        # Record only what actually landed, so the next run retries the rest.
        failed_paths = {rel for rel, _ in failures}
        write_manifest({p: h for p, h in local.items() if p not in failed_paths})
        return 1

    write_manifest(local)
    print("\nOK -- %d file(s) pushed to %s" % (len(changed), DEST))
    print("Now restart the services:")
    print("  bash /workspace/kolkata-care-voice-agent/deploy/start_vast.sh restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
