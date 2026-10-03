#!/bin/bash
# The one restart entrypoint. Run this after every pod stop/restart --
# it restores what the ephemeral container overlay lost and brings the
# whole stack up. Everything expensive (venvs, model checkpoints, the
# clinic DB) already lives on /workspace and survives a restart; this
# script's job is the SMALL remaining gap: reinstalling the ~2GB Ollama
# binary+backend (the official installer writes it to /usr/local, not
# /workspace -- see deploy/install_ollama.sh's docstring for why that
# could not just be fixed by an env var), then starting every service.
#
#   bash /workspace/kolkata-care-voice-agent/deploy/bootstrap_pod.sh
#
# Safe to re-run. Each step checks before acting.
set -u

REPO=/workspace/kolkata-care-voice-agent
source "$REPO/deploy/env.sh"

echo "== ollama binary =="
if [ -x /workspace/bin/ollama ]; then
    echo "  already present at /workspace/bin/ollama"
else
    bash "$REPO/deploy/install_ollama.sh"
fi

echo
echo "== bringing up services =="
bash "$REPO/deploy/start_all.sh"

echo
echo "== status =="
bash "$REPO/deploy/status.sh"

echo
echo "Note: model *weights* (Ollama, ASR, TTS) were never lost -- only"
echo "checked above, not re-downloaded. If any checkpoint IS actually"
echo "missing (a new pod with an empty /workspace, not a restart of this"
echo "one), see docs/adr/0001-pilot-single-l4-architecture.md section 3"
echo "for what needs fetching and from where."
