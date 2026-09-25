# Vast.ai overlay on top of deploy/env.sh. Source THIS, not env.sh directly.
#
# env.sh was written for RunPod and is kept as-is because it documents why
# each value is what it is. Everything below is a difference this image
# forces, and nothing else -- if a variable is not here, env.sh's value is
# the one in effect.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/env.sh"

# The image ships ONE python, a conda/uv venv at /venv/main, already on PATH
# and already activated by the login shell. env.sh's /workspace/venv does not
# exist here; there is no second environment to keep the agent's deps out of.
export VENV_PY="${VENV_PY:-/venv/main/bin/python3}"
export PATH="/workspace/bin:/venv/main/bin:${PATH:-}"

# clinic-api moves to 8090. env.sh puts it on 8080, which is free on RunPod but
# NOT here: Jupyter binds 8080 on this image (see /etc/portal.yaml), and it is
# the service this instance is administered through. Leaving it on 8080 means
# clinic-api silently fails to bind and every tool call the agent makes returns
# a connection error mid-call -- so CLINIC_API_BASE is overridden to match.
# The voice agent takes 8100 and the PCM variant 8101, matching start_all.sh.
export CLINIC_API_PORT="${CLINIC_API_PORT:-8090}"
export CLINIC_API_BASE="http://localhost:${CLINIC_API_PORT}"
export AGENT_PORT="${AGENT_PORT:-8100}"
export AGENT_PCM_PORT="${AGENT_PCM_PORT:-8101}"
export TTS_PORT="${TTS_PORT:-8002}"

# Tokens live in /workspace/.env, NOT in the repo -- this file is committed and
# a token in git is a token to rotate. setup_vast.sh writes that file once.
# shellcheck disable=SC1091
[ -f /workspace/.env ] && set -a && . /workspace/.env && set +a

# asr.py can glob the checkpoint out of HF_HOME on its own, but only after a
# successful download; pinning it explicitly makes a half-finished download
# fail loudly at startup instead of silently picking a partial file.
if [ -z "${VOICE_AGENT_NEMO_FILE:-}" ]; then
    _nemo_ckpt="$(ls -1 "${HF_HOME}"/hub/models--ai4bharat--indicconformer*/snapshots/*/*.nemo 2>/dev/null | head -1)"
    [ -n "$_nemo_ckpt" ] && export VOICE_AGENT_NEMO_FILE="$_nemo_ckpt"
fi

# Clinic time. Vast images run in UTC, and every "today" in this stack is a
# naive datetime.now() / date.today(): the relative-date rule ("কাল" = the
# call date + 1), the earliest-slot search, and the "no slot that has already
# passed, or starts within 30 minutes" cut in clinic-api. On a UTC clock all
# of those are 5.5 hours behind Kolkata -- between midnight and 05:30 IST
# "কাল" named TODAY's date, and in the daytime today's past slots were
# offered as free. (The cancellation rules alone use an explicit +05:30 and
# were right regardless.) The deploy notes for those stories say the same.
export TZ="${TZ:-Asia/Kolkata}"

export REPO="${REPO:-/workspace/kolkata-care-voice-agent}"
export LOGS="${LOGS:-/workspace/logs}"

# LD_LIBRARY_PATH is deliberately NOT set, unlike start_all.sh and the
# notebook's Cell 12. On this image the pip torch wheel gets cuDNN 9.19 from
# site-packages/nvidia/cudnn and preloads it itself, while the CUDA base image
# puts cuDNN 9.8 in /usr/lib/x86_64-linux-gnu. Adding that directory to the
# search path makes the 9.8 copy win, and the agent dies at startup with
# "cuDNN version incompatibility: compiled against 9.19, found 9.8".
# torchcodec does not need it either, provided it is the cu128 build (see
# setup_vast.sh stage_python). Whatever the image itself exports is left alone.
