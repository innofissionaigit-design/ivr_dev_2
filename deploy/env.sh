# Environment for every service in the stack. Source before launching.
#
# Everything referenced here MUST live under /workspace. RunPod wipes the
# container's overlay filesystem ("/" and "/root") on every restart and
# keeps only the network volume, so anything installed to /usr/local/bin
# or stored in /var/lib is gone the next time the pod boots. That is not a
# hypothetical: the ollama binary and the entire Postgres installation
# have each been lost to it three times.

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_CACHE_DIR=/workspace/.cache/pip

export OLLAMA_MODELS=/workspace/.ollama/models
# -1 keeps models resident indefinitely. Without it Ollama unloads after
# five idle minutes and the next caller pays a 47s cold start mid-call.
export OLLAMA_KEEP_ALIVE=-1

export PYTHONPATH=/workspace/kolkata-care-voice-agent:

# ASR checkpoints -- pilot single-L4 architecture (docs/adr/0001), three
# routed models. Bengali and Hindi are the AI4Bharat NeMo-fork checkpoints
# and load IN-PROCESS in the main venv (same fork, no conflict between
# the two languages -- only English needs mainline NeMo, hence its own
# venv and its own process below). Explicit paths, not the glob
# agent/asr.py's _resolve_nemo_file() used when there was only one
# checkpoint on the box -- with two "*indicconformer*" directories now
# coexisting under HF_HOME, an unqualified glob is ambiguous.
export VOICE_AGENT_NEMO_FILE_BN=/workspace/.cache/huggingface/hub/indicconformer_bn/snap/indicconformer_stt_bn_hybrid_rnnt_large.nemo
export VOICE_AGENT_NEMO_FILE_HI=/workspace/.cache/huggingface/hub/indicconformer_hi/snap/indicconformer_stt_hi_hybrid_rnnt_large.nemo

# English ASR runs as its OWN process in /workspace/venv-en-nemo (mainline
# NVIDIA NeMo -- cannot share a Python environment with the AI4Bharat fork
# above; see agent/asr.py's module docstring for exactly why). Reached
# over HTTP by agent/asr_router.py's English engine adapter.
export VOICE_AGENT_NEMO_FILE_EN=/workspace/.cache/huggingface/hub/fastconformer_en/snap/stt_en_fastconformer_hybrid_large_streaming_multi.nemo
export ENGLISH_ASR_URL=http://localhost:8003

# HF token for the gated IndicConformer bn/hi checkpoints -- read from a
# file, not committed anywhere, never printed. Only needed to (re-)download
# the checkpoints; not needed to load an already-downloaded .nemo file.
if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN
    HF_TOKEN=$(cat /workspace/.hf_token)
fi

# TTS -- all three languages are AI4Bharat Indic-TTS checkpoints (FastPitch
# + HiFi-GAN, coqui-tts format), including English: AI4Bharat's own
# v1-checkpoints-release ships en.zip in the same format as bn/hi, so one
# coqui-tts process serves all three languages -- no second TTS stack
# needed. See tts_server.py's TTS_CHECKPOINTS_ROOT.
export TTS_CHECKPOINTS_ROOT=/workspace/tts_checkpoints

# DATABASE_URL is deliberately NOT set: clinic-api/db.py then defaults to
# sqlite:////workspace/clinic.db, which persists across restarts. Postgres
# physically cannot run on this volume -- see that file's docstring. Set
# this only when pointing at a real external Postgres.
export CLINIC_API_BASE=http://localhost:8080

# The internal services (TTS, clinic API, English ASR) are reached only by the orchestrator on this
# machine, so they bind to loopback unless told otherwise. Only the orchestrator ports are public.
export INTERNAL_BIND_HOST=${INTERNAL_BIND_HOST:-127.0.0.1}

# Service token for the clinic API -- read from a file, never committed or printed, like the HF token
# above. When set, every /api/v1/* request must carry it (clinic-api/main.py) and the orchestrator
# sends it (agent/tools_client.py). Set CLINIC_API_REQUIRE_TOKEN=1 to make the API refuse everything
# if no token is configured, instead of running open.
if [ -f /workspace/.clinic_api_token ]; then
    export CLINIC_API_TOKEN
    CLINIC_API_TOKEN=$(cat /workspace/.clinic_api_token)
fi
export TTS_URL=http://localhost:8002/synthesize
export SILERO_VAD_REPO=/workspace/silero-vad

# /workspace/bin first: that is where the persistent ollama binary lives
# (deploy/install_ollama.sh -- NOT the official installer, which ignores
# OLLAMA_INSTALL_DIR and writes to /usr/local, on the ephemeral overlay).
export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}

# ---- live language routing (main.py) --------------------------------------
# "bn" alone = the original Bengali-only path (no language ID, no Hindi ASR
# loaded): the rollback lever if the 3-language path misbehaves.
export VOICE_AGENT_LANGUAGES=${VOICE_AGENT_LANGUAGES:-bn,hi,en}
# Where the SpeechBrain VoxLingua107 LID model lives (persistent, ~86 MB).
export VOICE_AGENT_LID_DIR=/workspace/lid_model

# ---- pronunciation lexicon (agent/pronunciation.py) -----------------------
# A live call (2026-09-28): the blood-test-list reply was blocked outright --
# every clause names English lab terms ("Sugar", "Calcium", "Lipid Profile",
# "Widal Test"...) that already have a curated Bengali/Hindi spoken form in
# the lexicon, but that module's own sign-off gate keeps every entry silent
# until a native listener approves it (REVIEWED is empty -- nobody has done
# that pass yet). Until sign-off happens, PRONUNCIATION_ALLOW_DRAFT=1 is the
# module's own documented "listening aid": it lets the draft pronunciations
# through so the pilot can be heard and tested end-to-end. Per its docstring
# this is NOT a release setting -- flip it off (or do the native review, see
# agent/pronunciation.review_sheet()) before a real launch.
export PRONUNCIATION_ALLOW_DRAFT=${PRONUNCIATION_ALLOW_DRAFT:-1}

# ---- admission control (agent/admission.py) -------------------------------
# Cap is per process; the WebM and PCM services each count their own calls.
export ADMISSION_MAX_CALLS=${ADMISSION_MAX_CALLS:-28}
# `touch` this file = human-only mode for every NEW call, no deploy needed.
export ADMISSION_BYPASS_FILE=/workspace/.ai_bypass
# Turn-latency load shedding is OFF until calibrated against a real load
# test; set e.g. ADMISSION_SHED_P95_S=2.5 once p95 is measured.
# The HTTP kill-switch (POST /api/admission/bypass) stays disabled unless a
# token exists -- read from a file, never committed:
if [ -f /workspace/.admission_token ]; then
    export ADMISSION_ADMIN_TOKEN
    ADMISSION_ADMIN_TOKEN=$(cat /workspace/.admission_token)
fi
