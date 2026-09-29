#!/bin/bash
# ============================================================================
# Kolkata Care voice agent -- one-time provisioning of a Vast.ai GPU instance.
#
# This is the Colab notebook (Kolkata_Care_Voice_Agent_Colab_CLEAN) rewritten
# as a script, with the Colab-isms taken out:
#
#   * /content -> /workspace, so the paths match deploy/env.sh and the repo's
#     existing start_all.sh instead of inventing a third layout.
#   * The notebook cloned the project from GitHub. This does not: the code is
#     pushed straight off the developer's working tree by deploy/vast_push.py,
#     so uncommitted edits are testable. Nothing here touches git.
#   * The notebook's Cell 8b patched two agent bugs (llm.py treating an
#     omitted slot key as a validation failure, reply_templates.py emitting
#     "Test Test"). BOTH ARE ALREADY FIXED in this tree -- llm.py's _validate
#     filters "missing" out of `fatal`, and test_rate_reply has the
#     name_has_test guard. Re-applying them would corrupt the current code, so
#     they are deliberately absent.
#   * bge-m3 is pulled as well as qwen2.5:7b. The notebook missed it;
#     agent/semantic_cache.py needs it and silently loses the whole L2 cache
#     without it (it degrades instead of crashing, which is why it went
#     unnoticed there).
#   * Ollama, its models and the HF cache all live under /workspace, not in
#     /usr/local or ~/.cache -- see deploy/env.sh on why anything outside the
#     workspace is treated as disposable.
#
# Idempotent. Every stage stamps itself and is skipped on re-run, so it is
# safe to re-run after a failure, a reboot, or a code push.
#
#   bash deploy/setup_vast.sh                 # everything not yet done
#   bash deploy/setup_vast.sh ollama asr      # just these stages
#   bash deploy/setup_vast.sh --force nemo    # redo one that already stamped
#   bash deploy/setup_vast.sh --list          # stage names
#
# Budget ~25-35 min on a cold instance, nearly all of it NeMo and the model
# downloads. Logs: /workspace/logs/setup.log
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/deploy/env_vast.sh"

STAMPS=/workspace/.setup_stamps
mkdir -p "$STAMPS" "$LOGS" /workspace/bin "$HF_HOME" "$OLLAMA_MODELS"

STAGES=(tokens system python nemo patches ollama asr vad tts clinic fallback)
FORCE=0

log()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '   \033[31mFATAL\033[0m %s\n' "$*"; exit 1; }

done_already() { [ "$FORCE" = 0 ] && [ -f "$STAMPS/$1" ]; }
stamp()        { date -Is > "$STAMPS/$1"; }

# ---------------------------------------------------------------------------
stage_tokens() {
    # /workspace/.env is the only place secrets live. It is outside the repo on
    # purpose: vast_push.py would otherwise ship them back and forth, and the
    # repo is a git checkout. The Vast image also auto-exports this file into
    # every supervisor service, so writing it here wires the tokens everywhere.
    if [ -f /workspace/.env ] && grep -q HF_TOKEN /workspace/.env; then
        ok "/workspace/.env already present"
        return
    fi
    : "${HF_TOKEN:?set HF_TOKEN before the first run}"
    : "${NGROK_TOKEN:=}"
    umask 077
    cat > /workspace/.env <<EOF
HF_TOKEN=$HF_TOKEN
HF_HUB_ENABLE_HF_TRANSFER=0
NGROK_TOKEN=$NGROK_TOKEN
EOF
    chmod 600 /workspace/.env
    ok "wrote /workspace/.env (0600)"
}

# ---------------------------------------------------------------------------
stage_system() {
    log "apt packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # ffmpeg/libsndfile/sox: audio decode for NeMo + pydub. psmisc: fuser, which
    # start_vast.sh kills services by PORT with -- never by command-line pattern,
    # because a pattern broad enough to match a uvicorn also matches the script
    # doing the killing.
    apt-get install -y -qq --no-install-recommends \
        ffmpeg libsndfile1 sox libportaudio2 libasound2-dev curl psmisc zstd \
        || die "apt-get install failed"
    ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3), sox, libsndfile"
}

# ---------------------------------------------------------------------------
stage_python() {
    log "torch + torchaudio (CUDA build matched to this driver)"
    # The notebook detected the CUDA tag at runtime; same idea, because the
    # wheel index is per-CUDA and a mismatch shows up only much later as a
    # cuDNN symbol error inside NeMo.
    local tag
    tag="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d .)"
    [ -z "$tag" ] && tag="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | tr -d .)"
    local index="https://download.pytorch.org/whl/cu${tag:-128}"
    echo "   index: $index"
    # torchcodec is named EXPLICITLY, from the same CUDA-specific index.
    # torchaudio >=2.9 has no audio backend of its own any more -- load() just
    # delegates to torchcodec -- and pip will happily satisfy that dependency
    # from PyPI with a build linked against a DIFFERENT CUDA major (it pulled
    # a CUDA-13 wheel next to this CUDA-12.8 torch). Nothing complains at
    # install time. It fails later, at the first torchaudio.load of a caller's
    # audio, with "Could not load this library: libtorchcodec_image.so" whose
    # real cause is a missing libnvrtc.so.13 three levels down -- i.e. the
    # agent starts clean, reports healthy, and then silently drops every turn.
    pip install -q torch torchaudio torchcodec --index-url "$index" \
        || die "torch install failed"

    log "runtime dependencies"
    # Split from NeMo's own install so a NeMo resolver failure does not also
    # take out the things the agent itself needs (httpx, fastapi, gTTS).
    pip install -q \
        httpx fastapi "uvicorn[standard]" \
        soundfile librosa pydub gTTS \
        sqlalchemy pydantic \
        pyngrok \
        "huggingface_hub>=0.30" \
        || die "pip install of runtime deps failed"

    "$VENV_PY" - <<'PY' || die "torch/torchaudio verification failed"
import sys, torch, torchaudio
print("   torch %s  cuda=%s  available=%s  device=%s" % (
    torch.__version__, torch.version.cuda, torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
print("   torchaudio %s" % torchaudio.__version__)
# Prove the decode path works NOW, on a file this script makes, rather than
# discovering at the first phone call that torchaudio.load raises. main.py
# calls this on every caller turn.
import struct, tempfile, math, os
sr = 16000
n = sr // 2
pcm = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
               for i in range(n))
path = os.path.join(tempfile.mkdtemp(), "probe.wav")
with open(path, "wb") as fh:
    fh.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
             + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
             + b"data" + struct.pack("<I", len(pcm)) + pcm)
try:
    w, got_sr = torchaudio.load(path)
except Exception as exc:
    print("   FATAL torchaudio.load failed: %r" % (exc,))
    print("   Usually a torchcodec built for the wrong CUDA major. Check with:")
    print("     ldd $(python3 -c 'import torchcodec,os;"
          "print(os.path.dirname(torchcodec.__file__))')/libtorchcodec_image.so | grep 'not found'")
    sys.exit(1)
print("   torchaudio.load OK -- %s @ %dHz" % (tuple(w.shape), got_sr))
PY
}

# ---------------------------------------------------------------------------
stage_nemo() {
    log "AI4Bharat NeMo fork (branch nemo-v2)"
    # Mainline NeMo CANNOT load this checkpoint -- IndicConformer uses a
    # multilingual AGGREGATE tokenizer and mainline's _setup_monolingual_tokenizer
    # raises KeyError: 'dir'. Only this fork handles it. See agent/asr.py's
    # module docstring; that is a debugged fact, not a preference.
    if [ ! -d /workspace/AI4Bharat_NeMo/.git ]; then
        rm -rf /workspace/AI4Bharat_NeMo
        git clone --depth=1 -b nemo-v2 \
            https://github.com/AI4Bharat/NeMo.git /workspace/AI4Bharat_NeMo \
            || die "NeMo clone failed"
    fi
    ok "cloned"

    log "NeMo ASR dependencies (5-10 min)"
    # Install from the fork's OWN requirement files rather than a hand-written
    # list: guessing the list is how the first attempt shipped a NeMo that
    # imported far enough to look installed and then died on `wrapt`.
    #
    # But not verbatim either. These files pin against a 2023 stack --
    # huggingface_hub==0.23.2, hydra <=1.3.2, plain `torch` -- and honouring
    # those pins would downgrade the CUDA-matched torch installed above and
    # undo the huggingface_hub patch in the next stage. So the pinned-elsewhere
    # packages are filtered out and the rest installed at whatever version
    # resolves cleanly against what is already here.
    local reqs=/tmp/nemo_reqs.txt
    cat /workspace/AI4Bharat_NeMo/requirements/requirements.txt \
        /workspace/AI4Bharat_NeMo/requirements/requirements_common.txt \
        /workspace/AI4Bharat_NeMo/requirements/requirements_asr.txt \
        /workspace/AI4Bharat_NeMo/requirements/requirements_lightning.txt \
        2>/dev/null \
    | sed 's/#.*//' | tr -d ' \t' | grep -v '^$' \
    | grep -viE '^(torch|torchaudio|torchvision|triton|huggingface_hub|pytorch-lightning|lightning|hydra-core|omegaconf|torchmetrics|numpy|numba|wandb|ipywidgets)([<>=!].*)?$' \
        > "$reqs"
    echo "   $(wc -l < "$reqs") packages after filtering"

    if ! pip install -q -r "$reqs"; then
        # One unbuildable package (texterrors and kaldi-python-io both need a
        # compiler and routinely fail) must not take the other thirty with it.
        warn "bulk install failed -- retrying one at a time"
        local failed=()
        while read -r pkg; do
            [ -z "$pkg" ] && continue
            pip install -q "$pkg" 2>/dev/null || failed+=("$pkg")
        done < "$reqs"
        [ ${#failed[@]} -gt 0 ] && warn "could not install: ${failed[*]}"
    fi

    # Extra deps this project needs that NeMo's own files do not list.
    pip install -q einops rotary-embedding-torch onnxruntime || true

    # The grep above drops these four because the fork pins them against a 2023
    # stack (hydra<=1.3.2, plain torch via pytorch-lightning) and honouring the
    # pins would downgrade the CUDA-matched torch installed in stage_python.
    # Filtering them out is right; leaving it there was not -- NOTHING else
    # installs them, so `nemo.collections.asr` died on `import hydra` and took
    # the patches stage's verification down with it. The failure reads like a
    # NeMo problem, not a missing dependency, which is why it is called out
    # here: install them UNPINNED, so they resolve against the torch already
    # present instead of dictating it.
    pip install -q "hydra-core>=1.3" "omegaconf>=2.3" \
                   "pytorch-lightning>=2.0" "torchmetrics>=1.0" \
        || die "hydra/lightning install failed -- NeMo cannot import without them"

    # --no-deps -e: editable so that PYTHONPATH (see env.sh) and the installed
    # package are the SAME tree -- the next stage patches files in place, and a
    # copied install would leave those patches applying to a directory nothing
    # imports. --no-deps because the resolution just happened, deliberately
    # ignoring pins that setup.py would re-assert.
    pip install -q --no-deps -e /workspace/AI4Bharat_NeMo \
        || warn "editable NeMo install failed -- PYTHONPATH fallback still applies"
    ok "NeMo installed"
}

# ---------------------------------------------------------------------------
stage_patches() {
    log "NeMo / huggingface_hub compatibility patches"
    # Every one of these is a real breakage between this 2023-era fork and
    # current upstream libraries, carried over from the notebook's Cells 4,
    # 12 and 13. Each is idempotent.
    "$VENV_PY" - <<'PY'
import glob, os, re, sys

NEMO = "/workspace/AI4Bharat_NeMo"
changed = []
touched = set()   # every file this script wrote, for the compile check below

def rewrite(path, fn, label):
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        print("   .. skip (absent) %s" % label); return
    new = fn(src)
    if new is None or new == src:
        print("   .. already ok   %s" % label); return
    touched.add(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("   ++ patched      %s" % label); changed.append(label)

def files_containing(needle):
    hits = []
    for p in glob.glob(os.path.join(NEMO, "**", "*.py"), recursive=True):
        try:
            with open(p, encoding="utf-8") as fh:
                if needle in fh.read():
                    hits.append(p)
        except OSError:
            pass
    return hits

# 1. huggingface_hub >=0.17 removed ModelFilter; NeMo still imports it.
def fix_modelfilter(src):
    old = "from huggingface_hub import HfApi, ModelCard, ModelCardData, ModelFilter"
    # Guard on the REPLACEMENT, not the original: the original line survives
    # inside the try block this patch creates, so "is the old text still here?"
    # is true forever and a second run wraps the wrapper -- which is not a
    # no-op, it is an IndentationError that stops NeMo importing at all.
    if "class ModelFilter" in src or old not in src:
        return None
    return src.replace(old,
        "try:\n"
        "    from huggingface_hub import HfApi, ModelCard, ModelCardData, ModelFilter\n"
        "except ImportError:  # removed in huggingface_hub >=0.17\n"
        "    from huggingface_hub import HfApi, ModelCard, ModelCardData\n"
        "    class ModelFilter:\n"
        "        def __init__(self, *a, **kw): pass", 1)
rewrite(os.path.join(NEMO, "nemo/core/classes/mixins/hf_io_mixin.py"),
        fix_modelfilter, "hf_io_mixin.py ModelFilter")

# 2. pytorch_lightning removed NeptuneLogger; NeMo still imports it.
def fix_neptune(src):
    if "NeptuneLogger = None" in src:
        return None
    def wrap(m):
        line = m.group(0)
        ind = " " * (len(line) - len(line.lstrip()))
        return ("%stry:\n%s    %s\n%sexcept ImportError:\n"
                "%s    NeptuneLogger = None  # dropped from pytorch_lightning"
                % (ind, ind, line.lstrip(), ind, ind))
    return re.sub(r"^[ \t]*from pytorch_lightning(?:\.\w+)* import[^\n]*\bNeptuneLogger\b[^\n]*$",
                  wrap, src, flags=re.MULTILINE)
for p in files_containing("NeptuneLogger"):
    rewrite(p, fix_neptune, "NeptuneLogger in %s" % os.path.basename(p))

# 3. np.sctypes was removed in NumPy 2.0.
def fix_sctypes(src):
    if "np.sctypes" not in src:
        return None
    src = re.sub(r"samples\.dtype(?:\.type)?\s+in\s+np\.sctypes\['int'\]",
                 "np.issubdtype(samples.dtype, np.integer)", src)
    src = re.sub(r"samples\.dtype(?:\.type)?\s+not\s+in\s+np\.sctypes\['int'\]",
                 "not np.issubdtype(samples.dtype, np.integer)", src)
    for key, repl in [("int", "[np.int8, np.int16, np.int32, np.int64]"),
                      ("uint", "[np.uint8, np.uint16, np.uint32, np.uint64]"),
                      ("float", "[np.float16, np.float32, np.float64]"),
                      ("complex", "[np.complex64, np.complex128]")]:
        src = re.sub(r"np\.sctypes\['%s'\]" % key, repl.replace("\\", "\\\\"), src)
    return src
for p in files_containing("np.sctypes"):
    rewrite(p, fix_sctypes, "np.sctypes in %s" % os.path.basename(p))

# 4. A non-contiguous audio buffer reaching the C decoder crashes the whole
#    process with "malloc(): unaligned tcache chunk" -- not an exception, a
#    SIGABRT that takes every in-flight call down with it.
def fix_contiguous(src):
    m = re.search(r"([ \t]*)def _convert_samples_to_float32\([^)]*\):[^\n]*\n", src)
    if not m or "C_CONTIGUOUS" in src[m.end():m.end() + 400]:
        return None
    ind = m.group(1) + "    "
    ins = ("%s# Non-contiguous input aborts the process inside the C decoder.\n"
           "%simport numpy as _np\n"
           "%sif not samples.flags['C_CONTIGUOUS']:\n"
           "%s    samples = _np.ascontiguousarray(samples)\n"
           % (ind, ind, ind, ind))
    return src[:m.end()] + ins + src[m.end():]
rewrite(os.path.join(NEMO, "nemo/collections/asr/parts/preprocessing/segment.py"),
        fix_contiguous, "segment.py contiguous guard")

# 5. transformers imports is_offline_mode from huggingface_hub's top level,
#    which 1.x removed. Resolve the path instead of hardcoding a python
#    version, which is what broke this when the notebook moved off 3.13.
import huggingface_hub
hf_init = os.path.join(os.path.dirname(huggingface_hub.__file__), "__init__.py")
def fix_offline(src):
    if "def is_offline_mode" in src:
        return None
    return src + (
        "\n\n# Compatibility shim for transformers (removed in huggingface_hub 1.x)\n"
        "import os as _os\n"
        "def is_offline_mode() -> bool:\n"
        "    return _os.getenv('HF_HUB_OFFLINE', '0') in ('1', 'true', 'True')\n")
rewrite(hf_init, fix_offline, "huggingface_hub is_offline_mode shim")

# Syntax-check everything this script rewrote. A patch that produces invalid
# Python otherwise surfaces as an import error several modules deep, where it
# reads like a dependency problem rather than "the patcher broke the file".
import py_compile
broken = []
for path in touched:
    try:
        py_compile.compile(path, doraise=True)
    except Exception as exc:
        broken.append("%s: %s" % (path, exc))
if broken:
    print("   PATCHED FILES DO NOT COMPILE:")
    for b in broken:
        print("     " + b)
    print("   Restore them and re-run:")
    print("     git -C /workspace/AI4Bharat_NeMo checkout -- .")
    print("     bash deploy/setup_vast.sh --force patches")
    sys.exit(1)

print("   %d file(s) changed, all compile" % len(changed))
PY

    log "verifying NeMo imports"
    PYTHONPATH="/workspace/AI4Bharat_NeMo:$REPO" "$VENV_PY" -c \
        "import nemo.collections.asr as a; print('   nemo.collections.asr OK ->', a.__file__)" \
        || die "nemo.collections.asr still will not import -- read the traceback above"
}

# ---------------------------------------------------------------------------
stage_ollama() {
    log "ollama"
    # Installed into /workspace/bin, models into /workspace/.ollama/models.
    # The official installer targets /usr/local, which on this image is
    # container storage and is gone on recycle -- see deploy/env.sh.
    if [ ! -x /workspace/bin/ollama ]; then
        # .tar.zst, not .tgz: that is the only Linux asset current releases
        # publish, which is why the notebook used it too. Both ollama.com's
        # /download/*.tgz and the GitHub *.tgz asset 404.
        local url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"
        curl -fsSL "$url" -o /tmp/ollama.tar.zst || die "ollama download failed: $url"
        rm -rf /workspace/ollama-dist
        mkdir -p /workspace/ollama-dist
        tar -I zstd -xf /tmp/ollama.tar.zst -C /workspace/ollama-dist \
            || die "ollama extract failed (is zstd installed? run the 'system' stage)"
        rm -f /tmp/ollama.tar.zst
        # The archive is bin/ + lib/, and the two must stay siblings: ollama
        # finds its bundled CUDA runners at ../lib/ollama relative to the REAL
        # path of its binary. Copying just the binary to /usr/local/bin is what
        # left the notebook debugging a server that started and then returned
        # nothing for every generate call.
        [ -d /workspace/ollama-dist/lib/ollama ] \
            || warn "no lib/ollama in the archive -- GPU runners may be missing"
        ln -sf /workspace/ollama-dist/bin/ollama /workspace/bin/ollama
    fi
    ok "$(/workspace/bin/ollama --version 2>&1 | head -1)"

    # OLLAMA_KEEP_ALIVE=-1 (from env.sh): without it Ollama evicts the model
    # after five idle minutes and the next caller eats a ~47s cold start in
    # the middle of a phone call.
    if ! curl -sf -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        setsid /workspace/bin/ollama serve > "$LOGS/ollama.log" 2>&1 < /dev/null &
        for _ in $(seq 60); do
            curl -sf -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
            sleep 1
        done
    fi
    curl -sf -m 5 http://127.0.0.1:11434/api/tags >/dev/null \
        || die "ollama server never came up -- see $LOGS/ollama.log"
    ok "server up on :11434"

    # qwen2.5:7b  -> agent/llm.py's OLLAMA_MODEL (intent + slot extraction)
    # bge-m3      -> agent/semantic_cache.py's EMBED_MODEL. The notebook never
    #                pulled this; without it every L2 cache lookup raises
    #                EmbeddingUnavailable and is swallowed, so the cache is
    #                simply never warm and nobody sees an error.
    for model in qwen2.5:7b bge-m3; do
        if /workspace/bin/ollama list 2>/dev/null | grep -q "^${model%%:*}"; then
            ok "$model already pulled"
        else
            echo "   pulling $model ..."
            /workspace/bin/ollama pull "$model" || die "ollama pull $model failed"
        fi
    done
    /workspace/bin/ollama list
}

# ---------------------------------------------------------------------------
stage_asr() {
    log "IndicConformer Bengali ASR checkpoint (~1.8 GB)"
    # Gated model: the licence must be accepted once, by hand, at
    # huggingface.co/ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large
    # with the same account HF_TOKEN belongs to. A 401/403 here means that
    # was not done -- no amount of retrying fixes it.
    "$VENV_PY" - <<'PY' || exit 1
import glob, os, sys
from huggingface_hub import snapshot_download

MODEL_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
token = os.environ.get("HF_TOKEN")
try:
    d = snapshot_download(repo_id=MODEL_ID, token=token,
                          ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
except Exception as exc:
    print("   FATAL %r" % (exc,))
    print("   If this is 401/403: accept the licence at")
    print("   https://huggingface.co/%s while signed in as the" % MODEL_ID)
    print("   owner of HF_TOKEN, then re-run: bash deploy/setup_vast.sh asr")
    sys.exit(1)

hits = glob.glob(os.path.join(d, "**", "*.nemo"), recursive=True)
if not hits:
    print("   FATAL no .nemo in %s -- download incomplete" % d)
    sys.exit(1)
print("   checkpoint: %s (%.0f MB)" % (hits[0], os.path.getsize(hits[0]) / 1e6))
PY
    ok "ASR checkpoint present"
}

# ---------------------------------------------------------------------------
stage_vad() {
    log "Silero VAD (local clone)"
    # agent/vad_stream.py prefers a LOCAL repo (SILERO_VAD_REPO) and only falls
    # back to torch.hub's GitHub fetch. Cloning it means turn-taking does not
    # depend on github.com being reachable mid-call.
    # Gate on hubconf.py, not .git: torch.hub's source="local" only needs the
    # tree, and the tarball fallback below leaves no .git behind. Checking for
    # .git would re-download a perfectly good checkout on every re-run.
    if [ ! -f "$SILERO_VAD_REPO/hubconf.py" ]; then
        rm -rf "$SILERO_VAD_REPO"
        # git-over-TLS to github.com is the fragile path -- on a pod in a
        # region that interferes with it, the clone dies with "GnuTLS recv
        # error (-110)" while plain HTTPS to codeload still works. Try git
        # first (cheap, gives a real checkout), then fall back to the tarball
        # rather than failing the whole run over a transport detail.
        # timeout 120: the interference does not reset the connection, it
        # STALLS it -- observed here as a clone sitting at 536K of .git for
        # minutes with no error. Without a cap the fallback never gets its
        # turn and the stage hangs instead of failing over.
        if ! timeout 120 git clone --depth=1 \
                https://github.com/snakers4/silero-vad.git \
                "$SILERO_VAD_REPO" 2>/dev/null; then
            warn "git clone failed -- falling back to the source tarball"
            rm -rf "$SILERO_VAD_REPO" /tmp/silero.tgz
            curl -fsSL --max-time 600 -o /tmp/silero.tgz \
                https://codeload.github.com/snakers4/silero-vad/tar.gz/refs/heads/master \
                || die "silero-vad: git clone AND tarball download both failed"
            mkdir -p "$SILERO_VAD_REPO"
            # --strip-components=1: the archive nests everything under
            # silero-vad-master/, and hubconf.py must sit at the repo root.
            tar -xzf /tmp/silero.tgz -C "$SILERO_VAD_REPO" --strip-components=1 \
                || die "silero-vad tarball extract failed"
            rm -f /tmp/silero.tgz
        fi
        [ -f "$SILERO_VAD_REPO/hubconf.py" ] \
            || die "silero-vad fetched but has no hubconf.py at its root"
    fi
    PYTHONPATH="/workspace/AI4Bharat_NeMo:$REPO" "$VENV_PY" - <<'PY' || die "silero load failed"
import os, torch
repo = os.environ["SILERO_VAD_REPO"]
m, utils = torch.hub.load(repo_or_dir=repo, source="local", model="silero_vad", onnx=False)
print("   silero_vad loaded from", repo)
PY
}

# ---------------------------------------------------------------------------
stage_tts() {
    log "Bengali TTS service (gTTS) on :$TTS_PORT"
    # The repo's own tts_server.py is the AI4Bharat FastPitch+HiFi-GAN wrapper
    # and needs coqui-TTS plus a checkpoint tree at /workspace/tts_checkpoints
    # that is not published anywhere this script can fetch unattended. gTTS is
    # what the notebook actually ran, it satisfies the exact contract
    # agent/tts.py expects (POST {"text","lang"} -> WAV bytes), needs no GPU,
    # and leaves all 24 GB for ASR + the LLM. Swap in tts_server.py later by
    # pointing TTS_URL elsewhere; nothing else changes.
    mkdir -p /workspace/tts_gtts
    cat > /workspace/tts_gtts/main.py <<'PY'
"""Bengali TTS shim: gTTS -> mp3 -> WAV, matching agent/tts.py's contract."""
import io

from fastapi import FastAPI, Request
from fastapi.responses import Response
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI()


@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "engine": "gtts"}


@app.post("/synthesize")
async def synthesize(req: Request):
    body = await req.json()
    # agent/tts.py has already run bn_normalize.verbalize() over this text, so
    # digits arrive spelled out in Bengali words -- do not "helpfully" reformat.
    text = (body.get("text") or "").strip() or "..."
    lang = body.get("lang") or "bn"
    if lang not in ("bn", "en", "hi"):
        lang = "bn"
    mp3 = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(mp3)
    mp3.seek(0)
    wav = io.BytesIO()
    AudioSegment.from_file(mp3, format="mp3").export(wav, format="wav")
    return Response(content=wav.getvalue(), media_type="audio/wav")
PY
    ok "wrote /workspace/tts_gtts/main.py"
}

# ---------------------------------------------------------------------------
stage_clinic() {
    log "clinic database (SQLite at ${CLINIC_DB_PATH:-/workspace/clinic.db})"
    # db.py defaults to sqlite:////workspace/clinic.db and DATABASE_URL is
    # deliberately left unset -- see that file's docstring on why Postgres
    # cannot live on a workspace volume.
    cd "$REPO/clinic-api" || die "no clinic-api dir -- was the code pushed?"
    if [ -f /workspace/clinic.db ] && [ "$FORCE" = 0 ]; then
        ok "clinic.db exists -- leaving live data alone (use --force clinic to reseed)"
    else
        [ "$FORCE" = 1 ] && rm -f /workspace/clinic.db
        PYTHONPATH="$REPO/clinic-api:$REPO" "$VENV_PY" seed.py || die "seed.py failed"
        ok "seeded"
    fi
    "$VENV_PY" - <<'PY'
import sqlite3
c = sqlite3.connect("/workspace/clinic.db")
for t in ("departments", "doctors", "lab_tests"):
    try:
        print("   %-14s %d rows" % (t, c.execute("select count(*) from %s" % t).fetchone()[0]))
    except Exception as exc:
        print("   %-14s -- %s" % (t, exc))
PY
}

# ---------------------------------------------------------------------------
stage_fallback() {
    log "fallback audio"
    # These play when live TTS is DOWN, so they must not be generated on
    # demand by the thing that is down. The notebook shipped sine-wave beeps;
    # gTTS is already here, so synthesize the real Bengali lines once and keep
    # the beep only as a last resort. setup_addon.sh says the same thing.
    mkdir -p "$REPO/static/fallback_audio"
    "$VENV_PY" - "$REPO/static/fallback_audio" <<'PY'
import io, math, os, struct, sys

out_dir = sys.argv[1]
LINES = {
    "sorry_repeat.wav": ("দুঃখিত, শুনতে পাইনি, আবার বলুন", 523, 1.0),
    "system_busy.wav":  ("একটু সমস্যা হচ্ছে, একটু ধরুন", 392, 1.2),
    "check_failed.wav": ("এখনই দেখতে পারছি না, স্টাফের কাছে দিচ্ছি", 330, 1.0),
}

def beep(freq, dur, sr=22050):
    n = int(sr * dur)
    b = io.BytesIO()
    b.write(b"RIFF"); b.write(struct.pack("<I", 36 + n * 2))
    b.write(b"WAVEfmt "); b.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    b.write(b"data"); b.write(struct.pack("<I", n * 2))
    for i in range(n):
        b.write(struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)
                                      * math.exp(-3 * i / n))))
    return b.getvalue()

for name, (text, freq, dur) in LINES.items():
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("   .. keep %s" % name); continue
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        mp3 = io.BytesIO(); gTTS(text=text, lang="bn").write_to_fp(mp3); mp3.seek(0)
        AudioSegment.from_file(mp3, format="mp3").export(path, format="wav")
        print("   ++ %s (bengali speech)" % name)
    except Exception as exc:
        with open(path, "wb") as fh:
            fh.write(beep(freq, dur))
        print("   ++ %s (BEEP fallback -- gTTS failed: %s)" % (name, exc))
PY
}

# ---------------------------------------------------------------------------
usage() { echo "usage: $0 [--force] [--list] [stage ...]   stages: ${STAGES[*]}"; }

want=()
for a in "$@"; do
    case "$a" in
        --force|-f) FORCE=1 ;;
        --list|-l)  usage; exit 0 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         usage; exit 2 ;;
        *)          want+=("$a") ;;
    esac
done
[ ${#want[@]} -eq 0 ] && want=("${STAGES[@]}")

start_ts=$(date +%s)
for s in "${want[@]}"; do
    case " ${STAGES[*]} " in *" $s "*) ;; *) die "unknown stage: $s" ;; esac
    if done_already "$s"; then
        printf '   \033[90m-- skip %s (stamped %s)\033[0m\n' "$s" "$(cat "$STAMPS/$s")"
        continue
    fi
    "stage_$s" || die "stage '$s' failed"
    stamp "$s"
done

printf '\n\033[1;32mSetup finished in %ds\033[0m\n' "$(( $(date +%s) - start_ts ))"
echo "Next:  bash $REPO/deploy/start_vast.sh start"
