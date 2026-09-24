#!/bin/bash
# ============================================================================
# Bring the stack up / down / round again on a Vast.ai instance.
#
# This is deploy/start_all.sh's job, done for this image instead of RunPod:
# one python at /venv/main rather than two venvs, clinic-api on :8080, and an
# ngrok tunnel because the browser will not hand a page its MICROPHONE unless
# the page is a secure context -- Vast's own mapped ports are plain http, so
# http://<ip>:<port>/ loads the UI and then silently fails at getUserMedia.
#
#   bash deploy/start_vast.sh start       # start whatever is not running
#   bash deploy/start_vast.sh restart     # after a code push
#   bash deploy/start_vast.sh stop
#   bash deploy/start_vast.sh status
#   bash deploy/start_vast.sh logs agent  # tail one service
#   bash deploy/start_vast.sh url         # print the public URL again
#
# Services are launched with setsid + </dev/null so they outlive the shell
# (and the Jupyter terminal) that started them. A plain background job dies
# with its websocket, which has silently left the stack down after a deploy.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/deploy/env_vast.sh"
mkdir -p "$LOGS" /workspace/bin

ok()   { printf '   \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m   %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Kill by PORT, never by command-line pattern. A pattern broad enough to match
# "uvicorn main:app" also matches this script's own command line, and has
# previously killed the thing doing the launching.
kill_port() { fuser -k "$1/tcp" >/dev/null 2>&1; }

stop_service() {  # stop_service <name> <port>
    # The restart loop below would bring a service straight back if only the
    # uvicorn were killed, so stop the loop first: it runs under setsid, which
    # makes its pid its own process-group id, and killing the group takes the
    # loop and its uvicorn together.
    local name=$1 port=$2 pidf="$LOGS/$1.pid"
    if [ -f "$pidf" ]; then
        kill -TERM -- "-$(cat "$pidf")" 2>/dev/null
        rm -f "$pidf"
    fi
    kill_port "$port"
}

launch() {  # launch <name> <port> <workdir> <module:app>
    local name=$1 port=$2 dir=$3 app=$4
    local log="$LOGS/$name.log"
    stop_service "$name" "$port"; sleep 1
    # Each service runs inside a small restart loop, for two reasons:
    #   * the agent has been seen to vanish mid-call with nothing in its log --
    #     no traceback, no shutdown line -- which is what a signal or a crash
    #     in native code looks like. The loop records the exit status (and the
    #     signal, when there is one), and PYTHONFAULTHANDLER=1 makes a native
    #     crash print the Python stack of every thread, so the next occurrence
    #     leaves evidence instead of just a dead port;
    #   * while that is being chased, the public URL comes back by itself in a
    #     few seconds instead of serving ERR_NGROK_8012 until someone notices.
    # Restarts are logged with a "####" marker so they stay visible;
    # `start_vast.sh status` counts them.
    ( cd "$dir" && setsid env \
        PYTHONPATH="/workspace/AI4Bharat_NeMo:$REPO:${PYTHONPATH:-}" \
        PYTHONFAULTHANDLER=1 \
        bash -c '
            echo $$ > "$3"
            while true; do
                "$0" -m uvicorn "$1" --host 0.0.0.0 --port "$2" --log-level info
                rc=$?
                sig=""; [ "$rc" -gt 128 ] && sig=" (signal $((rc - 128)))"
                echo "#### $(date -Is) $1 on :$2 exited rc=${rc}${sig} -- restarting in 3s"
                sleep 3
            done' "$VENV_PY" "$app" "$port" "$LOGS/$name.pid" \
        > "$log" 2>&1 < /dev/null & )
    ok "$name starting on :$port  ($log)"
}

wait_http() {  # wait_http <url> <seconds> -- returns 1 on timeout
    local url=$1 secs=$2
    for _ in $(seq "$secs"); do
        curl -sf -m 2 "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# ---------------------------------------------------------------------------
do_start() {
    echo "== ollama =="
    if curl -sf -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        ok "already running"
    elif [ -x /workspace/bin/ollama ]; then
        setsid /workspace/bin/ollama serve > "$LOGS/ollama.log" 2>&1 < /dev/null &
        wait_http http://127.0.0.1:11434/api/tags 60 && ok "up on :11434" \
            || warn "did not come up -- see $LOGS/ollama.log"
    else
        warn "/workspace/bin/ollama missing -- run deploy/setup_vast.sh ollama"
    fi

    echo "== tts (gtts) =="
    if [ -f /workspace/tts_gtts/main.py ]; then
        launch tts "$TTS_PORT" /workspace/tts_gtts main:app
    else
        warn "/workspace/tts_gtts missing -- run deploy/setup_vast.sh tts"
    fi

    echo "== clinic-api =="
    launch clinic "$CLINIC_API_PORT" "$REPO/clinic-api" main:app

    echo "== voice agent =="
    # NeMo restores the IndicConformer checkpoint at import time, so this port
    # stays closed for 30-60s after launch. That is loading, not a failure.
    launch agent "$AGENT_PORT" "$REPO" main:app
    launch agent-pcm "$AGENT_PCM_PORT" "$REPO" main_pcm:app

    echo
    echo "Waiting for the voice agent to finish loading NeMo (up to 180s)..."
    if wait_http "http://127.0.0.1:$AGENT_PORT/api/health" 180; then
        ok "voice agent ready"
        do_url
    else
        warn "not ready yet -- last 30 lines of $LOGS/agent.log:"
        tail -30 "$LOGS/agent.log"
    fi
}

do_stop() {
    stop_service agent     "$AGENT_PORT"
    stop_service agent-pcm "$AGENT_PCM_PORT"
    stop_service clinic    "$CLINIC_API_PORT"
    stop_service tts       "$TTS_PORT"
    pkill -f "ngrok http" >/dev/null 2>&1
    ok "services stopped (ollama left running -- it holds the warm model)"
}

do_status() {
    probe() {
        local name=$1 url=$2 body
        body=$(curl -sf -m 4 "$url" 2>/dev/null)
        if [ -n "$body" ]; then
            printf '   %-11s \033[32mUP\033[0m    %s\n' "$name" "$(echo "$body" | head -c 110)"
        else
            printf '   %-11s \033[33m....\033[0m  not ready\n' "$name"
        fi
    }
    echo "== services =="
    probe ollama    "http://127.0.0.1:11434/api/tags"
    probe tts       "http://127.0.0.1:$TTS_PORT/health"
    probe clinic    "http://127.0.0.1:$CLINIC_API_PORT/api/health"
    probe agent     "http://127.0.0.1:$AGENT_PORT/api/health"
    probe agent-pcm "http://127.0.0.1:$AGENT_PCM_PORT/api/health"
    echo
    echo "== resident models =="
    /workspace/bin/ollama ps 2>/dev/null || echo "   ollama not reachable"
    echo
    echo "== gpu =="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo
    echo "== crashes / restarts since last start =="
    local n c
    for n in agent agent-pcm clinic tts; do
        # grep -c already prints 0 on no match (and exits 1), so no "|| echo 0".
        c=$(grep -c '^####' "$LOGS/$n.log" 2>/dev/null)
        printf '   %-10s %s\n' "$n" "${c:-0}"
    done
    grep -h '^####' "$LOGS"/agent*.log 2>/dev/null | tail -3 | sed 's/^/   /'
    echo
    [ -f /workspace/.public_url ] && echo "== public url ==" && cat /workspace/.public_url
}

do_url() {
    # ngrok, not Vast's own mapped port, and the reason is not preference:
    # getUserMedia() is gated on a secure context. Vast maps container ports
    # over plain http, so the UI loads and the mic silently never opens.
    # ngrok gives a real TLS cert, which is what the notebook relied on too.
    if [ -z "${NGROK_TOKEN:-}" ]; then
        warn "NGROK_TOKEN unset in /workspace/.env -- no public https URL"
        echo "   Local only: http://127.0.0.1:$AGENT_PORT/"
        return
    fi

    # Run the ngrok BINARY detached, not pyngrok. pyngrok starts ngrok as a
    # child and registers an atexit handler that kills it, so a script that
    # prints the URL and exits takes the tunnel down on its way out -- the URL
    # is real for about a second and then serves ERR_NGROK_3200 "endpoint
    # offline". The tunnel has to outlive whatever created it, exactly like
    # every other service here.
    local ngrok_bin
    ngrok_bin="$(command -v ngrok || echo /root/.config/ngrok/ngrok)"
    [ -x "$ngrok_bin" ] || { warn "no ngrok binary found"; return; }

    if ! curl -sf -m 2 http://127.0.0.1:4040/api/tunnels >/dev/null 2>&1; then
        "$ngrok_bin" config add-authtoken "$NGROK_TOKEN" >/dev/null 2>&1
        # 127.0.0.1, not a bare port: a bare port means "localhost", which
        # ngrok resolves to [::1] first, and uvicorn is bound IPv4-only.
        setsid "$ngrok_bin" http "127.0.0.1:$AGENT_PORT" --log stdout \
            > "$LOGS/ngrok.log" 2>&1 < /dev/null &
        wait_http http://127.0.0.1:4040/api/tunnels 30 \
            || { warn "ngrok agent did not start -- see $LOGS/ngrok.log"; return; }
    fi

    # Read the URL back from ngrok's own local API rather than scraping stdout.
    # The API answers as soon as the web service binds, which is a couple of
    # seconds BEFORE the tunnel session is established -- so a single query
    # returns an empty tunnel list and looks like failure. Poll instead.
    "$VENV_PY" - <<'PY' | tee /workspace/.public_url
import json, time, urllib.request

deadline = time.time() + 45
while time.time() < deadline:
    try:
        data = json.load(urllib.request.urlopen(
            "http://127.0.0.1:4040/api/tunnels", timeout=5))
        urls = [t["public_url"] for t in data.get("tunnels", [])
                if t.get("proto") == "https"]
        if urls:
            print(urls[0])
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("NO_TUNNEL")
PY
    echo
    echo "   Open that URL in Chrome/Firefox and allow the microphone."
    echo "   WebSocket endpoint: <url>/ws/audio"
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start ;;
    status)  do_status ;;
    url)     do_url ;;
    logs)    tail -f "$LOGS/${2:-agent}.log" ;;
    *)       echo "usage: $0 {start|stop|restart|status|url|logs <name>}"; exit 2 ;;
esac
