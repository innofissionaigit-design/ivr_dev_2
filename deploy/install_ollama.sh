#!/bin/bash
# Installs Ollama into /workspace/bin, NOT /usr/local.
#
# The official install script (curl -fsSL https://ollama.com/install.sh |
# sh) ignores OLLAMA_INSTALL_DIR in practice on this account's pods and
# always writes the binary to /usr/local/bin and its CUDA backend
# libraries (~2GB) to /usr/local/lib/ollama -- both on the ephemeral
# container overlay, both gone on the next restart. This script instead
# pulls the same release tarball the installer would and extracts it
# straight into /workspace/bin, which survives a restart.
#
# The model WEIGHTS were always the part that actually mattered to keep
# (multi-GB, slow to re-download); OLLAMA_MODELS already points at
# /workspace/.ollama/models in deploy/env.sh regardless of where the
# ollama binary itself lives. This script just closes the remaining gap
# so the ~2GB binary+backend doesn't ALSO need re-fetching every restart.
set -eu

# /workspace/bin IS the final bin dir (matches deploy/start_all.sh's
# hardcoded /workspace/bin/ollama), and libs go in /workspace/lib/ollama
# -- NOT /workspace/bin/lib/ollama -- because the official installer's own
# layout is <install_dir>/bin/ollama + <install_dir>/lib/ollama, and the
# ollama binary locates its backend libs RELATIVE to itself
# (../lib/ollama), so the two must keep that same relative shape.
INSTALL_DIR=/workspace
BINDIR="$INSTALL_DIR/bin"
mkdir -p "$BINDIR" "$INSTALL_DIR/lib/ollama"

echo "Fetching latest Ollama release for linux-amd64..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if curl --fail --silent --head --location \
    "https://ollama.com/download/ollama-linux-amd64.tar.zst" >/dev/null 2>&1; then
    curl -fL --progress-bar "https://ollama.com/download/ollama-linux-amd64.tar.zst" \
        | zstd -d | tar -xf - -C "$INSTALL_DIR"
else
    curl -fL --progress-bar "https://ollama.com/download/ollama-linux-amd64.tgz" \
        | tar -xzf - -C "$INSTALL_DIR"
fi

chmod +x "$BINDIR/ollama"
echo "Installed: $("$BINDIR/ollama" --version 2>&1 | head -1)"
echo "Binary at $BINDIR/ollama, backend libs at $INSTALL_DIR/lib/ollama -- both under /workspace, persistent."
