#!/usr/bin/env bash
# build.sh — Runs INSIDE the Docker container. Do not run directly on the host.
# Called by docker-build.sh via: docker run ... /build/os_build/build.sh
#
# Sequence:
#   1. Clean any leftover artifacts from a previous partial build
#   2. lb config  → triggers auto/config → sets all lb flags
#   3. lb build   → triggers auto/build  → assembles the ISO
#   4. Move ISO to output/
set -euo pipefail

cd /build/os_build

echo "[aios-build] ── Step 1: Clean previous build artifacts ──────────────"
# --purge clears cache/ as well, forcing fresh package downloads.
# Safe: does NOT remove config/ (our package lists, hooks, includes).
lb clean --purge 2>/dev/null || true

echo "[aios-build] ── Step 1b: Stage os_agent + model into includes.chroot ─"
# Copy the os_agent Python package and GGUF model into the chroot overlay.
# These end up at /opt/ai-daemon/ inside the live filesystem.
# The model is bind-mounted into the container at /build/model.gguf by
# docker-build.sh.

INCLUDES="/build/os_build/config/includes.chroot"
DAEMON_DIR="${INCLUDES}/opt/ai-daemon"

# Copy os_agent source (production code only — no tests, no stubs)
mkdir -p "${DAEMON_DIR}/os_agent"
rsync -a --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='Agent_benchmark_testing/' \
    --exclude='framework_testing/' \
    --exclude='memory_IO_testing/' \
    --exclude='shell_testing/' \
    --exclude='tools_testing/' \
    --exclude='dbus_stub.py' \
    --exclude='unix_socket.py' \
    /build/os_agent/ "${DAEMON_DIR}/os_agent/"

echo "[aios-build] os_agent files staged:"
find "${DAEMON_DIR}/os_agent" -type f | sort | head -50
echo "[aios-build] Total: $(find "${DAEMON_DIR}/os_agent" -type f | wc -l) files"

# Copy the GGUF model
mkdir -p "${DAEMON_DIR}/models"
if [[ -f /build/model.gguf ]]; then
    echo "[aios-build] Copying GGUF model ($(du -h /build/model.gguf | cut -f1))..."
    cp /build/model.gguf "${DAEMON_DIR}/models/qwen3.5-4b-os-q4km.gguf"
else
    echo "[aios-build] WARNING: No model.gguf found at /build/model.gguf"
    echo "[aios-build]          ISO will boot without a default model."
fi

echo "[aios-build] ── Step 2: Configure live-build ────────────────────────"
# Runs auto/config which calls: lb config noauto <our flags>
lb config

echo "[aios-build] ── Step 3: Build ISO ──────────────────────────────────"
# Runs auto/build which calls: lb build noauto
# Tee to build.log so errors are inspectable from the host after the build.
lb build 2>&1 | tee /build/os_build/build.log

# Propagate lb build exit code (pipefail already handles this, but be explicit)
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[aios-build] ERROR: lb build failed. Inspect build.log for details." >&2
    exit 1
fi

echo "[aios-build] ── Step 4: Collect ISO ────────────────────────────────"
ISO_FILE="$(ls /build/os_build/*.iso 2>/dev/null | head -1 || true)"
if [[ -z "${ISO_FILE}" ]]; then
    echo "[aios-build] ERROR: lb build completed but no .iso file was produced." >&2
    echo "[aios-build]        Check build.log for the root cause." >&2
    exit 1
fi

mkdir -p /build/os_build/output
mv "${ISO_FILE}" /build/os_build/output/

echo "[aios-build] ── Done ───────────────────────────────────────────────"
ls -lh /build/os_build/output/*.iso
