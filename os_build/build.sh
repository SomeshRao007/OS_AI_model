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
