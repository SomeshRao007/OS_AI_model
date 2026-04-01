#!/usr/bin/env bash
# docker-build.sh — Build the AI OS ISO inside a Debian 13 Docker container.
# Nothing is installed on the host machine. All live-build tooling runs inside.
#
# Usage:
#   ./docker-build.sh                          # use default GGUF model
#   ./docker-build.sh --model /path/to/x.gguf # swap in a custom model
#   ./docker-build.sh --no-cache               # force Docker image rebuild
#
# Test the built ISO in QEMU (install qemu-system-x86 on your host if needed):
#   qemu-system-x86_64 -enable-kvm -m 4G -smp 2 \
#     -cdrom os_build/output/*.iso -boot d -vga virtio
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/os_build"
DEFAULT_MODEL="${REPO_ROOT}/finetuning/q4_k_m-deploy/qwen3.5-4b-os-q4km.gguf"
MODEL_PATH="${DEFAULT_MODEL}"
IMAGE_NAME="aios-builder:latest"
NO_CACHE=""

# ── Argument parsing ─────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the AI OS live ISO inside a Debian 13 Docker container.

Options:
  --model PATH   Path to the GGUF model to bake into the ISO.
                 Default: ${DEFAULT_MODEL}
  --no-cache     Rebuild the Docker builder image from scratch.
  --help, -h     Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --model requires a path argument." >&2
                exit 1
            fi
            MODEL_PATH="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# ── Pre-flight checks ────────────────────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found in PATH. Install Docker and retry." >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: GGUF model not found: ${MODEL_PATH}" >&2
    echo "       Use --model /path/to/your.gguf to specify a different path." >&2
    exit 1
fi

# Resolve to absolute path (handles relative paths from caller)
MODEL_PATH="$(realpath "${MODEL_PATH}")"

# ── Build Docker image ───────────────────────────────────────────────────────

echo "==> Building Docker builder image: ${IMAGE_NAME}"
# shellcheck disable=SC2086
docker build ${NO_CACHE} \
    -t "${IMAGE_NAME}" \
    -f "${BUILD_DIR}/Dockerfile.builder" \
    "${BUILD_DIR}"

# ── Run live-build inside the container ─────────────────────────────────────

echo ""
echo "==> Model:   ${MODEL_PATH}"
echo "==> ISO out: ${BUILD_DIR}/output/"
echo ""
echo "==> Starting live-build (first run: 20-60 min, cached: ~10 min)..."
echo "    Build log: ${BUILD_DIR}/build.log"
echo ""

# NOTE: --privileged is required for live-build.
# live-build uses mount(2) (for squashfs loop devices) and chroot — both need
# elevated kernel capabilities. This container is a local build tool, not a
# network-facing service. The model is mounted read-only.
docker run --rm \
    --privileged \
    -v "${BUILD_DIR}:/build/os_build" \
    -v "${MODEL_PATH}:/build/model.gguf:ro" \
    "${IMAGE_NAME}" \
    /build/os_build/build.sh

# ── Report result ────────────────────────────────────────────────────────────

echo ""
echo "==> Build complete!"
if ls "${BUILD_DIR}/output/"*.iso &>/dev/null; then
    ls -lh "${BUILD_DIR}/output/"*.iso
    echo ""
    echo "==> To test in QEMU:"
    echo "    qemu-system-x86_64 -enable-kvm -m 4G -smp 2 \\"
    # shellcheck disable=SC2005
    echo "      -cdrom $(ls "${BUILD_DIR}/output/"*.iso | head -1) -boot d -vga virtio"
else
    echo "WARNING: No ISO found in output/. Check ${BUILD_DIR}/build.log for errors."
    exit 1
fi
