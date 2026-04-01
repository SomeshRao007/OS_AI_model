FROM debian:trixie

# Suppress interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install live-build and all its build-time dependencies in one layer
# so Docker cache invalidation is coarse-grained (one apt run = one layer).
RUN apt-get update && apt-get install -y --no-install-recommends \
    # live-build core
    live-build \
    debootstrap \
    debian-archive-keyring \
    # ISO creation
    xorriso \
    isolinux \
    syslinux-common \
    squashfs-tools \
    dosfstools \
    mtools \
    # BIOS + UEFI bootloaders (unsigned — fine for development/VM testing)
    grub-pc-bin \
    grub-efi-amd64-bin \
    # Utilities needed during build
    rsync \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
