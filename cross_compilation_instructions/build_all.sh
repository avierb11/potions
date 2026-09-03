#!/usr/bin/env bash
set -euo pipefail

# Build wheels for all OS/architecture combinations × Python 3.10–3.14.
#
# Usage:
#   ./cross_compilation_instructions/build_all.sh            # Linux only (no macOS SDK)
#   export SDKROOT=/path/to/your/MacOSX.sdk                 # enables macOS builds
#   ./cross_compilation_instructions/build_all.sh            # builds all 5 targets
#
# Build artefacts land in:  dist/<os>/<arch>/

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_VERSIONS=(3.10 3.11 3.12 3.13 3.14)
WHEELS_DIR="$PROJECT_DIR/dist"
mkdir -p "$WHEELS_DIR"/{linux/{x86_64,arm64},macos/{aarch64,x86_64},windows/x86_64}

# ---------- helper ----------------------------------------------------------

log() { printf '\n\e[1;32m==>\e[0m %s\n' "$*"; }
section() { printf '\n\e[2;36m---\e[0m %s ---\n' "$*" | tee -a build.log 2>&1 || true; }

# ---------- Linux x86_64 (native) ------------------------------------------
section "Python versions"
for py in "${PYTHON_VERSIONS[@]}"; do
    log "Building Linux x86_64 (python$py)..."
    maturin build --release -o "$WHEELS_DIR/linux/x86_64" -i "python$py"
done

# ---------- Linux arm64 via zig ----------------------------------------------
log ""
section "Building Python versions for Linux aarch64"
for py in "${PYTHON_VERSIONS[@]}"; do
    # MACOSX_DEPLOYMENT_TARGET is required by zig's bundled linker when no native libc is present.
    export MACOSX_DEPLOYMENT_TARGET=11.0
    log "Building Linux arm64 (python$py)..."
    maturin build --release -o "$WHEELS_DIR/linux/arm64" \
        --target aarch64-unknown-linux-gnu --zig -i "python$py"
done

# ---------- macOS Apple Silicon (aarch64) -----------------------------------
if [[ -n "${SDKROOT:-}" ]]; then
    log ""
    section "Building Python versions for macOS arm64"
    for py in "${PYTHON_VERSIONS[@]}"; do
        export MACOSX_DEPLOYMENT_TARGET=11.0
        log "Building macOS arm64 (python$py)..."
        maturin build --release -o "$WHEELS_DIR/macos/aarch64" \
            --target aarch64-apple-darwin --zig -i "python$py"
    done
else
    log ""
    printf '\e[1;33mSkipping macOS arm64 — set SDKROOT to your downloaded Xcode SDK.\n' >&2
fi

# ---------- macOS Intel (x86_64) --------------------------------------------
if [[ -n "${SDKROOT:-}" ]]; then
    log ""
    section "Building Python versions for macOS x86_64"
    for py in "${PYTHON_VERSIONS[@]}"; do
        export MACOSX_DEPLOYMENT_TARGET=11.0
        log "Building macOS x86_64 (python$py)..."
        maturin build --release -o "$WHEELS_DIR/macos/x86_64" \
            --target x86_64-apple-darwin --zig -i "python$py"
    done
else
    log ""
    printf '\e[1;33mSkipping macOS x86_64 — set SDKROOT to your downloaded Xcode SDK.\n' >&2
fi

# ---------- Windows x86_64 --------------------------------------------------
log ""
section "Building Python versions for Windows x86_64"
for py in "${PYTHON_VERSIONS[@]}"; do
    log "Building Windows x86_64 (python$py)..."
    maturin build --release -o "$WHEELS_DIR/windows/x86_64" \
        --target x86_64-pc-windows-msvc --zig -i "python$py"
done

log ""
printf '\e[1;37mBuild complete!\nWheel artefacts:\n' | tee -a build.log 2>&1 || true

# Clean up intermediate artifacts — keep only wheels.
find "$WHEELS_DIR" -type d -empty ! -path "$WHEELS_DIR" -delete 2>/dev/null || true
find "$WHEELS_DIR" -mindepth 2 -type f ! -name '*.whl' -delete 2>/dev/null || true
tree "$WHEELS_DIR" 2>/dev/null | grep -E '\.whl' && printf 'Total files: %s\n' "$(find "$WHEELS_DIR" -name '*.whl' | wc -l)"
echo
log "Done! $(( $(find "$WHEELS_DIR" -name '*.whl' | wc -l) )) wheels built."
