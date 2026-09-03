# Cross-compilation Instructions

These instructions describe how to build this Python extension for different operating systems from a Linux host. The project uses [maturin](https://github.com/PyO3/maturin) with the [zig linker](https://zig.tools/) so that we don't need native toolchains for macOS or Windows installed.

---

## Quick builds (all architectures & Python versions)

To build all 5 OS/architecture combinations for every supported Python version at once, run the script in this directory:

```sh
./cross_compilation_instructions/build_all.sh
```

This produces **25 wheels** (5 targets × Python 3.10–3.14) and places each one into `dist/<os>/<arch>/`.

**macOS builds require the Xcode SDK root to be set before running.** Every macOS target needs it, no exceptions:

```sh
export SDKROOT=/path/to/your/MacOSX.sdk   # wherever you extracted the Xcode SDK
./cross_compilation_instructions/build_all.sh
```

Individual architecture builds are documented below; `build_all.sh` is a thin wrapper around those same commands.

---

## General requirements

Regardless of target, you need:

- **Rust** (via `rustup`)
- **[zig](https://ziglang.org/download/)** — provides cross-compilation support and bundled C/C++ libraries
- [maturin](https://www.maturin.rs/) — the build tool for Python ↔ Rust interop

Verify everything is available:

```sh
which rustc
which zig
which maturin
```

---

## Compiling on Linux (native, x86_64)

This builds natively using your system's LLVM/Clang toolchain. No special setup required beyond the prerequisites above.

**Command:**

```sh
maturin develop --release -i python3.XX
```

- `develop` installs the built wheel into the active Python environment so you can import it directly (`import potions`).
- Replace `-i python3.XX` with whichever Python interpreter you want to link against.
- Replace `--release` with no flags (or `-O`) if you need debug symbols or faster iteration.

---

## Compiling on Linux (aarch64) via zig

Use this when building an ARM64 wheel from your x86_64 host, e.g. to run natively on a Raspberry Pi 4 / Apple Silicon Mac with Rosetta disabled.

**Steps:**

1. **Set environment variables for the macOS SDK fallback** (required by zig's bundled linker when no native libc is available):
   ```sh
   export MACOSX_DEPLOYMENT_TARGET=10.9
   ```

2. **Build:**
   ```sh
   maturin build --release --target aarch64-unknown-linux-gnu --zig -i python3.XX
   ```

The target `aarch64-unknown-linux-gnu` tells maturin to produce an ARM64 Linux binary; zig supplies the C library and linker. Replace `-i python3.XX` with whichever Python version you want to link against — maturin will download the corresponding interpreter via zig if it isn't on your PATH.

---

## Compiling for macOS (Apple Silicon, aarch64)

**Steps:**

1. **Set environment variables.** Replace `SDKROOT` with where *you* extracted the SDK:
   ```sh
   export SDKROOT=/path/to/your/MacOSX.sdk
   export MACOSX_DEPLOYMENT_TARGET=11.0
   ```

2. **Build:**
   ```sh
   maturin build --release --target aarch64-apple-darwin --zig -i python3.XX
   ```

Replace `-i python3.XX` with whichever Python interpreter you want to link against. The wheel will be placed in `dist/macos/aarch64/`.

---

## Compiling for macOS (Intel, x86_64)

**Steps:**

1. **Set environment variables.** Replace `SDKROOT` with where *you* extracted the SDK:
   ```sh
   export SDKROOT=/path/to/your/MacOSX.sdk
   export MACOSX_DEPLOYMENT_TARGET=10.9
   ```

2. **Build:**
   ```sh
   maturin build --release --target x86_64-apple-darwin --zig -i python3.XX
   ```

Replace `-i python3.XX` with whichever Python interpreter you want to link against. The wheel will be placed in `dist/macos/x86_64/`.

**Known limitations:** Cross-compiled macOS wheels sometimes behave differently from native builds due to SDK differences. Test thoroughly if your code touches low-level system APIs or uses C extensions that depend on specific macOS features.

---

## Compiling for Windows (x86_64)

Make sure the LLVM toolchain is installed using:

Ubuntu / Debian:
```sh
sudo apt install clang llvm lld
```

Fedora / RHEL / CentOS:
```sh
sudo dnf install clang llvm lld
```

Then, run this command to build:

```sh
maturin build --release --target x86_64-pc-windows-msvc --zig -i python3.XX
```

Replace `-i python3.XX` with whichever Python interpreter you want to link against. The wheel will be placed in `dist/windows/x86_64/`.

---

## Output location

Wheels are placed in a per-architecture layout under `dist/`:

```
dist/
├── linux/{x86_64,arm64}/    # native host + ARM64 via zig
├── macos/{aarch64,x86_64}/   # Apple Silicon / Intel
└── windows/x86_64/
```

You can transfer the `.whl` files from `dist/<os>/<arch>/` to your macOS or Windows machine and install via `pip install potions‑x.y.z‑cpXX‑cpXX.whl`.
