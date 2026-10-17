"""Build the Rust extension pyfe4ai._rust_native in place (no maturin needed).

Runs ``cargo build --release`` in ``rust/native`` for this interpreter
(``PYO3_PYTHON``) and copies the library to
``pyfe4ai/_rust_native<EXT_SUFFIX>``, where
:mod:`pyfe4ai.utils.native_backend` finds it (it is preferred to the C++
extension; ``PYFE4AI_NATIVE_IMPL=cpp`` selects the latter). Needs a Rust
toolchain and, the first time, the crates of ``rust/native/Cargo.toml`` from
crates.io (``--offline`` uses only cached ones). ``rust/core`` alone needs no
crates: ``cargo test`` there runs offline.

Usage::

    python tools/build_rust.py [--offline] [--debug]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "native"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use only crates already in the cargo cache")
    parser.add_argument("--debug", action="store_true", help="debug build")
    args = parser.parse_args()
    cargo = shutil.which("cargo") or shutil.which("cargo", path="/opt/homebrew/opt/rustup/bin")
    if cargo is None:
        sys.exit("cargo not found (install Rust, or add it to PATH)")
    env = dict(os.environ, PYO3_PYTHON=sys.executable)
    target_dir = Path(env.get("CARGO_TARGET_DIR", CRATE / "target"))
    cmd = [cargo, "build", "--manifest-path", str(CRATE / "Cargo.toml")]
    if not args.debug:
        cmd.append("--release")
    if args.offline:
        cmd.append("--offline")
    subprocess.run(cmd, check=True, env=env)
    profile = "debug" if args.debug else "release"
    names = {"darwin": "lib_rust_native.dylib", "win32": "_rust_native.dll"}
    built = target_dir / profile / names.get(sys.platform, "lib_rust_native.so")
    dest = ROOT / "pyfe4ai" / ("_rust_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copy2(built, dest)
    print("built", dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
