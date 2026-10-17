"""Build the Rust extension pyfe4ai._rust_native (no maturin needed).

Runs ``cargo build --release`` in ``rust/native`` for this interpreter
(``PYO3_PYTHON``) and copies the library to
``pyfe4ai/_rust_native<EXT_SUFFIX>``, where
:mod:`pyfe4ai.utils.native_backend` finds it, or to ``--dest`` (what
``setup.py`` does for ``pip install``). Needs a Rust toolchain and, the first
time, the crates of ``rust/native/Cargo.toml`` from crates.io (``--offline``
uses only cached ones). ``rust/core`` alone needs no crates: ``cargo test``
there runs offline.

Usage::

    python tools/build_rust.py [--offline] [--debug] [--dest PATH]
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
    parser.add_argument("--dest", default=None, help="where to put the library (default: in the source tree)")
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
    subprocess.run(cmd, check=True, env=env, cwd=CRATE)  # build.rs of the crate adds the link arguments
    profile = "debug" if args.debug else "release"
    names = {"darwin": "lib_rust_native.dylib", "win32": "_rust_native.dll"}
    built = target_dir / profile / names.get(sys.platform, "lib_rust_native.so")
    dest = Path(args.dest) if args.dest else ROOT / "pyfe4ai" / ("_rust_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # copy to a new file and rename it over the old one: overwriting a loaded
    # library in place makes macOS kill the next process that maps it (the
    # kernel caches the code signature per file)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(built, tmp)
    os.replace(tmp, dest)
    print("built", dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest)


if __name__ == "__main__":
    main()
