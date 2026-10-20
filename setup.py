"""Build of the optional native extension; the package metadata is in pyproject.toml.

``pyfe4ai._rust_native`` (Rust and PyO3, sources in ``rust/``) runs the
Ring-LWE, dense-LWE and ristretto255 hot paths and provides the BLS12-381
pairing group; see :mod:`pyfe4ai.backend`. It is built with
cargo (``tools/build_rust.py``) when a Rust toolchain is on the ``PATH``. The
extension is optional: if it does not build, the package installs without it,
the schemes run their Python implementation, and pairing-based schemes need
a charm-crypto group (BLS12-381 comes only from the extension). Set
``PYFE4AI_NO_NATIVE=1`` to skip it. In a source checkout::

    python tools/build_rust.py
"""

import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildRust(build_ext):
    """build_ext that builds the Rust extension with cargo instead of a C compiler."""

    def build_extension(self, ext) -> None:
        dest = os.path.abspath(self.get_ext_fullpath(ext.name))
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "build_rust.py")
        try:
            subprocess.run([sys.executable, script, "--dest", dest], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.warn("{} not built ({}); the package uses its Python implementation".format(ext.name, exc))

    def copy_extensions_to_source(self) -> None:
        # in-place and editable builds: copy the extension only if it was built
        inplace, self.inplace = self.inplace, False
        built = all(os.path.exists(self.get_ext_fullpath(ext.name)) for ext in self.extensions)
        self.inplace = inplace
        if built:
            super().copy_extensions_to_source()


def _extensions() -> list:
    if os.environ.get("PYFE4AI_NO_NATIVE"):
        return []
    return [Extension("pyfe4ai._rust_native", sources=[], optional=True)]


setup(ext_modules=_extensions(), cmdclass={"build_ext": BuildRust})
