"""Build of the optional native extension; the package metadata is in pyproject.toml.

``pyfe4ai._rlwe_native`` (C++, CPython API only) speeds up the Ring-LWE schemes;
see :mod:`pyfe4ai.utils.native_backend`. It needs a 64-bit GCC or Clang
(``unsigned __int128``). The extension is optional: if it does not build, the
package installs without it and runs the Python implementation. Set
``PYFE4AI_NO_NATIVE=1`` to skip it. In a source checkout::

    python setup.py build_ext --inplace
"""

import os
import sys

from setuptools import Extension, setup


def _extensions() -> list:
    if os.environ.get("PYFE4AI_NO_NATIVE") or sys.platform == "win32":
        return []
    compile_args = [
        "-std=c++17", "-O3", "-fvisibility=hidden",
        # Box-Muller must round exactly as CPython's random.gauss: no FMA contraction
        "-ffp-contract=off", "-fno-fast-math",
    ]
    link_args = []
    if sys.platform != "darwin":
        compile_args.append("-pthread")
        link_args.append("-pthread")
    return [
        Extension(
            "pyfe4ai._rlwe_native",
            sources=["pyfe4ai/_native/rlwe_native.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        )
    ]


setup(ext_modules=_extensions())
