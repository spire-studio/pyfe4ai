"""``pyfe4ai.backend_info()`` and the missing-extension notice.

The notice is given once per process, so its tests run fresh interpreters in
which the extension cannot be imported (``sys.modules["pyfe4ai._rust_native"]
= None`` before pyfe4ai is imported), whether or not it is built here. In this
test session tests/conftest.py has given it already (silently), so that it
does not land in another test.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives import sampling

ROOT = Path(__file__).resolve().parents[2]
KEYS = {"backend", "selected", "native_available", "native_version", "native_build", "native_path", "import_error",
        "threads", "max_threads", "gaussian_sampler", "scheme_sampler", "cache_dir", "version"}


# ------------------------------------------------------------------ backend_info()

def test_backend_info_reports_the_selection_of_this_thread():
    info = pyfe4ai.backend_info()
    assert set(info) == KEYS and info == native_backend.backend_info()
    json.dumps(info)  # plain values
    assert info["backend"] == native_backend.active_backend()
    assert info["selected"] == native_backend.get_backend()
    assert info["native_available"] is native_backend.available()
    assert (info["threads"], info["max_threads"]) == (native_backend.get_threads(), native_backend.MAX_THREADS)
    assert info["gaussian_sampler"] == sampling.get_gaussian_sampler()
    assert info["scheme_sampler"] == sampling.PROOF_SAMPLER == "facct"
    assert info["cache_dir"] == os.environ["PYFE4AI_CACHE_DIR"]  # tests/conftest.py: the run's cache
    assert info["version"] == pyfe4ai.__version__
    with native_backend.backend("python"):
        inner = pyfe4ai.backend_info()
        assert (inner["backend"], inner["selected"]) == ("python", "python")
        assert inner["native_available"] is info["native_available"]
    with sampling.gaussian_sampler("facct"):
        assert pyfe4ai.backend_info()["gaussian_sampler"] == "facct"


def test_backend_info_describes_the_extension():
    info = pyfe4ai.backend_info()
    if native_backend.available():
        assert info["backend"] in ("native", "python") and info["import_error"] is None
        assert re.fullmatch(r"\d+\.\d+\.\d+\S*", info["native_version"]), info["native_version"]
        assert info["native_build"] == native_backend.ext().build_info()
        assert "pyfe4ai-native " + info["native_version"] in info["native_build"]["compiler"]
        assert Path(info["native_path"]).name.startswith("_rust_native")
    else:  # pragma: no cover - depends on the build
        assert info["backend"] == "python" and isinstance(info["import_error"], str)
        assert info["native_version"] is info["native_build"] is info["native_path"] is None


def test_the_session_has_given_the_notice():
    assert native_backend._NATIVE_NOTICE == {}


def _python(code: str, tmp_path, block: bool = True, **env) -> subprocess.CompletedProcess:
    """*code* in a fresh interpreter (the extension made unimportable if *block*)."""
    prelude = "import sys\n" + ("sys.modules['pyfe4ai._rust_native'] = None\n" if block else "\n")
    environ = {k: v for k, v in os.environ.items() if k not in ("PYFE4AI_BACKEND", "PYTHONWARNINGS")}
    environ.update(PYTHONPATH=str(ROOT), PYFE4AI_CACHE_DIR=str(tmp_path / "cache"), **env)
    return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], capture_output=True, text=True,
                          env=environ, cwd=str(tmp_path), timeout=300)


def test_importing_pyfe4ai_loads_neither_the_backend_nor_the_extension(tmp_path):
    res = _python("""
        import pyfe4ai
        from pyfe4ai import backend_info
        print('pyfe4ai.backend' in sys.modules, 'pyfe4ai._rust_native' in sys.modules)
        info = backend_info()
        print('pyfe4ai.backend' in sys.modules, info['backend'], info['native_available'] == ('pyfe4ai._rust_native' in sys.modules))
    """, tmp_path, block=False)
    assert res.returncode == 0, res.stderr
    first, second = res.stdout.split("\n")[:2]
    assert first == "False False"
    assert second.startswith("True ") and second.endswith(" True")


# ------------------------------------------------------------------ the notice

#: A round of the MCFE over Ring-LWE (small): its first operation, kg.setup(), is at line 9 (11 after the
#: prelude of _python).
_ROUND = """
    import warnings
    warnings.simplefilter("always")
    import numpy as np
    import pyfe4ai
    import pyfe4ai.schemes.mcfe, pyfe4ai.schemes.quadratic  # the pairing modules load their kernels: no notice
    from pyfe4ai.primitives.pairing import bls12_381
    kg, MCFE = pyfe4ai.create("mcfe/ring_lwe", n=2, lst_nid=["a", "b"], eta=1, ring_n=16)
    kg.setup()
    pp = kg.get_public_parameters()
    for i in "ab":
        c = MCFE({"id": i, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(i)}})
        c.encrypt_lst_ndarray([np.arange(3)], label="L")
    print("pairing", bls12_381.available(), "backend", pyfe4ai.backend_info()["backend"])
"""


def _notices(stderr: str) -> list:
    return [line for line in stderr.splitlines() if "FEWarning: the native extension" in line]


@pytest.mark.parametrize("env", [{}, {"PYFE4AI_BACKEND": "native"}], ids=["auto", "env-native"])
def test_one_notice_at_the_first_fallback(env, tmp_path):
    res = _python(_ROUND, tmp_path, **env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "pairing False backend python"
    notices = _notices(res.stderr)
    assert len(notices) == 1, res.stderr
    # attributed to the application's first operation (kg.setup(), line 11 with the two-line prelude)
    assert notices[0].startswith("<string>:11: FEWarning: " + native_backend.MISSING_EXTENSION_NOTICE)


def test_no_notice_when_python_is_chosen(tmp_path):
    res = _python(_ROUND, tmp_path, PYFE4AI_BACKEND="python")
    assert res.returncode == 0, res.stderr
    assert _notices(res.stderr) == [] and "FEWarning" not in res.stderr
    res = _python("""
        import warnings
        warnings.simplefilter("always")
        from pyfe4ai import backend
        backend.set_backend("python")
        backend.ops(); backend.ring(17, 4)
        backend.set_backend("auto")
        with backend.backend("python"):
            backend.ops()
        print("quiet so far")
        backend.ops()
    """, tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "quiet so far" and len(_notices(res.stderr)) == 1


def test_the_notice_can_be_filtered(tmp_path):
    res = _python("""
        import warnings
        from pyfe4ai import FEWarning
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="the native extension", category=FEWarning)
        from pyfe4ai import backend
        backend.ops()
        print("pending", backend._NATIVE_NOTICE)
    """, tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "pending {}" and "FEWarning" not in res.stderr


def test_native_selection_without_the_extension(tmp_path):
    res = _python("""
        from pyfe4ai import backend
        print(backend.get_backend(), backend.active_backend())
        for select in (lambda: backend.set_backend("native"), lambda: backend.backend("native").__enter__()):
            try:
                select()
            except ImportError as exc:
                print("ImportError", "not built" in str(exc))
    """, tmp_path, PYFE4AI_BACKEND="native")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split("\n")[:3] == ["native python", "ImportError True", "ImportError True"]


def test_one_notice_for_several_threads(tmp_path):
    res = _python("""
        import threading, warnings
        warnings.simplefilter("always")
        from pyfe4ai import backend
        start = threading.Barrier(8)
        def run():
            start.wait()
            for _ in range(200):
                backend.ops()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            threads = [threading.Thread(target=run) for _ in range(8)]
            for t in threads: t.start()
            for t in threads: t.join()
        print(sum("the native extension" in str(w.message) for w in seen))
    """, tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_with_the_extension_no_notice(tmp_path):
    res = _python("""
        import warnings
        warnings.simplefilter("always")
        from pyfe4ai import backend
        backend.ops(); backend.ring(17, 4)
        with backend.backend("python"):
            backend.ops()
        print(backend.available(), backend._NATIVE_NOTICE)
    """, tmp_path, block=False)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "True {'pending': True}" and "FEWarning" not in res.stderr


def test_the_notice_text():
    """No paths or caller values; and no file a wheel does not ship (benchmarks/run.py, tools/build_rust.py):
    it points to building from source and the installation documentation."""
    text = native_backend.MISSING_EXTENSION_NOTICE
    for part in ("pyfe4ai._rust_native", "5 to 35 times slower", "from source with a Rust toolchain",
                 "Installation in the documentation", "PYFE4AI_BACKEND=python", "warnings filter"):
        assert part in text, part
    assert os.sep not in text and "/" not in text and ".py" not in text
