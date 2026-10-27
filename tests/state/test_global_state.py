"""Module-level state of the library: an allow-list.

The library keeps state that outlives a call in the objects that own it
(key generators, crypto objects, servers, precomputations), and process-wide
only what every object shares: the backend and sampler selection, public
tables derived from parameters (ring and NTT contexts, sampler tables,
discrete-log tables), the process-wide locks of the threshold stores, the
OS random buffer and the registries filled at import. All of it is listed on
``docs/threading.rst`` ("Process-wide state").

This test scans ``pyfe4ai/`` (its syntax tree, nothing is imported) for
module- and class-level state and fails on anything not in :data:`ALLOWED`:
a new cache, lock, registry, lazily set global or ``os.register_at_fork``
call -- also state made dynamically: through ``globals()`` or ``vars()``,
``exec`` / ``eval``, ``setattr`` on ``sys.modules[...]`` (or an attribute
assigned there, other than the module's own dunders such as ``__class__``),
an attribute assigned on another module (``module.NAME = value``), on a
function (``f.cache = ...``) or on a class through an instance
(``type(self).x = ...``), a mutable default argument, a cache decorator
imported under another name, ``getattr(os, "register_at_fork")``. The scan
reads the syntax only: state in closures, a module object passed through a
call (as :func:`pyfe4ai.core.forksafe.renew_lock` receives one) or a name
computed at run time is not followed. Put a new cache on the object that owns it; if it must be
process-wide, add it here with the reason and to the documentation page
(and register its fork handling with :mod:`pyfe4ai.core.forksafe`). An
allow-listed name that is gone fails too, so that the list stays current.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "pyfe4ai"
DOCS = ROOT / "docs" / "threading.rst"
FORKSAFE = "pyfe4ai/core/forksafe.py"

#: Calls that make a mutable container.
CONTAINERS = {"dict", "list", "set", "OrderedDict", "defaultdict", "deque", "bytearray", "Counter", "ChainMap"}
#: Calls that make a lock, per-thread or per-context state, a weak container,
#: a counter or a random generator.
STATEFUL = {"Lock", "RLock", "Condition", "Semaphore", "BoundedSemaphore", "Event", "Barrier", "local",
            "WeakSet", "WeakValueDictionary", "WeakKeyDictionary", "ContextVar", "count", "Random",
            "SystemRandom", "Queue", "SimpleQueue", "LifoQueue", "PriorityQueue"}
#: Library classes whose instances are immutable declarations (option and
#: record-field descriptions), made at class or module level by design.
DECLARATIVE = {"Option", "Field"}
#: Methods that mutate a container.
MUTATORS = {"append", "extend", "add", "update", "setdefault", "pop", "popitem", "clear", "insert", "remove",
            "discard", "move_to_end", "__setitem__", "__delitem__"}
CACHE_DECORATORS = {"lru_cache", "cache"}

#: (module path, name) -> why it is module- or class-level. Names: a module
#: attribute, ``Class.attribute`` or a cached function.
ALLOWED = {
    # the backend and its selection (pyfe4ai.backend)
    ("pyfe4ai/backend/__init__.py", "_state"): "per-thread backend selection (backend() blocks)",
    ("pyfe4ai/backend/__init__.py", "_default_backend"): "process-wide backend selection (set_backend)",
    ("pyfe4ai/backend/__init__.py", "_threads"): "process-wide threads per native call (set_threads)",
    ("pyfe4ai/backend/__init__.py", "_random_override"): "test hook: the native calls' random bytes",
    ("pyfe4ai/backend/__init__.py", "_seed_override"): "test hook: a fixed DRBG seed",
    ("pyfe4ai/backend/__init__.py", "_PYTHON"): "the Python implementation (its rings per (q, n))",
    ("pyfe4ai/backend/__init__.py", "_NATIVE"): "the native implementation (its rings per (q, n))",
    ("pyfe4ai/backend/__init__.py", "_NATIVE_NOTICE"): "the missing-extension notice is still due (once per process)",
    ("pyfe4ai/backend/native_impl.py", "_contexts"): "native ring contexts per (q, n, factors): public tables",
    ("pyfe4ai/backend/native_impl.py", "_contexts_lock"): "guards _contexts; renewed in a forked child",
    ("pyfe4ai/backend/native_impl.py", "_facct_table"): "native FACCT tables per width: public",
    ("pyfe4ai/backend/native_impl.py", "_facct_wide_table"): "native wide FACCT tables per width: public",
    ("pyfe4ai/backend/native_impl.py", "_bigint_width"): "mantissa and exponent of a width: public",
    ("pyfe4ai/backend/native_impl.py", "_ec_group"): "the ec_group module, imported on first use",
    ("pyfe4ai/backend/native_impl.py", "_bls_kernels"): "the extension's BLS12-381 functions, made on first use",
    # sampling
    ("pyfe4ai/primitives/sampling.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/primitives/sampling.py", "_default_sampler"): "process-wide Gaussian sampler (set_gaussian_sampler)",
    ("pyfe4ai/primitives/sampling.py", "_sampler_override"): "per-context Gaussian sampler (gaussian_sampler blocks)",
    ("pyfe4ai/primitives/sampling.py", "_XOF_LOCK"): "guards the counter claims of every XofRandom; renewed in a child",
    ("pyfe4ai/primitives/sampling.py", "_OS_RANDOM"): "buffered OS random bytes; dropped in a forked child",
    ("pyfe4ai/primitives/sampling.py", "_facct_params"): "FACCT tables per width: public",
    # lattices
    ("pyfe4ai/primitives/lattice/ring_lwe_utils.py", "_twiddle_cache"): "NTT tables per (q, n, root): public",
    ("pyfe4ai/primitives/lattice/ring_lwe_utils.py", "_ntt_root_cache"): "NTT roots per (q, n): public",
    ("pyfe4ai/primitives/lattice/ring_lwe_utils.py", "_rns_factors"): "registry of RNS moduli (q -> factors): public",
    ("pyfe4ai/primitives/lattice/ring_lwe_utils.py", "RingNTT._cache"): "NTT rings per (q, n): public",
    ("pyfe4ai/primitives/lattice/lwe_matrix.py", "SeededMatrix._shared"): "seeded public matrices, held weakly",
    # groups and pairings
    ("pyfe4ai/primitives/groups/ristretto255.py", "_BASE_COMB"): "base-point comb table, made on first use",
    ("pyfe4ai/primitives/groups/ec_group.py", "_dlog_table"): "baby-step tables per bound: public",
    ("pyfe4ai/primitives/groups/dlog_solver.py", "_cached_dlog_table"): "baby-step tables per group and bound",
    ("pyfe4ai/primitives/groups/integer_groups.py", "_small_primorial"): "a constant, computed on first use",
    ("pyfe4ai/primitives/groups/integer_groups.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/primitives/pairing/pairing_backend.py", "PAIRING_IMPORT_ERROR"): "set at import (charm missing)",
    ("pyfe4ai/primitives/pairing/pairing_backend.py", "_CharmPairingGroup"): "charm's class, set at import",
    ("pyfe4ai/primitives/pairing/pairing_backend.py", "_charm_pair"): "charm's pair, set at import",
    ("pyfe4ai/primitives/pairing/pairing_backend.py", "_CharmGroup"): "charm's group class, set at import",
    # schemes: the OS random source
    ("pyfe4ai/schemes/mife/ddh.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/schemes/quadratic/miqfe.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/schemes/quadratic/quad.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/schemes/sife/damgard_ddh.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/schemes/sife/part_fh_ipe_pairing.py", "_CSPRNG"): "the OS random source (no state of its own)",
    ("pyfe4ai/schemes/threshold/sharing.py", "_CSPRNG"): "the OS random source (no state of its own)",
    # threshold stores: the stores are per object, their locks process-wide
    ("pyfe4ai/schemes/threshold/store.py", "_SHARE_CACHE_LOCK"): "guards the share-sum caches; renewed in a child",
    ("pyfe4ai/schemes/threshold/store.py", "_STORE_LOCK"): "guards the weight-share stores; renewed in a child",
    # registries and class defaults
    ("pyfe4ai/core/config.py", "_CONFIGS"): "config classes per scheme, filled when the classes are defined",
    ("pyfe4ai/core/config.py", "STANDARD_FIXED"): "a constant (empty), never mutated",
    ("pyfe4ai/core/config.py", "SchemeConfig.OPTIONS"): "class default, replaced by every subclass",
    ("pyfe4ai/core/config.py", "SchemeConfig.CRYPTO"): "class default, replaced by every subclass",
    ("pyfe4ai/core/formats/record.py", "Record._fields"): "class default, replaced by every record class",
    # the fork handler itself
    ("pyfe4ai/core/forksafe.py", "_HANDLERS"): "what a forked child runs",
    ("pyfe4ai/core/forksafe.py", "_TRACKED"): "objects a forked child notifies, held weakly",
}


def _call_name(node) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _top(body):
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            for part in (node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])):
                yield from _top(part)
            for handler in getattr(node, "handlers", []):
                yield from _top(handler.body)


def _is_sys_modules(node, sys_names) -> bool:
    """``sys.modules[...]`` or ``sys.modules.get(...)``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        node = node.func.value
    elif isinstance(node, ast.Subscript):
        node = node.value
    else:
        return False
    return (isinstance(node, ast.Attribute) and node.attr == "modules" and isinstance(node.value, ast.Name)
            and node.value.id in sys_names)


def _const(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _dynamic(tree, path, hit, library_modules) -> None:
    """State made dynamically: ``globals()`` / ``vars()``, ``exec`` / ``eval``,
    ``setattr`` on ``sys.modules[...]`` or an attribute assigned there (not a
    dunder), an attribute assigned on another module (bound by ``import``, or
    by ``from package import module`` for a module of *library_modules*)."""
    sys_names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "sys":
                    sys_names.add(a.asname or "sys")
                else:
                    modules.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for a in node.names:
                if "{}.{}".format(node.module, a.name) in library_modules:
                    modules.add(a.asname or a.name)
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                if isinstance(default, (ast.Dict, ast.List, ast.Set)) or _call_name(default) in CONTAINERS | STATEFUL:
                    hit(path, getattr(node, "name", "<lambda>"), "mutable default argument")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if isinstance(node.func, ast.Name) and name in ("globals", "vars") and not node.args:
                hit(path, name + "()", "module namespace by " + name + "()")
            elif isinstance(node.func, ast.Name) and name in ("exec", "eval"):
                hit(path, name, "exec / eval")
            elif (isinstance(node.func, ast.Name) and name in ("setattr", "delattr") and len(node.args) >= 2
                  and _is_sys_modules(node.args[0], sys_names)):
                hit(path, _const(node.args[1]) or "<computed>", "set through sys.modules")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                if not isinstance(t, ast.Attribute):
                    continue
                if _is_sys_modules(t.value, sys_names):
                    if not (t.attr.startswith("__") and t.attr.endswith("__")):
                        hit(path, t.attr, "set through sys.modules")
                elif isinstance(t.value, ast.Name) and t.value.id in modules:
                    hit(path, t.value.id + "." + t.attr, "set on another module")
                elif isinstance(t.value, ast.Name) and t.value.id in functions:
                    hit(path, t.value.id + "." + t.attr, "function attribute")
                elif _call_name(t.value) == "type" or (isinstance(t.value, ast.Attribute) and t.value.attr == "__class__"):
                    hit(path, "<class>." + t.attr, "class attribute set through an instance")


def scan(sources: dict, library_classes: set | None = None, library_modules: set | None = None) -> tuple[dict, list]:
    """``(found, fork_calls)`` for ``{path: source}``: *found* maps
    ``(path, name)`` to the rules that matched; *fork_calls* lists the
    ``os.register_at_fork`` calls outside :data:`FORKSAFE`. *library_modules*:
    the dotted names of the modules (default: those of *sources*)."""
    trees = {path: ast.parse(src) for path, src in sources.items()}
    if library_modules is None:
        library_modules = {path[:-3].replace("/", ".").removesuffix(".__init__") for path in sources}
    if library_classes is None:
        library_classes = {n.name for t in trees.values() for n in ast.walk(t) if isinstance(n, ast.ClassDef)}
    found, fork_calls = {}, []

    def hit(path, name, rule):
        found.setdefault((path, name), set()).add(rule)

    for path, tree in trees.items():
        module_names = set()

        def bindings(body, prefix):
            for node in _top(body):
                if isinstance(node, ast.ClassDef):
                    bindings(node.body, prefix + node.name + ".")
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [n.id for t in targets for n in (t.elts if isinstance(t, ast.Tuple) else [t])
                         if isinstance(n, ast.Name)]
                if not prefix:
                    module_names.update(names)
                value, call = node.value, _call_name(node.value)
                rules = set()
                if isinstance(value, (ast.Dict, ast.List, ast.Set)) and not (
                        value.keys if isinstance(value, ast.Dict) else value.elts):
                    rules.add("empty container")
                if call in CONTAINERS:
                    rules.add("container")
                if call in STATEFUL:
                    rules.add("lock / thread / weak / random state")
                if call in library_classes and call not in DECLARATIVE:
                    rules.add("library object")
                if not prefix and isinstance(value, ast.Constant) and value.value is None:
                    rules.add("placeholder (None)")
                for name in names:
                    for rule in rules:
                        hit(path, prefix + name, rule)

        bindings(tree.body, "")
        _dynamic(tree, path, hit, library_modules)
        caches = CACHE_DECORATORS | {a.asname for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                                     and n.module == "functools" for a in n.names
                                     if a.name in CACHE_DECORATORS and a.asname}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and path != FORKSAFE:
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "register_at_fork":
                    fork_calls.append("{}:{}".format(path, node.lineno))
                elif _call_name(node) == "getattr" and len(node.args) >= 2 and _const(node.args[1]) == "register_at_fork":
                    fork_calls.append("{}:{}".format(path, node.lineno))
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                if name in caches:
                    hit(path, node.name, "cached function")
            for sub in ast.walk(node):
                if isinstance(sub, ast.Global):
                    for name in sub.names:
                        hit(path, name, "global rebinding")
                elif isinstance(sub, (ast.Assign, ast.AugAssign, ast.Delete)):
                    targets = sub.targets if not isinstance(sub, ast.AugAssign) else [sub.target]
                    for t in targets:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                                and t.value.id in module_names:
                            hit(path, t.value.id, "mutated in a function")
                elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr in MUTATORS and isinstance(sub.func.value, ast.Name) \
                        and sub.func.value.id in module_names:
                    hit(path, sub.func.value.id, "mutated in a function")
    return found, fork_calls


def _library_sources() -> dict:
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in sorted(PACKAGE.rglob("*.py"))}


@pytest.fixture(scope="module")
def library():
    return scan(_library_sources())


def test_no_module_level_state_outside_the_allow_list(library):
    found, _ = library
    new = sorted(set(found) - set(ALLOWED))
    assert not new, (
        "new module- or class-level state (move it into the object that owns it, or allow-list it here with "
        "the reason, document it in docs/threading.rst and register its fork handling with "
        "pyfe4ai.core.forksafe):\n" + "\n".join("{}: {} ({})".format(p, n, ", ".join(sorted(found[(p, n)])))
                                                   for p, n in new))


def test_every_allow_listed_name_still_exists(library):
    found, _ = library
    gone = sorted(set(ALLOWED) - set(found))
    assert not gone, "allow-listed but no longer found (remove it here and from docs/threading.rst): {}".format(gone)


def test_only_forksafe_registers_fork_handlers(library):
    _, fork_calls = library
    assert not fork_calls, "os.register_at_fork outside {} (use pyfe4ai.core.forksafe): {}".format(
        FORKSAFE, fork_calls)


@pytest.mark.skipif(not DOCS.exists(), reason="the documentation sources are not in this tree")
def test_the_documentation_lists_the_process_wide_state():
    text = DOCS.read_text(encoding="utf-8")
    missing = sorted("{}: {}".format(p, n) for p, n in ALLOWED if "``{}``".format(n) not in text)
    assert not missing, "process-wide state missing from docs/threading.rst: {}".format(missing)


# -- the scan itself


def _scan_one(source: str, path: str = "pyfe4ai/x.py", classes=("Holder", "Option")):
    return scan({path: textwrap.dedent(source)}, set(classes), {"pyfe4ai.x", "pyfe4ai.primitives.sampling"})


@pytest.mark.parametrize("source,name,rule", [
    ("_cache = {}\n", "_cache", "empty container"),
    ("_cache: dict = {}\n", "_cache", "empty container"),
    ("_seen = []\n", "_seen", "empty container"),
    ("from collections import OrderedDict\n_lru = OrderedDict()\n", "_lru", "container"),
    ("import collections\n_d = collections.defaultdict(list)\n", "_d", "container"),
    ("import threading\n_LOCK = threading.Lock()\n", "_LOCK", "lock / thread / weak / random state"),
    ("import threading\n_L = threading.RLock()\n", "_L", "lock / thread / weak / random state"),
    ("import threading\n_tls = threading.local()\n", "_tls", "lock / thread / weak / random state"),
    ("import weakref\n_live = weakref.WeakSet()\n", "_live", "lock / thread / weak / random state"),
    ("import itertools\n_ids = itertools.count()\n", "_ids", "lock / thread / weak / random state"),
    ("class Holder:\n    pass\n_one = Holder()\n", "_one", "library object"),
    ("_lazy = None\n", "_lazy", "placeholder (None)"),
    ("class K:\n    _cache = {}\n", "K._cache", "empty container"),
    ("import functools\n@functools.lru_cache(maxsize=8)\ndef table(n):\n    return n\n", "table", "cached function"),
    ("from functools import cache\n@cache\ndef table(n):\n    return n\n", "table", "cached function"),
    ("X = 1\ndef f():\n    global X\n    X = 2\n", "X", "global rebinding"),
    ("TABLE = {1: 2}\ndef f(k):\n    TABLE[k] = k\n", "TABLE", "mutated in a function"),
    ("TABLE = {1: 2}\ndef f(k):\n    del TABLE[k]\n", "TABLE", "mutated in a function"),
    ("TABLE = {1: 2}\ndef f(k):\n    TABLE.setdefault(k, k)\n", "TABLE", "mutated in a function"),
    ("NAMES = ['a']\ndef f(k):\n    NAMES.append(k)\n", "NAMES", "mutated in a function"),
    ("if True:\n    _c = {}\n", "_c", "empty container"),
])
def test_the_scan_finds_state(source, name, rule):
    found, _ = _scan_one(source)
    assert rule in found.get(("pyfe4ai/x.py", name), set()), found


@pytest.mark.parametrize("source", [
    "LIMIT = 16\n",
    "NAMES = ('a', 'b')\n",
    "TABLE = {1: 2}\n",
    "ALL = frozenset({1})\n",
    "class K:\n    x = None\n",  # a class default
    "OPT = Option('doc', default=1)\n",  # a declaration
    "def f():\n    cache = {}\n    cache[1] = 2\n    return cache\n",  # local state
    "class K:\n    def __init__(self):\n        self._cache = {}\n        self._lock = None\n",  # per object
])
def test_the_scan_ignores_constants_and_object_state(source):
    found, fork_calls = _scan_one(source)
    assert not found and not fork_calls, found


@pytest.mark.parametrize("source,name,rule", [
    ("def f(k):\n    globals()[k] = {}\n", "globals()", "module namespace by globals()"),
    ("globals().setdefault('_cache', {})\n", "globals()", "module namespace by globals()"),
    ("def f():\n    vars()['x'] = 1\n", "vars()", "module namespace by vars()"),
    ("exec('_cache = {}')\n", "exec", "exec / eval"),
    ("import sys\nsetattr(sys.modules[__name__], '_cache', {})\n", "_cache", "set through sys.modules"),
    ("import sys\nsetattr(sys.modules.get(__name__), name, {})\n", "<computed>", "set through sys.modules"),
    ("import sys as s\ns.modules[__name__]._lock = None\n", "_lock", "set through sys.modules"),
    ("import pyfe4ai.backend as b\ndef f():\n    b._threads = 4\n", "b._threads", "set on another module"),
    ("import os\nos.environ = {}\n", "os.environ", "set on another module"),
    ("from pyfe4ai.primitives import sampling\ndef f():\n    sampling._default_sampler = 'facct'\n",
     "sampling._default_sampler", "set on another module"),
    ("def f(x, _cache={}):\n    _cache[x] = 1\n", "f", "mutable default argument"),
    ("def f(x, *, _seen=set()):\n    _seen.add(x)\n", "f", "mutable default argument"),
    ("def f(x):\n    f.cache = getattr(f, 'cache', {})\n", "f.cache", "function attribute"),
    ("class C:\n    def f(self):\n        type(self)._cache = {}\n", "<class>._cache",
     "class attribute set through an instance"),
    ("class C:\n    def f(self):\n        self.__class__._cache = {}\n", "<class>._cache",
     "class attribute set through an instance"),
    ("from functools import lru_cache as memo\n@memo(None)\ndef f(x):\n    return x\n", "f", "cached function"),
])
def test_the_scan_finds_dynamic_state(source, name, rule):
    found, _ = _scan_one(source)
    assert rule in found.get(("pyfe4ai/x.py", name), set()), found


@pytest.mark.parametrize("source", [
    "import sys\nimport types\nclass _M(types.ModuleType):\n    pass\nsys.modules[__name__].__class__ = _M\n",
    "import sys\nfrom pyfe4ai.core import forksafe\nforksafe.renew_lock(sys.modules[__name__], '_LOCK')\n",
    "def f(self, record, key):\n    setattr(self, key, getattr(record, key))\n",
    "import os\nx = os.environ.get('X')\n",
    "def f(obj):\n    obj.attr = 1\n",  # an object, not a module
    "from pyfe4ai.primitives.sampling import XofRandom\nXofRandom.attr = 1\n",  # a class, not a module
    "def f(x, limit=16, names=('a',), table=None):\n    return x\n",  # immutable defaults
    "def f():\n    def g():\n        pass\n    g.attr = 1\n",  # a local function
    "from functools import partial as p\n@p\ndef f(x):\n    return x\n",  # another decorator
])
def test_the_scan_ignores_other_dynamic_code(source):
    found, fork_calls = _scan_one(source)
    assert not found and not fork_calls, found


def test_the_scan_finds_fork_handlers_outside_forksafe():
    source = "import os\nos.register_at_fork(after_in_child=print)\n"
    assert _scan_one(source)[1] == ["pyfe4ai/x.py:2"]
    assert _scan_one(source, path=FORKSAFE)[1] == []


def test_the_scan_finds_fork_handlers_registered_through_getattr():
    assert _scan_one("import os\ngetattr(os, 'register_at_fork')(after_in_child=print)\n")[1] == ["pyfe4ai/x.py:2"]
