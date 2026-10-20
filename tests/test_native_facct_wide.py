"""discrete_gaussian_facct on the native backend returns the reference's integers.

Known answers taken from the Python reference before the native dispatch, at
the widths of SIFE FullySec Ring-LWE (keys and noise of eta 1 / 4 / 10, both
layouts) and of the tMIFE-LWE / tMCFE-Ring-LWE flooding; agreement with the
Python code on XofRandom streams and on the same OS-style bytes; the
XofRandom counter after a native draw, and a thread sharing it.
"""

import hashlib
import os
import random
import threading

import gmpy2 as gp
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl

from pyfe4ai.primitives import sampling as su

pytestmark = pytest.mark.skipif(
    native_backend.ext() is None or not hasattr(native_backend.ext(), "facct_draw"),
    reason="Rust extension with the FACCT dispatch not built",
)

# sigma1 / sigma2 / sigma3 of sife/fullysec_ring_lwe (standard profile; eta 1: X 100, Y 1;
# eta 4: X = Y = 4; eta 10: X = Y = 50; elementwise and packed layouts) and sigma_f of
# tMIFE-LWE (toy / standard) and tMCFE-Ring-LWE (standard, and n 10 / eta 4 / X = Y = 100)
WIDTHS = {
    'fullysec_eta1_elem_sigma1': 476269688.0213009,
    'fullysec_eta1_elem_sigma2': 453906837974878.75,
    'fullysec_eta1_elem_sigma3': 1978534036514586.5,
    'fullysec_eta1_pack_sigma1': 154186239.90175474,
    'fullysec_eta1_pack_sigma2': 146946552285963.78,
    'fullysec_eta1_pack_sigma3': 509037789112732.56,
    'fullysec_eta4_elem_sigma1': 93052262.90505761,
    'fullysec_eta4_elem_sigma2': 125999223627111.6,
    'fullysec_eta4_elem_sigma3': 519508107760402.0,
    'fullysec_eta4_pack_sigma1': 137987192.89236647,
    'fullysec_eta4_pack_sigma2': 186844238196143.03,
    'fullysec_eta4_pack_sigma3': 792712967124762.6,
    'fullysec_eta10_elem_sigma1': 9.766867262245419e+21,
    'fullysec_eta10_elem_sigma2': 3.7683898996357106e+28,
    'fullysec_eta10_elem_sigma3': 1.0658616208749218e+29,
    'fullysec_eta10_pack_sigma1': 1.1876072353530993e+20,
    'fullysec_eta10_pack_sigma2': 4.582193031064149e+26,
    'fullysec_eta10_pack_sigma3': 1.3746579093192497e+27,
    'tmife_toy': 414933754069760176222717870080,
    'tmife_std': 41547121354216193517533187480826480086738402952552749465600,
    'tmcfe_std': 447376781648190700506077024983381091686950567609395922862080,
    'tmcfe_wide': 186834914345066495681357996796918057666081031937484318459297792000,
}
KAT = {
    'fullysec_eta1_elem_sigma1': [-92107988, 178507553, 939114358, 100381402, -148565210, 62458618],
    'fullysec_eta1_elem_sigma2': [295443816991946, -447794098211790, -131687560795088, 427249621407702, -297467416605417, 337393388869541],
    'fullysec_eta1_elem_sigma3': [-386328903140974, 748714946597637, 3916459270158027, 421030118310897, -623127654779362, 261970431459534],
    'fullysec_eta1_pack_sigma1': [26020494, 226167313, -40330338, 25095350, -37141302, -150728674],
    'fullysec_eta1_pack_sigma2': [27284466455116, 219811027597940, -42289424553198, 26314382394431, -38945478423710, -158050470093932],
    'fullysec_eta1_pack_sigma3': [109137865820466, 786524931389859, 996074561159888, 105257529577724, -155781913694840, 65492607864883],
    'fullysec_eta4_elem_sigma1': [13010247, 131874111, -20165169, 12547675, -18570651, -75364337],
    'fullysec_eta4_elem_sigma2': [27284466455116, 195147432890082, 247534840332589, 26314382394431, -38945478423710, 16373151966220],
    'fullysec_eta4_elem_sigma3': [109137865820466, 798852789504556, 1008402419274585, 105257529577724, -155781913694840, 65492607864883],
    'fullysec_eta4_pack_sigma1': [26020494, 207094392, -40330338, 25095350, -37141302, -150728674],
    'fullysec_eta4_pack_sigma2': [27284466455116, 266786962863760, -42289424553198, 26314382394431, -38945478423710, -158050470093932],
    'fullysec_eta4_pack_sigma3': [590887633983892, -895588196423580, -263375121590177, 854499242815404, -594934833210834, 674786777739082],
    'fullysec_eta10_elem_sigma1': [1831029548648985608858, 14639945498520288438756, -2837995240978839039434, 1765928309351872384108, -2613586814951699276589, -10606587502669757031968],
    'fullysec_eta10_elem_sigma2': [7679894560016634935175777929, 57540933000552324247119503761, 72286677290363709638957857332, 7406840171627795748157457755, -10962177632299172082595387858, 4608632568372069833696166086],
    'fullysec_eta10_elem_sigma3': [15359789120033269870351555858, 151838680772833458819872231009, -23806829582437016996912437637, 14813680343255591496314915511, -21924355264598344165190775716, -88974504777595545196425853593],
    'fullysec_eta10_pack_sigma1': [77448823961536706319, -117386536081631520478, -34521103937067770906, 112000924754300707369, -77979298458610548799, 88445652531816956853],
    'fullysec_eta10_pack_sigma2': [324843912137169253462588068, -492354817833331412869403563, -144792004327659099783677336, 469765926700662474122251501, -327068883442144059269987193, 370967954196809989400465903],
    'fullysec_eta10_pack_sigma3': [239996705000519841724243060, 2030146394799547790593778936, -371981712225578390576756838, 231463755363368617129920554, -342568051009349127581105870, -1390226637149930393694153962],
    'tmife_toy': [61439156480133079481406223433, 593919421799951805522006091917, -95227318329748067987649750551, 59254721373022365985259662046, -87697421058393376660763102867, -355898019110382180785703414372],
    'tmife_std': [26354469132874804358420199877980039423250945268931582180364, -39944568342507719098935469556583300753440413258544298285088, -11746922956429148603228681133596663803052581285628940052630, 38111939772727372928831547780127318602387389983034094108266, -26534980250330285922318742235122727514995472268764648098689, 30096495987393567699858470074794984972101012895817518347340],
    'tmcfe_std': [77883383589556808196375355068273456933885540043521316600476, 660321116556539834970348128538788214168738172335193174034379, -120714967238829810101919435168462682463805158225945137914561, 75114283114868295649452803936764759166636288906289267962658, -111169688443140128881003840328453928648846256392738805986320, -451154337545313829597730169532687401725226304171714558563557],
    'tmcfe_wide': [-36135700767282289558350897636988262502900970905209089584763516182, 70031879702180275285624696251135901203759805424375379939783221228, 368415004922693225233148358243302252686495757818884178774351860898, 39381517265728068989460311670398522053957406638100587721606313571, -58284933614477051890763741438124453343446306071636243112956050365, 24503693728665182617362463015705268808582167258007695855405096064],
}

KAT_SEED = b"pyfe4ai/discrete_gaussian_facct/wide-kat"


def _k(sigma):
    return su._facct_params(su._exact(sigma)).k


def _native_widths():
    max_k = getattr(native_impl, "MAX_NATIVE_FACCT_WIDE_K", native_impl.MAX_NATIVE_FACCT_K)
    return [name for name, s in WIDTHS.items() if _k(s) <= max_k]


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    with native_backend.backend("auto"):
        yield


class _Spy:
    """The extension, counting facct_draw / facct_draw_wide calls."""

    def __init__(self, ext) -> None:
        self._ext, self.calls, self.wide = ext, 0, 0

    def __getattr__(self, name):
        return getattr(self._ext, name)

    def facct_draw(self, *args, **kwargs):
        self.calls += 1
        return self._ext.facct_draw(*args, **kwargs)

    def facct_draw_wide(self, *args, **kwargs):
        self.calls += 1
        self.wide += 1
        return self._ext.facct_draw_wide(*args, **kwargs)


@pytest.fixture
def spy(monkeypatch):
    s = _Spy(native_backend.ext())
    monkeypatch.setattr(native_backend, "_ext", s)
    return s


class _StreamBits:
    """_OSRandomBuffer.getrandbits over a byte string."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getrandbits(self, k: int) -> int:
        n = (k + 7) // 8
        block = self.data[self.pos:self.pos + n]
        self.pos += n
        return int.from_bytes(block, "big") >> (8 * n - k)


def _both(fn):
    out = []
    for name in ("python", "native"):
        with native_backend.backend(name):
            out.append([int(v) for v in fn()])
    return out


@pytest.mark.parametrize("name", list(KAT))
def test_known_answers(name, spy):
    for backend in ("python", "native"):
        with native_backend.backend(backend):
            got = su.discrete_gaussian_facct(WIDTHS[name], 6, rng=su.XofRandom(KAT_SEED))
        assert [int(v) for v in got] == KAT[name] and all(type(v) is type(gp.mpz(0)) for v in got)
    assert spy.calls == (name in _native_widths())
    assert spy.wide == (_k(WIDTHS[name]) > native_impl.MAX_NATIVE_FACCT_K and name in _native_widths())


@pytest.mark.parametrize("name", list(WIDTHS))
def test_xof_samples_and_counter_identical(name):
    sigma = WIDTHS[name]

    def run():
        rng = su.XofRandom(name.encode() * 3)
        out = su.discrete_gaussian_facct(sigma, 700, rng=rng)
        out += [su.discrete_gaussian_facct(sigma, 1, rng=rng)[0] for _ in range(3)]
        return out + [rng.getrandbits(64)]  # the stream continues where the reference leaves it

    py, nat = _both(run)
    assert py == nat


@pytest.mark.parametrize("name", _native_widths())
def test_os_path_on_the_same_bytes(name, monkeypatch):
    """rng=None: the native sampler reads the bytes as _OSRandomBuffer reads the OS."""
    data = random.Random(name).randbytes(512 * 300 + 4096)
    with native_backend.backend("python"):
        monkeypatch.setattr(su, "_OS_RANDOM", _StreamBits(data))
        py = [int(v) for v in su.discrete_gaussian_facct(WIDTHS[name], 300)]
        monkeypatch.setattr(su, "_OS_RANDOM", su._OS_SOURCE)
    with native_backend.backend("native"):
        monkeypatch.setattr(native_backend, "_random_override", lambda n: data[:n])
        nat = [int(v) for v in su.discrete_gaussian_facct(WIDTHS[name], 300)]
    assert py == nat


def test_matrix_above_2_40_samples_natively(spy):
    sigma = WIDTHS["fullysec_eta4_elem_sigma2"]
    rows = su.discrete_gaussian_matrix(3, 50, sigma, sampler="facct")
    assert spy.calls == 1 and len(rows) == 3 and all(len(r) == 50 for r in rows)


def test_drbg_streams_do_not_depend_on_the_threads(monkeypatch):
    monkeypatch.setattr(native_backend, "_seed_override", bytes(range(32)))
    sigma = WIDTHS["fullysec_eta4_pack_sigma3"]
    out = {}
    for threads in (1, 3):
        monkeypatch.setattr(native_backend, "_threads", threads)
        out[threads] = su.discrete_gaussian_facct(sigma, 1000)
    assert out[1] == out[3]
    monkeypatch.setattr(native_backend, "_seed_override", bytes(range(1, 33)))
    assert su.discrete_gaussian_facct(sigma, 1000) != out[1]


def test_os_path_distribution():
    sigma = 2.0 ** 45
    v = [int(x) for x in su.discrete_gaussian_facct(sigma, 20000)]
    assert abs(sum(v) / len(v)) < 4 * sigma / len(v) ** 0.5
    assert abs(sum(x * x for x in v) / len(v) / sigma ** 2 - 1) < 0.05


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_shared_xof_random_never_draws_a_block_twice():
    """A thread drawing from the XofRandom of a native draw waits for its end and
    gets other blocks (it does not fail), and the native samples are the
    reference's from the counter at which the draw ran."""
    seed, count = b"\x04" * 32, 3000
    sigma = WIDTHS["fullysec_eta4_elem_sigma2"]
    rng = su.XofRandom(seed)
    stop, outs = threading.Event(), []

    def other():
        while not stop.is_set():
            outs.append(rng.getrandbits(64))

    t = threading.Thread(target=other)
    with native_backend.backend("native"):
        t.start()
        native = su.discrete_gaussian_facct(sigma, count, rng=rng)
        stop.set()
        t.join()
    head = len(seed).to_bytes(8, "big") + seed
    block = {int.from_bytes(hashlib.shake_256(head + c.to_bytes(8, "big")).digest(8), "big"): c
             for c in range(rng._counter)}
    used = [block[v] for v in outs]
    assert len(set(used)) == len(used)
    free = sorted(set(range(rng._counter)) - set(used))
    assert free == list(range(free[0], free[-1] + 1))  # one run of calls: the native draw
    ref = su.XofRandom(seed)
    ref._counter = free[0]
    with native_backend.backend("python"):
        assert su.discrete_gaussian_facct(sigma, count, rng=ref) == native
    assert ref._counter == free[-1] + 1


def test_python_path_where_the_native_one_does_not_apply(spy, monkeypatch):
    sigma = WIDTHS["fullysec_eta4_elem_sigma1"]
    su.discrete_gaussian_facct(sigma, 3, rng=random.Random(1))  # another bit source
    with native_backend.backend("python"):
        su.discrete_gaussian_facct(sigma, 3)
    monkeypatch.setattr(su, "_OS_RANDOM", _StreamBits(bytes(4096)))  # a replaced OS source
    su.discrete_gaussian_facct(sigma, 3)
    assert spy.calls == 0
    monkeypatch.setattr(su, "_OS_RANDOM", su._OS_SOURCE)
    su.discrete_gaussian_facct(sigma, 3)
    assert spy.calls == 1
    # the MPFR context does not matter: the reference sets its own
    with gp.context(gp.get_context(), round=gp.RoundDown, emax=1000):
        py, nat = _both(lambda: su.discrete_gaussian_facct(sigma, 20, rng=su.XofRandom(b"rd")))
    assert py == nat and spy.calls == 2


# -- the wide kernel (k > 2^50): exact fallback, fixed-point decisions, table checks ---------


WIDE = [name for name, s in WIDTHS.items() if _k(s) > native_impl.MAX_NATIVE_FACCT_K]


def _wide_table(sigma, counter, c_exp_shift=0, mantissa_bits=128):
    """The wide table of _facct_wide_table with a counting exact threshold (and
    optionally a useless or an imprecise c); the counter is cleared after the
    table's own check at construction."""
    p = su._facct_params(su._exact(sigma))
    mant, exp = p.neg_c.as_mantissa_exp()
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - mantissa_bits
    mant = mant >> shift if shift > 0 else mant << -shift
    mant, exp = mant << (128 - mantissa_bits), exp + shift - (128 - mantissa_bits)

    def exact(x, y):
        counter.append((x, y))
        with gp.context(precision=su.FACCT_PRECISION + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(p.neg_c * (y * (y + 2 * p.k * x))) * (1 << su.FACCT_PRECISION)))

    table = native_backend.ext().FacctWideTable(p.k, p.y_bits, list(p.cdt), mant, exp + c_exp_shift, exact)
    counter.clear()
    return table


@pytest.mark.parametrize("name", ["fullysec_eta1_elem_sigma3", "fullysec_eta10_elem_sigma2", "tmcfe_wide"])
def test_wide_fixed_point_test_decides_alone(name):
    """The exact Python threshold is for about one test in 2^99: never here."""
    calls = []
    native_backend.ext().facct_draw_wide(_wide_table(WIDTHS[name], calls), 20000)
    assert calls == []


@pytest.mark.parametrize("name", ["fullysec_eta1_elem_sigma3", "fullysec_eta10_pack_sigma1", "tmife_std"])
def test_wide_exact_fallback_gives_the_same_samples(name):
    """A table whose fixed-point test always defers: every decision is the Python threshold's."""
    calls = []
    head = su.XofRandom(b"fallback/" + name.encode())._seed
    nat, _ = native_backend.ext().facct_draw_wide(_wide_table(WIDTHS[name], calls, c_exp_shift=400), 300,
                                                  xof_head=head, xof_counter=0)
    assert calls  # deferred
    with native_backend.backend("python"):
        ref = su.discrete_gaussian_facct(WIDTHS[name], 300, rng=su.XofRandom(b"fallback/" + name.encode()))
    assert nat == [int(v) for v in ref]


@pytest.mark.parametrize("name", WIDE[:2] + ["tmcfe_std"])
def test_wide_tables_with_a_53_bit_c_are_refused(name):
    with pytest.raises(ValueError, match="disagrees"):
        _wide_table(WIDTHS[name], [], mantissa_bits=53)


def test_wide_tables_are_checked():
    p = su._facct_params(su._exact(WIDTHS["tmcfe_std"]))
    ext = native_backend.ext()
    for k, y_bits in ((p.k, p.y_bits + 1), (2 ** 40, 40), (2 ** 448, 448)):
        with pytest.raises(ValueError, match="unsupported"):
            ext.FacctWideTable(k, y_bits, list(p.cdt), 1 << 127, -600, lambda x, y: 0)


@pytest.mark.parametrize("name", ["fullysec_eta10_elem_sigma1", "fullysec_eta10_pack_sigma3", "tmife_std",
                                  "tmcfe_wide"])
def test_wide_xof_agreement_at_scale(name):
    py, nat = _both(lambda: su.discrete_gaussian_facct(WIDTHS[name], 5000, rng=su.XofRandom(b"scale" + name.encode())))
    assert py == nat


# -- October 2026 review of the wide native FACCT: counters near 2^64, re-entrant draws ---


def test_counter_near_2_64_behaves_as_the_reference():
    """The reference raises OverflowError at call 2^64; the native draw must not
    wrap to call 0 (it leaves such counters to the Python code)."""
    for backend in ("python", "native"):
        rng = su.XofRandom(b"\x05" * 32)
        rng._counter = 2 ** 64 - 3
        with native_backend.backend(backend), pytest.raises(OverflowError):
            su.discrete_gaussian_facct(gp.mpz(2) ** 100, 2, rng=rng)
        assert rng._counter >= 2 ** 64 - 3


def test_a_busy_xof_random_refuses_other_draws():
    rng = su.XofRandom(b"\x06" * 32)
    seen = []

    def draw(start):
        with pytest.raises(RuntimeError, match="being drawn"):
            rng.getrandbits(8)
        seen.append(start)
        return "result", 5

    assert rng._claim_dynamic(draw) == "result" and seen == [0] and rng._counter == 5

    def failing(start):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        rng._claim_dynamic(failing)
    assert rng._counter == 5 and rng._busy is None  # released, nothing consumed
    rng.getrandbits(8)
    assert rng._counter == 6


_SIGNAL = """
import signal, sys, time
from pyfe4ai import backend as nb
from pyfe4ai.primitives import sampling as su
other, rng = su.XofRandom(b"o" * 32), su.XofRandom(b"r" * 32)
signal.signal(signal.SIGALRM, lambda *args: other.getrandbits(8))
signal.setitimer(signal.ITIMER_REAL, 0.0005, 0.0005)
t0 = time.time()
with nb.backend("native"):
    while time.time() - t0 < 1.5:
        su.discrete_gaussian_facct(1e30, 1, rng=rng)
signal.setitimer(signal.ITIMER_REAL, 0)
print("ok", flush=True)
"""


@pytest.mark.skipif(not hasattr(__import__("signal"), "setitimer"), reason="needs setitimer")
def test_a_signal_handler_drawing_from_another_xof_random_does_not_hang():
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        res = subprocess.run([sys.executable, "-c", _SIGNAL], capture_output=True, text=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("a signal handler drawing from an XofRandom hung")
    assert res.returncode == 0 and res.stdout.strip() == "ok", (res.returncode, res.stderr[-300:])
