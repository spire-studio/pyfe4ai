"""discrete_gaussian_facct on the native backend returns the reference's integers.

Known answers taken from the Python reference before the native dispatch, at
the widths of SIFE FullySec Ring-LWE (keys and noise of eta 1 / 4 / 10, both
layouts) and of the tMIFE-LWE / tMCFE-Ring-LWE flooding; agreement with the
Python code on XofRandom streams and on the same OS-style bytes; the
XofRandom counter after a native draw, and a thread sharing it.
"""

import hashlib
import random
import threading

import gmpy2 as gp
import pytest

from pyfe4ai.utils import native_backend
from pyfe4ai.utils import sampling_utils as su

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
    max_k = getattr(native_backend, "MAX_NATIVE_FACCT_WIDE_K", native_backend.MAX_NATIVE_FACCT_K)
    return [name for name, s in WIDTHS.items() if _k(s) <= max_k]


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with native_backend.backend("auto"):
        yield


class _Spy:
    """The extension, counting facct_draw calls."""

    def __init__(self, ext) -> None:
        self._ext, self.calls = ext, 0

    def __getattr__(self, name):
        return getattr(self._ext, name)

    def facct_draw(self, *args, **kwargs):
        self.calls += 1
        return self._ext.facct_draw(*args, **kwargs)


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


def test_shared_xof_random_never_draws_a_block_twice():
    """The native draw holds the XofRandom claim lock: a thread drawing from the
    same XofRandom gets other blocks, and the native samples are the reference's
    from the counter at which the draw ran."""
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
