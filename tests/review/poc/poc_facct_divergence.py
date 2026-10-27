"""PoC (October 2026 code review): the native FACCT acceptance test disagreed
with the Python reference on crafted bits, because native_backend._facct_table
truncated c = 1/(2 sigma^2) to 53 bits (default gmpy2 context) while the
'undecided' band is 2^-100. Run from this directory with PYTHONPATH set to a
checkout of the code to examine.
"""
import os, random, sys
import gmpy2 as gp
sys.argv = [sys.argv[0], os.path.join(os.environ["PYTHONPATH"], "rust/core/src/facct.rs")]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "facct_port.py")).read())
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives import sampling as su
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe

class StreamBits:
    def __init__(self, data): self.data, self.pos = data, 0
    def getrandbits(self, k):
        if k <= 0: return 0
        n = (k + 7) // 8
        b = self.data[self.pos:self.pos + n]; assert len(b) == n
        self.pos += n
        return int.from_bytes(b, "big") >> (8 * n - k)

nids = ["c0", "c1"]
kg = mife_rlwe.MIFERingLWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "ring_n": 16, "sec_param": 64,
                                         "bound_x": 3, "bound_y": 1})
kg.setup()
pp = kg.get_public_parameters()
cl = mife_rlwe.MIFERingLWE({"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys("c0")}})
sigma2 = pp["sigma2"]
params = su._facct_params(su._exact(sigma2))
k, cdt, neg_c = params.k, params.cdt, params.neg_c
cm, ce = table_c(neg_c)                       # what native_impl._facct_table passes (native_backend's then)
print("sigma2 =", sigma2, " k =", k, " y_bits =", params.y_bits, " c mantissa significant bits:",
      cm.bit_length() - (cm & -cm).bit_length() + 1)

# find the (x, y) with the largest |A - T|
best = None
for x in range(15):
    for y in range(1, k):
        t = y * (y + 2 * k * x)
        A = exp_neg(cm, ce, t) << (P - F)
        with gp.context(precision=P + 64, round=gp.RoundToNearest):
            T = int(gp.floor(gp.exp(neg_c * t) * (1 << P)))
        if best is None or abs(A - T) > abs(best[2] - best[3]):
            best = (x, y, A, T)
x, y, A, T = best
print("x=%d y=%d  log2|A-T|/2^192 = %.2f (band 2^-100)" % (x, y, math.log2(abs(A - T)) - P))
if T < A - MARGIN - 1:
    v = T          # Python: v >= T -> reject;  native: v + 1 + margin <= A -> accept
elif A + MARGIN < T:
    v = A + MARGIN  # Python: v < T -> accept; native: v >= A + margin -> reject
else:
    sys.exit("no gap at this width")
print("native decision:", decide(cm, ce, k, x, y, v, T)[0], " python decision:", v < T)

u = cdt[x - 1] if x > 0 else 0
ny = (params.y_bits + 7) // 8
prefix = u.to_bytes(24, "big") + (y << (8 * ny - params.y_bits)).to_bytes(ny, "big") + v.to_bytes(24, "big")

def run(data):
    xs = [[1, -2, 3, 0] + [0] * 12]
    with su.gaussian_sampler("facct"):
        with native_backend.backend("python"):
            su._OS_RANDOM, saved = StreamBits(data), su._OS_RANDOM
            try:
                py = cl.encrypt(xs)
            finally:
                su._OS_RANDOM = saved
        with native_backend.backend("native"):
            native_backend._random_override = lambda n: data
            try:
                nat = cl.encrypt(xs)
            finally:
                native_backend._random_override = None
    return py, nat

rest = random.Random(7).randbytes(1 << 20)
py, nat = run(os.urandom(0) + rest)
print("control (random bytes):  python == native:", py == nat)
py, nat = run(prefix + rest)
print("crafted first trial:     python == native:", py == nat)
print("first sample, python reference:", int(su.discrete_gaussian_facct(sigma2, 1, rng=StreamBits(prefix + rest))[0]))

# Same crafted bytes with the table's c computed at full precision (the fix): agreement.
orig = native_impl._facct_table.__wrapped__
def precise(sigma_q):
    with gp.context(precision=1000):
        return orig(sigma_q)
native_impl._facct_table = precise
py, nat = run(prefix + rest)
print("crafted, c at full precision (fix): python == native:", py == nat)
