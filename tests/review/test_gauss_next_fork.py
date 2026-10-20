"""The rounded Python sampler and ``gauss_next`` (October 2026 code review; the
code dated from v0.3.0).

The rounded Python sampler (``sampling_utils.discrete_gaussian_matrix``, used
for Python-path key generation and noise) drew ``SystemRandom().gauss``,
which caches the second Box-Muller value in the module-global
``_CSPRNG.gauss_next``: a future secret or noise coefficient stayed in memory
between calls, and after ``fork()`` parent and child drew the same first
sample. The sampler now drops the unused value.
"""

import os

import pytest

from pyfe4ai.utils import sampling_utils as su


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_parent_and_child_do_not_share_a_gaussian_sample():
    sigma = 2.0 ** 40
    su.discrete_gaussian_matrix(1, 1, sigma, sampler="rounded")  # an odd count leaves one value cached
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        os.write(write, str(int(su.discrete_gaussian_matrix(1, 1, sigma, sampler="rounded")[0][0])).encode())
        os._exit(0)
    os.close(write)
    parent = int(su.discrete_gaussian_matrix(1, 1, sigma, sampler="rounded")[0][0])
    child = int(os.read(read, 100).decode())
    os.close(read)
    os.waitpid(pid, 0)
    assert parent != child, "the same Gaussian sample in parent and child: {}".format(parent)
