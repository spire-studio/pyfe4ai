"""The commands of ``benchmarks/run.py``: one module per command, each with
``main(argv=None, prog=None)`` parsing its own options (the options and the
output of the former script of that name, see :data:`COMMANDS`).

The modules import ``_common`` from ``benchmarks/`` and ``pyfe4ai`` (install
it, or set ``PYTHONPATH`` to the repository); ``run.py`` imports a command's
module only when it runs, so that the pairing commands do not load the
pairing groups for the others.
"""

from __future__ import annotations

from typing import NamedTuple


class Command(NamedTuple):
    module: str  #: module in this package
    former: str  #: the script it replaces (benchmarks/<former>), until 0.4.0
    summary: str


#: command name -> :class:`Command`, in the order ``run.py --help`` lists them.
COMMANDS = {
    # timing of the scheme families, CSV or JSON rows of _common.BENCHMARK_HEADER
    "sife": Command("sife", "benchmark_sife_instantiations.py",
                    "SIFE instantiations (DDH, LWE, FullySec-LWE, Ring-LWE, Paillier, Damgard) under one workload"),
    "mife": Command("mife", "benchmark_mife_instantiations.py", "MIFE instantiations under one multi-input workload"),
    "mcfe": Command("mcfe", "benchmark_mcfe_instantiations.py", "label-aware MCFE instantiations"),
    "dmcfe": Command("dmcfe", "benchmark_dmcfe_instantiations.py", "decentralized MCFE instantiations"),
    "tmife": Command("tmife", "benchmark_tmife_instantiations.py", "threshold MIFE instantiations"),
    "tmcfe": Command("tmcfe", "benchmark_tmcfe_instantiations.py", "threshold MCFE instantiations"),
    "lwe-family": Command("lwe_family", "benchmark_lwe_family.py", "SIFE / MIFE / MCFE / dMCFE / tMCFE over LWE"),
    "fh-family": Command("fh_family", "benchmark_fh_multi_ipe_family.py",
                         "MIFE / MCFE / dMCFE / tMCFE over the pairing-based FH-Multi-IPE (needs a pairing group)"),
    "quadratic": Command("quadratic", "benchmark_quadratic_family.py",
                         "the single-input quadratic schemes SGP and Quad (needs a pairing group)"),
    "ndarray": Command("ndarray", "benchmark_ndarray_helpers.py",
                       "the ndarray helper pipelines (--group sife | fh | quadratic | all)"),
    # building blocks
    "ntt": Command("ntt", "benchmark_ntt.py", "schoolbook vs NTT negacyclic polynomial multiplication"),
    "sampling": Command("sampling", "benchmark_sampling.py",
                        "the rounded vs the exact (FACCT) Gaussian sampler per width (--e2e: an MCFE round)"),
    # the FL path, before / after records
    "fl-compare": Command("fl_compare", "bench_fl_compare.py",
                          "the FL aggregation round per family, appending JSON records (see PERFORMANCE.md)"),
    "fl-coverage": Command("fl_coverage", "fl_path_coverage.py",
                           "not a timing: which registry entries run the ML adapter's FL path"),
    "replicated-sharing": Command("replicated_sharing", "bench_replicated_sharing.py",
                                  "replicated key sharing of the LWE threshold layers (positional KIND S T SHARING [REPS])"),
    "offline-potential": Command("offline_potential", "bench_offline_potential.py",
                                 "offline / online split of the DDH and DCR MCFE"),
    "fullysec-packing": Command("fullysec_packing", "bench_fullysec_packing.py",
                                "elementwise vs packed layout of sife/fullysec_ring_lwe"),
}
