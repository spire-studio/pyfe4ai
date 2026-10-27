"""The benchmarks of PyFE4AI, one command each.

Usage::

    python benchmarks/run.py --help                 # the commands
    python benchmarks/run.py <command> --help       # the options of one command
    python benchmarks/run.py <command> [options]

for example ``python benchmarks/run.py sife --profile small --rounds 1`` or
``python benchmarks/run.py fl-compare --families ring_lwe --backend native``.
Each command keeps the options and the output of the script it replaces
(``benchmarks/<script>.py`` until 0.4.0; see ``benchmarks/README.md``). Run it
from the repository with ``pyfe4ai`` installed (or ``PYTHONPATH`` set to the
repository).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # _common and commands, also when run.py is imported
    sys.path.insert(0, HERE)

from commands import COMMANDS  # noqa: E402


def _epilog() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["commands:"]
    lines += ["  {:<{}}  {}".format(name, width, c.summary) for name, c in COMMANDS.items()]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description=__doc__.splitlines()[0], epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [-h] command [options of the command]",
    )
    p.add_argument("command", choices=list(COMMANDS), metavar="command", help="one of the commands below")
    p.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    module = importlib.import_module("commands." + COMMANDS[args.command].module)
    module.main(args.args, prog="run.py " + args.command)


if __name__ == "__main__":
    main()
