from importlib import resources
from logging.config import dictConfig
from pathlib import Path
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def init_logging_config() -> None:
    """Initialise logging from the bundled config.yaml."""
    ref = resources.files("pyfe4ai").joinpath("config.yaml")
    dict_config = load(ref.read_text(encoding="utf-8"), Loader=Loader)
    for handler in dict_config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
    dictConfig(dict_config)