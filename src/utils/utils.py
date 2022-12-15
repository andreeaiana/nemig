# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py

import warnings

from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger
from src.utils import rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)
