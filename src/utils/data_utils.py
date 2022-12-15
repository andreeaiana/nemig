from typing import Any

import os
import pickle
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def update_cache(obj, file) -> None:
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_cache(fpath: str) -> Any:
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            log.warning(f'File {fpath} does not exist')
        
def check_integrity(fpath: str) -> bool:
    if not os.path.isfile(fpath):
        return False
    return True
