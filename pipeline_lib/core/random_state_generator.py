from typing import Optional

import numpy as np
from numpy.random import RandomState

_random_state = None


def get_random_state() -> Optional[RandomState]:
    """
    Get the global random state object.

    Returns
    ----------
    RandomState or None: The global random state object if initialized, else None.
    """
    global _random_state
    return _random_state


def initialize_random_state(seed: int):
    """
    Initialize the global random state object with the provided seed.

    Parameters
    ----------
    seed (int): The seed value to initialize the random state object.
    """
    global _random_state
    _random_state = np.random.RandomState(seed)
