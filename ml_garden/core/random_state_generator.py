from typing import Optional

import numpy as np
from numpy.random import RandomState


class RandomStateManager:
    _instance: Optional[RandomState] = None

    @classmethod
    def get_state(cls) -> RandomState:
        if cls._instance is None:
            message = "Random state has not been initialized."
            raise ValueError(message)
        return cls._instance

    @classmethod
    def initialize(cls, seed: int) -> None:
        cls._instance = np.random.RandomState(seed)
