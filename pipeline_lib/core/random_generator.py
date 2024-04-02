from typing import Generator, Union

from numpy.random import default_rng
from numpy.random.bit_generator import SeedSequence


def get_random_generator() -> Generator:
    global _generator
    return _generator


def initialize_generator(seed: Union[int, SeedSequence]):
    global _generator
    _generator = default_rng(seed)
