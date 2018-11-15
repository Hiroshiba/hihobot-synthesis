from typing import NamedTuple
import numpy as np


class Wave(NamedTuple):
    wave: np.ndarray
    sampling_rate: int
