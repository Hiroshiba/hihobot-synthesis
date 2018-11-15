from pathlib import Path

from nnmnkwii.io import hts


def load_hts_labels(p: Path):
    hts_labels = hts.load(str(p))
    return hts_labels
