import json
from pathlib import Path
from typing import Tuple, List, Dict, NamedTuple, Optional

import numpy as np


class Config(NamedTuple):
    lowest_frequency: Optional[float]

    mgc_dim: int
    lf0_dim: int
    vuv_dim: int
    bap_dim: int

    duration_linguistic_dim: int
    acoustic_linguisic_dim: int
    duration_dim: int
    acoustic_dim: int

    fs: int
    frame_period: int
    fftlen: int
    alpha: float
    hop_length: int

    mgc_start_idx: int
    lf0_start_idx: int
    vuv_start_idx: int
    bap_start_idx: int

    windows: Tuple[int, int, List[float]]

    use_phone_alignment: bool

    num_hidden_layers: Dict[str, int]
    hidden_size: Dict[str, int]

    batch_size: int
    n_workers: int
    pin_memory: bool
    nepoch: int
    n_save_epoch: int
    lr: float
    weight_decay: float

    X_channel: Dict[str, int]
    Y_channel: Dict[str, int]
    X_min: Dict[str, np.ndarray]
    X_max: Dict[str, np.ndarray]
    Y_mean: Dict[str, np.ndarray]
    Y_var: Dict[str, np.ndarray]
    Y_scale: Dict[str, np.ndarray]


def load_from_json(p: Path):
    d = json.load(p.open())
    return Config(
        lowest_frequency=d['lowest_frequency'],

        mgc_dim=d['mgc_dim'],
        lf0_dim=d['lf0_dim'],
        vuv_dim=d['vuv_dim'],
        bap_dim=d['bap_dim'],

        duration_linguistic_dim=d['duration_linguistic_dim'],
        acoustic_linguisic_dim=d['acoustic_linguisic_dim'],
        duration_dim=d['duration_dim'],
        acoustic_dim=d['acoustic_dim'],

        fs=d['fs'],
        frame_period=d['frame_period'],
        fftlen=d['fftlen'],
        alpha=d['alpha'],
        hop_length=d['hop_length'],

        mgc_start_idx=d['mgc_start_idx'],
        lf0_start_idx=d['lf0_start_idx'],
        vuv_start_idx=d['vuv_start_idx'],
        bap_start_idx=d['bap_start_idx'],

        windows=d['windows'],

        use_phone_alignment=d['use_phone_alignment'],

        num_hidden_layers=d['num_hidden_layers'],
        hidden_size=d['hidden_size'],

        batch_size=d['batch_size'],
        n_workers=d['n_workers'],
        pin_memory=d['pin_memory'],
        nepoch=d['nepoch'],
        n_save_epoch=d['n_save_epoch'],
        lr=d['lr'],
        weight_decay=d['weight_decay'],

        X_channel=d['X_channel'],
        Y_channel=d['Y_channel'],
        X_min={k: np.array(v) for k, v in d['X_min'].items()},
        X_max={k: np.array(v) for k, v in d['X_max'].items()},
        Y_mean={k: np.array(v) for k, v in d['Y_mean'].items()},
        Y_var={k: np.array(v) for k, v in d['Y_var'].items()},
        Y_scale={k: np.array(v) for k, v in d['Y_scale'].items()},
    )
