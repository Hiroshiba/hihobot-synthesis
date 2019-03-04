import argparse
from pathlib import Path

import librosa
from nnmnkwii.io import hts

from hihobot_synthesis import Synthesizer, load_from_json


def sample(
        config: Path,
        duration_model: Path,
        acoustic_model: Path,
        label_file: Path,
        output_wave: Path,
):
    synthesizer = Synthesizer(
        config=load_from_json(config),
        duration_model_path=duration_model,
        acoustic_model_path=acoustic_model,
    )

    hts_labels = hts.load(str(label_file))
    wave = synthesizer.test_one_utt(hts_labels)

    librosa.output.write_wav(str(output_wave), wave.wave, sr=wave.sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('./sample/config.json'))
    parser.add_argument('--duration_model', type=Path, default=Path('./sample/model_duration_200'))
    parser.add_argument('--acoustic_model', type=Path, default=Path('./sample/model_acoustic_350'))
    parser.add_argument('--label_file', type=Path, default=Path('./sample/test_label_phone_align_490.lab'))
    parser.add_argument('--output_wave', type=Path, default=Path('./sample/sample.wav'))
    args = parser.parse_args()
    sample(
        config=args.config,
        duration_model=args.duration_model,
        acoustic_model=args.acoustic_model,
        label_file=args.label_file,
        output_wave=args.output_wave,
    )
