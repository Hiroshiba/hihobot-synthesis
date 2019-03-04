from pathlib import Path

import numpy as np
import pysptk
import pyworld
import torch
import torch.nn as nn
from nnmnkwii import paramgen
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing import minmax_scale
from nnmnkwii.preprocessing import trim_zeros_frames
from torch.autograd import Variable

from hihobot_synthesis.config import Config
from hihobot_synthesis.wave import Wave

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")
_binary_dict, _continuous_dict = hts.load_question_set(Path(__file__).parent / "questions_jp.hed")


class MyRNN(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=True):
        super(MyRNN, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, D_out)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(_device)),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(_device)))
        return h, c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        return output


class Synthesizer(object):
    def __init__(self, config: Config, duration_model_path: Path, acoustic_model_path: Path):
        self.duration_model = MyRNN(
            config.X_channel["duration"],
            config.hidden_size["duration"],
            config.Y_channel["duration"],
            config.num_hidden_layers["duration"],
            bidirectional=True,
        )
        if _use_cuda:
            self.duration_model.load_state_dict(torch.load(duration_model_path))
        else:
            self.duration_model.load_state_dict(torch.load(duration_model_path, map_location='cpu'))
        self.duration_model.to(_device)
        self.duration_model.eval()

        self.acoustic_model = MyRNN(
            config.X_channel["acoustic"],
            config.hidden_size["acoustic"],
            config.Y_channel["acoustic"],
            config.num_hidden_layers["acoustic"],
            bidirectional=True,
        )
        if _use_cuda:
            self.acoustic_model.load_state_dict(torch.load(acoustic_model_path))
        else:
            self.acoustic_model.load_state_dict(torch.load(acoustic_model_path, map_location='cpu'))
        self.acoustic_model.to(_device)
        self.acoustic_model.eval()

        self.config = config

    def gen_parameters(self, y_predicted):
        # Number of time frames
        T = y_predicted.shape[0]

        # Split acoustic features
        mgc = y_predicted[:, :self.config.lf0_start_idx]
        lf0 = y_predicted[:, self.config.lf0_start_idx:self.config.vuv_start_idx]
        vuv = y_predicted[:, self.config.vuv_start_idx]
        bap = y_predicted[:, self.config.bap_start_idx:]

        # Perform MLPG
        ty = "acoustic"
        mgc_variances = np.tile(self.config.Y_var[ty][:self.config.lf0_start_idx], (T, 1))
        mgc = paramgen.mlpg(mgc, mgc_variances, self.config.windows)
        lf0_variances = np.tile(self.config.Y_var[ty][self.config.lf0_start_idx:self.config.vuv_start_idx], (T, 1))
        lf0 = paramgen.mlpg(lf0, lf0_variances, self.config.windows)
        bap_variances = np.tile(self.config.Y_var[ty][self.config.bap_start_idx:], (T, 1))
        bap = paramgen.mlpg(bap, bap_variances, self.config.windows)

        return mgc, lf0, vuv, bap

    def gen_waveform(self, y_predicted, do_postfilter=False):
        y_predicted = trim_zeros_frames(y_predicted)

        # Generate parameters and split streams
        mgc, lf0, vuv, bap = self.gen_parameters(y_predicted)

        if do_postfilter:
            mgc = merlin_post_filter(mgc, self.config.alpha)

        spectrogram = pysptk.mc2sp(mgc, fftlen=self.config.fftlen, alpha=self.config.alpha)
        aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), self.config.fs, self.config.fftlen)
        f0 = lf0.copy()
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

        if self.config.lowest_frequency is not None:
            f0[f0 < self.config.lowest_frequency] = self.config.lowest_frequency

        f0[vuv < 0.5] = 0

        generated_waveform = pyworld.synthesize(
            f0.flatten().astype(np.float64),
            spectrogram.astype(np.float64),
            aperiodicity.astype(np.float64),
            self.config.fs,
            self.config.frame_period,
        )
        wave = generated_waveform.astype(np.float32) / 2 ** 15
        return Wave(wave=wave, sampling_rate=self.config.fs)

    def gen_duration(self, hts_labels):
        duration_model = self.duration_model

        # Linguistic features for duration
        duration_linguistic_features = fe.linguistic_features(
            hts_labels,
            _binary_dict, _continuous_dict,
            add_frame_features=False,
            subphone_features=None,
        ).astype(np.float32)

        # Apply normalization
        ty = "duration"
        duration_linguistic_features = minmax_scale(
            duration_linguistic_features,
            self.config.X_min[ty],
            self.config.X_max[ty],
            feature_range=(0.01, 0.99),
        )

        #  Apply model
        x = Variable(torch.from_numpy(duration_linguistic_features).to(_device)).float()
        try:
            duration_predicted = duration_model(x).cpu().data.numpy()
        except:
            h, c = duration_model.init_hidden(batch_size=1)
            xl = len(x)
            x = x.view(1, -1, x.size(-1))
            duration_predicted = duration_model(x, [xl], h, c).cpu().data.numpy()
            duration_predicted = duration_predicted.reshape(-1, duration_predicted.shape[-1])

        # Apply denormalization
        duration_predicted = duration_predicted * self.config.Y_scale[ty] + self.config.Y_mean[ty]
        duration_predicted = np.round(duration_predicted)

        # Set minimum state duration to 1
        duration_predicted[duration_predicted <= 0] = 1
        hts_labels.set_durations(duration_predicted)

        return hts_labels

    def gen_acoustic_feature(self, hts_labels):
        acoustic_model = self.acoustic_model

        # Predict durations
        duration_modified_hts_labels = self.gen_duration(hts_labels)

        # Linguistic features
        acoustic_subphone_features = "coarse_coding" if self.config.use_phone_alignment else "full"
        linguistic_features = fe.linguistic_features(
            duration_modified_hts_labels,
            _binary_dict,
            _continuous_dict,
            add_frame_features=True,
            subphone_features=acoustic_subphone_features,
        )
        # Trim silences
        indices = duration_modified_hts_labels.silence_frame_indices()
        linguistic_features = np.delete(linguistic_features, indices, axis=0)

        # Apply normalization
        ty = "acoustic"
        linguistic_features = minmax_scale(
            linguistic_features,
            self.config.X_min[ty],
            self.config.X_max[ty],
            feature_range=(0.01, 0.99),
        )

        # Predict acoustic features
        x = Variable(torch.from_numpy(linguistic_features).to(_device)).float()
        try:
            acoustic_predicted = acoustic_model(x).cpu().data.numpy()
        except:
            h, c = acoustic_model.init_hidden(batch_size=1)
            xl = len(x)
            x = x.view(1, -1, x.size(-1))
            acoustic_predicted = acoustic_model(x, [xl], h, c).cpu().data.numpy()
            acoustic_predicted = acoustic_predicted.reshape(-1, acoustic_predicted.shape[-1])

        # Apply denormalization
        acoustic_predicted = acoustic_predicted * self.config.Y_scale[ty] + self.config.Y_mean[ty]
        return acoustic_predicted

    def test_one_utt(self, hts_labels, post_filter=True):
        acoustic_predicted = self.gen_acoustic_feature(
            hts_labels=hts_labels,
        )
        return self.gen_waveform(acoustic_predicted, post_filter)
