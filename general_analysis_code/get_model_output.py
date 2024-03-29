import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torch.cuda.amp import autocast
import scipy
from deepspeech_pytorch.configs.train_config import (
    SpectConfig,
    BiDirectionalConfig,
    AdamConfig,
    SGDConfig,
)
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate
from deepspeech_pytorch.model import (
    SequenceWise,
    MaskConv,
    InferenceBatchSoftmax,
    Lookahead,
)
from deepspeech_pytorch.enums import RNNType, SpectrogramWindow
from omegaconf import OmegaConf
import pytorch_lightning as pl
import soundfile as sf
import pandas as pd
from gensim.models import KeyedVectors

import dill
import os

torchaudio.set_audio_backend("sox_io")
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "girl's": "girl",
    "husband's": "husband",
}


# This class is used for encoding words into numerical representations (vectors) using the GloVe model.
class glove_encoder:
    # The constructor of the glove_encoder class.
    # Input: glove_input_file - str, path to the GloVe model file.
    def __init__(self, glove_input_file):
        # Load the GloVe model from a word2vec format file.
        # The GloVe model is not binary and does not include a file header.

        input_dir = os.path.dirname(glove_input_file)
        out_put_file = os.path.join(input_dir, "glove_encoder.pkl")
        if glove_input_file.endswith(".txt"):
            glove_model = KeyedVectors.load_word2vec_format(
                glove_input_file, binary=False, no_header=True
            )
            with open(out_put_file, "wb") as f:
                dill.dump(glove_model, f)

        elif glove_input_file.endswith(".pkl"):
            with open(glove_input_file, "rb") as f:
                glove_model = dill.load(f)

        self.glove_model = glove_model

    # This method is used to encode a single word into a vector using the GloVe model.
    # Input: word - str, a word to encode.
    # Output: If the word is in the GloVe model, return a vector representation of the word.
    # If not, print the word and return a zero vector of length 300.
    def encode(self, word, auto_expand=True, CONTRACTION_MAP=CONTRACTION_MAP):
        if self.glove_model.key_to_index.get(word) is not None:
            return self.glove_model[word]
        elif auto_expand and word in CONTRACTION_MAP.keys():
            expanded_words = CONTRACTION_MAP[word].split()
            print(f"Automatically expand {word} to {CONTRACTION_MAP[word]}")
            return np.mean(
                [self.encode(expanded_word) for expanded_word in expanded_words], axis=0
            )
        else:
            print(f"word can not be found in the GloVe model: {word}")
            return np.zeros(300)

    # This method is used to encode a sequence of words into a matrix using the GloVe model.
    # Input: words_iterable - Iterable[str], a sequence of words to encode.
    # word_onsets - Iterable[float], starting times for each word in the sequence.
    # word_offsets - Iterable[float], ending times for each word in the sequence.
    # time_length - int, the total length of the time sequence.
    # sr - int, the sampling rate used for converting times to sample indices, default is 100.
    # Output: A matrix of shape (time_length, 300) where each row is the vector representation of a word.
    def encode_sequences(
        self, words_iterable, word_onsets, word_offsets, time_length, sr=100
    ):
        assert len(words_iterable) == len(word_onsets) == len(word_offsets)
        features = np.zeros((time_length, 300))
        t_stim = np.arange(time_length) / sr
        for i in range(len(words_iterable)):
            word = words_iterable[i]
            onset = word_onsets[i]
            offset = word_offsets[i]
            indexs = (onset <= t_stim) & (t_stim <= offset)
            features[indexs, :] = self.encode(word)

        return features


class SpectrogramParser(nn.Module):
    def __init__(self, audio_conf: SpectConfig, normalize: bool = False):
        """
        Parses audio file into spectrogram with optional normalization
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        """
        super().__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        if self.window == "hamming":
            window = torch.hamming_window
        else:
            raise NotImplementedError()

        self.transform = torchaudio.transforms.Spectrogram(
            n_fft, win_length, hop_length, window_fn=window, power=1, normalized=False
        )

    @torch.no_grad()
    def forward(self, audio):
        if audio.shape[-1] == 1:
            audio = audio.squeeze(dim=-1)  # mono
        else:
            audio = audio.mean(dim=-1)  # multiple channels, average

        # trim final samples if extra samples left out from downsampling doing conversion
        audio = audio[: -round(len(audio) % self.transform.hop_length) - 1]

        spect = self.transform(audio)
        spect = torch.log1p(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        # reshape to [time x frequency]
        spect = spect.T.contiguous()

        return spect


class BatchRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type=nn.LSTM,
        bidirectional=False,
        batch_norm=True,
    ):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = (
            SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        )
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
        )
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = (
                x.view(x.size(0), x.size(1), 2, -1)
                .sum(2)
                .view(x.size(0), x.size(1), -1)
            )  # (TxNxH*2) -> (TxNxH) by sum
        return x


LABELS = list("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
MODEL_CFG = BiDirectionalConfig(
    rnn_type=RNNType.lstm, hidden_size=1024, hidden_layers=7
)
OPTIM_CFG = AdamConfig(
    learning_rate=0.00015,
    learning_anneal=0.99,
    weight_decay=1e-05,
    eps=1e-08,
    betas=[0.9, 0.999],
)
SPECT_CFG = SpectConfig(
    sample_rate=16000,
    window_size=0.02,
    window_stride=0.01,
    window=SpectrogramWindow.hamming,
)
PRECISION = 16


class DeepSpeech(pl.LightningModule):
    def __init__(
        self,
        labels=LABELS,
        model_cfg=MODEL_CFG,
        precision=PRECISION,
        optim_cfg=OPTIM_CFG,
        spect_cfg=SPECT_CFG,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.bidirectional = (
            True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False
        )

        self.labels = labels
        num_classes = len(self.labels)

        self.conv = MaskConv(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(
            math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2)
            + 1
        )
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=self.model_cfg.rnn_type.value,
                bidirectional=self.bidirectional,
                batch_norm=False,
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional,
                )
                for x in range(self.model_cfg.hidden_layers - 3)
            ),
        )

        self.lookahead = (
            nn.Sequential(
                # consider adding batch norm?
                Lookahead(
                    self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context
                ),
                nn.Hardtanh(0, 20, inplace=True),
            )
            if not self.bidirectional
            else None
        )

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size),
            nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False),
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = nn.CTCLoss(
            blank=self.labels.index("_"), reduction="sum", zero_infinity=True
        )
        self.evaluation_decoder = GreedyDecoder(
            self.labels
        )  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder, target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder, target_decoder=self.evaluation_decoder
        )

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x.transpose(1, 2).unsqueeze(1).contiguous(), output_lengths)

        sizes = x.size()
        x = x.view(
            sizes[0], sizes[1] * sizes[2], sizes[3]
        )  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def unpack_batch(self, batch):
        inputs = batch.get("inputs", None)
        input_lengths = batch.get("input_lengths", None)
        labels = batch.get("labels", None)
        label_lengths = batch.get("label_lengths", None)

        return inputs, labels, input_lengths, label_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = self.unpack_batch(batch)
        if inputs is None:  # skip step
            return None

        out, output_sizes = self(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, output_sizes, target_sizes)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay,
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay,
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = self.unpack_batch(batch)
        if inputs is None:  # skip step
            return

        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)

        self.wer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes,
        )
        self.cer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes,
        )
        self.log("wer", self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log("cer", self.cer.compute(), prog_bar=True, on_epoch=True)

    def test_step(self, *args):
        return self.validation_step(*args)

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (
                    torch.div(
                        seq_len
                        + 2 * m.padding[1]
                        - m.dilation[1] * (m.kernel_size[1] - 1)
                        - 1,
                        m.stride[1],
                        rounding_mode="floor",
                    )
                    + 1
                )
        return seq_len.int()

    @torch.no_grad()
    def activation_fx(self, layer, log=True):
        # waveform 2 spectrogram parser
        spect_parser = SpectrogramParser(audio_conf=SPECT_CFG, normalize=True).to(
            self.device
        )

        def activation(x, /, layer=layer):
            # convert to spectrogram
            x = spect_parser(x)
            lengths = torch.tensor([x.shape[0]], dtype=int)
            output_lengths = self.get_seq_lens(lengths)

            # make into 4D tensor of [batch x channel x frequency x time]
            # and move to same device as the model
            x = x.T[np.newaxis, np.newaxis, ...].contiguous().to(device=self.device)

            for module in self.conv.seq_module:
                x = module(x)
                mask = torch.BoolTensor(x.size()).fill_(0)
                if x.is_cuda:
                    mask = mask.to(self.device)
                for i, length in enumerate(output_lengths):
                    length = length.item()
                    if (mask[i].size(2) - length) > 0:
                        mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                x = x.masked_fill(mask, 0)

                if isinstance(module, torch.nn.Hardtanh):
                    layer -= 1
                    if layer < 0:
                        break

            sizes = x.size()
            x = x.view(
                sizes[0], sizes[1] * sizes[2], sizes[3]
            )  # Collapse feature dimension
            x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
            if layer < 0:
                # import pdb; pdb.set_trace()
                return x.squeeze(dim=1).cpu()

            for rnn in self.rnns:
                x = rnn(x, output_lengths)
                layer -= 1
                if layer < 0:
                    return x.squeeze(dim=1).cpu()

            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)

            x = self.fc(x)

            # identity in training mode, softmax in eval mode
            if log:
                x = torch.nn.functional.log_softmax(x, dim=-1)
            else:
                x = torch.nn.functional.softmax(x, dim=-1)
            layer -= 1
            if layer < 0:
                return x.squeeze(dim=1).cpu()

            return None

        return activation


class deepspeech_encoder:
    def __init__(self, state_dict_path, device="cpu", compile_torch=True):
        model = DeepSpeech().to(device).eval()
        model.load_state_dict(
            torch.load(state_dict_path, map_location=device)["state_dict"]
        )
        if compile_torch:
            model = torch.compile(model)
        self.model = model
        self.device = device

    def extract_deepSpeech_feature(self, waveform, sample_rate):
        """

        state_dict_path: the path of the pretrained model weight
        """
        waveform = waveform.to(self.device)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        waveform = waveform.reshape(-1, 1)

        dfs = []
        for i in range(8):
            activate_layer = self.model.activation_fx(i)
            # get activation of first layer
            activation = activate_layer(waveform)
            activation = activation.detach().numpy()
            time_stamp = np.arange(-0.5, (len(activation) - 1) * 320, step=320)
            assert len(activation) == len(time_stamp)
            df = pd.DataFrame(
                {"time_stamp": [time_stamp], "activation": [activation], "layer": [i]},
                index=[i],
            )
            dfs.append(df)
        df = pd.concat(dfs)
        return df


def plot_average_envelop_and_origin(wav_path, feature_df):
    from scipy.signal import hilbert

    # read test.wav
    waveform, sample_rate = sf.read(wav_path)
    # extract envelope
    envelope = np.abs(hilbert(waveform))
    wav_time = np.arange(0, len(waveform))
    # normalize envelope
    envelope = envelope / np.max(envelope)
    plt.figure(figsize=(20, 10))

    plt.plot(wav_time, envelope)
    for i in range(len(feature_df["layer"].unique())):
        time_stamp = feature_df[feature_df["layer"] == i]["time_stamp"].values[0]
        activation = feature_df[feature_df["layer"] == i]["activation"].values[0]
        activation = np.mean(activation, axis=1)
        activation = activation / np.max(activation)
        plt.plot(time_stamp, activation)
        # legend
    plt.legend(
        [
            "envelope",
            "layer 0",
            "layer 1",
            "layer 2",
            "layer 3",
            "layer 4",
            "layer 5",
            "layer 6",
            "layer 7",
        ]
    )
    plt.savefig("test.png")
    plt.tight_layout()
    plt.close()


def get_cochleagram(wav_path, output_sr=100, nonlinearity="power", n=None):
    """
    This is a wrap of function human_cochleagram from package pycochleagram.
    wav_path: path to the wav file
    output_sr: sampling rate of the cochleagram

    return: cochleagram with given sampling rate.
    """
    from pycochleagram.cochleagram import human_cochleagram
    y, sr = sf.read(wav_path)
    # get the cochleagram
    coch = human_cochleagram(y, sr, strict=False, nonlinearity=nonlinearity, n=n)
    coch = np.flipud(coch)
    # resample to 100Hz
    coch = scipy.signal.resample(
        coch, np.int32(np.round(output_sr / sr * coch.shape[1])), axis=1
    )
    coch = coch.T
    return coch


if __name__ == "__main__":
    glove_input_file = (
        "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/glove.840B.300d.txt"
    )
    encoder_glove = glove_encoder(glove_input_file)


# if __name__ == '__main__':
#     #example of get_cochleagram
#     wav1 = '/scratch/snormanh_lab/shared/Sigurd/stim5_alarm_clock.wav'
#     coch = get_cochleagram(wav1)
#     #plot the cochleagram
#     plt.imshow(coch.T, aspect='auto', origin='lower', cmap='jet')
#     plt.savefig(wav1.replace('.wav', '.png'))
#     plt.close()

#     wav2 = '/scratch/snormanh_lab/shared/Sigurd/LJ001-0001.wav'
#     coch = get_cochleagram(wav2)
#     #plot the cochleagram
#     plt.imshow(coch.T, aspect='auto', origin='lower', cmap='jet')
#     plt.savefig(wav2.replace('.wav', '.png'))
#     plt.close()
#     pass


# if __name__ == "__main__":
#     #example of extract_deepSpeech_feature
#     feature_df = extract_deepSpeech_feature('test.wav','/scratch/snormanh_lab/shared/Sigurd/PyTCI/Examples/resources/deepspeech2-pretrained.ckpt')
#     plot_average_envelop_and_origin('test.wav',feature_df)

#     #for example:
#     # print the layers
#     print(feature_df['layer'].unique())
#     #get the time stamp of the second layer
#     time_stamp = feature_df[feature_df['layer']==1]['time_stamp'].values[0]
#     print(time_stamp)
#     #get the activation of the second layer
#     activation = feature_df[feature_df['layer']==1]['activation'].values[0]
#     print(activation)
