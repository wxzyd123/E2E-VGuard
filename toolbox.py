import sys
import json
import torch
import yaml
import torchaudio
import phonemizer
from munch import Munch
from torch import nn


def build_models_vits(hps, checkpoint_path=None):
    sys.path.append("tts_models/vits")
    from tts_models.vits.text.symbols import symbols
    from tts_models.vits.models import (
        SynthesizerTrn,
        MultiPeriodDiscriminator,
    )

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        try:
            checkpoint_dict = checkpoint['model']
        except:
            checkpoint_dict = checkpoint
        for layer_name, layer_params in net_g.state_dict().items():
            if layer_name in checkpoint_dict:
                checkpoint_dict_param = checkpoint_dict[layer_name]
                if checkpoint_dict_param.shape == layer_params.shape:
                    net_g.state_dict()[layer_name].copy_(checkpoint_dict_param)
                    # print(f"[·] Load the {layer_name} successfully!")
                else:
                    print(
                        f"[>] Layer {layer_name}, the layer size is {layer_params.shape}, the checkpoint size is {checkpoint_dict_param.shape}")
            else:
                print(f"[!] The layer {layer_name} is not found!")


    return net_g, net_d


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def build_models_styletts2(config_path, checkpoint_path=None):
    sys.path.append("tts_models/styletts2")
    from tts_models.styletts2.models import load_F0_models, load_ASR_models, build_model
    from tts_models.styletts2.Utils.PLBERT.util import load_plbert

    device = torch.device("cuda")

    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    config = yaml.safe_load(open(config_path))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load(checkpoint_path, map_location='cpu')
    params = params_whole['net']

    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]

    return model, model_params, global_phonemizer


def get_spec(hps, waves, waves_len):
    from tts_models.vits.mel_processing import spectrogram_torch

    spec_np = []
    spec_lengths = torch.LongTensor(len(waves))

    device = waves.device
    for index, wave in enumerate(waves):
        audio_norm = wave[:, :waves_len[index]]
        spec = spectrogram_torch(audio_norm,
                                 hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        spec_np.append(spec)
        spec_lengths[index] = spec.size(1)

    max_spec_len = max(spec_lengths)
    spec_padded = torch.FloatTensor(len(waves), spec_np[0].size(0), max_spec_len)
    spec_padded.zero_()

    for i, spec in enumerate(waves):
        spec_padded[i][:, :spec_lengths[i]] = spec_np[i]

    return spec_padded.to(device), spec_lengths.to(device)


# def build_vocoder(device):
#     config_path = "hifigan/checkpoints/config.json"
#     with open(config_path, "r") as f:
#         data = f.read()

#     global h
#     json_config = json.loads(data)
#     h = AttrDict(json_config)

#     generator = Generator(h).to(device)

#     checkpoint_path = "hifigan/checkpoints/g_02500000"
#     state_dict_g = load_checkpoint(checkpoint_path, device)
#     generator.load_state_dict(state_dict_g['generator'])

#     generator.eval()
#     generator.remove_weight_norm()

#     return generator, h


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, win_length=None,
                 n_mels=100, mel_fmin=0, mel_fmax=None, normalize=False, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1,
            normalized=normalize,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mels,
            center=padding == "center",
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        mel = safe_log(mel)
        return mel
    


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))