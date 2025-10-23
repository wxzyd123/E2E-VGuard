import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import torch
import time
import random
import logging
import whisper
import librosa
import torchaudio

import soundfile as sf
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from onnx2torch import convert
import nemo.collections.asr as nemo_asr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoModel
import torchaudio.compliance.kaldi as kaldi

import torch.nn.functional as F
multi_head_attention_forward = F.multi_head_attention_forward

sys.path.append("tts_models/vits")
from tts_models.vits.mel_processing import spectrogram_torch
import tts_models.vits.utils as utils
import tts_models.vits.commons as commons

sys.path.append("tts_models/gsv")
sys.path.append("tts_models/gsv/GPT_SoVITS")
from tts_models.gsv.GPT_SoVITS.inference_webui import change_sovits_weights

from toolbox import build_models_styletts2, build_models_vits
from masker import Masker

# Avoid Monkey Patching !!!
F.multi_head_attention_forward = multi_head_attention_forward


class E2E_VGuard():
    def __init__(self, asr_name, epsilon, max_items, device):
        self.device = device

        self.epsilon = epsilon
        self.max_items = max_items
        self.lr_optim = 1e-3
        self.sampling_rate = 16000

        config_path = f"tts_models/configs/LibriTTS_VITS.json"
        self.hps = utils.get_hparams_from_file(config_path=config_path)
        self.checkpoint_path_vits = "checkpoints/VITS/pretrained_ljs.pth"

        self.initialize()

        self.asr_name = asr_name

        self.long_script  = "The model is composed of multiple blocks with residual connections between them trained with CTC loss Each block consists of one or more modules with D timechannel separable convolutional layers batch normalization and ReLU layers Model achieves near state of the art accuracy on LibriSpeech and Wall Street Journal while having fewer parameters than all competing models Neural Modules NeMo toolkit makes it easy to use this model for transfer learning or fine tuning Encoder and Decoder checkpoints trained with NeMo can be used for fine tuning on new datasets"

        self.criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.ctc_loss = torch.nn.CTCLoss(blank=0)
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.build_encoders()
        self.target_asr(asr_name)

        self.step = 0


    def param_no_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def initialize(self):
        self.plateau_length = 5
        self.max_lr = 0.001
        self.min_lr = 1e-6
        self.plateau_drop = 2.
        self.last_ls = []
        self.lr = self.max_lr
    
    
    def target_asr(self, asr_name):
        ###### Initialize ASR Systems ######

        # Whisper Series
        if asr_name == "whisper-base":
            self.asr_model = whisper.load_model("base", device=self.device)
        elif asr_name == "whisper-small":
            self.asr_model = whisper.load_model("small", device=self.device)
        elif asr_name == "whisper-medium":
            self.asr_model = whisper.load_model("medium", device=self.device)
        elif asr_name == "whisper-large":
            self.asr_model = whisper.load_model("large", device=self.device)
        
        # Wav2Vec 2.0 Base and Large
        elif asr_name == "wav2vec2-base":
            self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.asr_model.to(self.device)
        elif asr_name == "wav2vec2-large":
            self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2Vec2-large-960h-lv60-self")
            self.asr_model.to(self.device)

        # Conformer Small
        elif asr_name == "conformer":
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_small")
            self.asr_model.to(self.device)
        
        # CitriNet 256
        elif asr_name == "citrinet":
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_256")
            self.asr_model.to(self.device)
        
        self.param_no_grad(self.asr_model)


    def _transcribe(self, audio_path):
        wave, sr = librosa.load(audio_path, sr=16000)

        if self.asr_name.startswith("whisper"):
            output_text = self.asr_model.transcribe(audio_path)["text"]
        elif self.asr_name.startswith("wav2vec2"):
            with torch.no_grad():
                model_inputs = self.processor(wave, sampling_rate=sr, return_tensors="pt", padding=True)
                model_inputs = model_inputs.to(self.device)
                logits = self.asr_model(model_inputs.input_values).logits
                pred_ids = torch.argmax(logits, dim=-1).cpu()
                output_text = self.processor.batch_decode(pred_ids)[0].lower()
        else:
            output_text = self.asr_model.transcribe([audio_path])[0]

        return output_text


    def save_audio(self, output_path, wave, sr):
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()
        
        sf.write(output_path, wave, samplerate=sr)


    def build_encoders(self):
        ###### Feature Encoders ######
        # VITS Posterior Encoder
        self.vits = build_models_vits(self.hps, checkpoint_path=self.checkpoint_path_vits)[0]
        self.vits.to(self.device)
        self.encoder_1 = self.vits.enc_q

        # # GSV Encoder
        sovits_path = "checkpoints/GSV/base_models/gsv-v2final-pretrained/s2G2333k.pth"
        self.gsv = change_sovits_weights(sovits_path=sovits_path)
        self.gsv.to(self.device)
        self.encoder_2 = self.gsv.enc_q

        # Acoustic Encoder
        self.encoder_3 = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40).to(self.device)


        ###### Timbre Encoders ######
        # WavLM encoder
        self.encoder_4 = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        self.encoder_4.to(self.device)

        # CosyVoice Encoder: CAM++
        pretrained_model_dir = "checkpoints/CosyVoice/base_models/CosyVoice-300M"
        onnx_path = f"{pretrained_model_dir}/campplus.onnx"
        self.encoder_5 = convert(onnx_path).to(self.device)
        self.encoder_5.eval()

        
        ###### Style Encoder ######
        # StyleTTS2 Style Encoder
        config_path = "checkpoints/StyleTTS2/base_models/config.yml"
        checkpoint_path = "checkpoints/StyleTTS2/base_models/epochs_2nd_00020.pth"
        tts_styletts2 = build_models_styletts2(config_path, checkpoint_path)[0]
        self.encoder_6 = tts_styletts2.style_encoder

        print("Successfully build encoder!")


        ###### Disable Gradient ######
        encoders = [self.vits, self.encoder_1, self.encoder_4, self.gsv, self.encoder_2,
                    self.encoder_5, self.encoder_6]
        for encoder in encoders:
            self.param_no_grad(encoder)

    
    def get_vits_emb(self, wave, sr):
        ### Modified from the VITS model
        spec = spectrogram_torch(wave, sampling_rate=sr)
        spec_len = torch.tensor([spec.shape[2]]).to(self.device)
        speaker_id = torch.tensor([0]).to(self.device)            
        
        try:
            g = self.vits.emb_g(speaker_id).unsqueeze(-1)
        except:
            g = None
        latent, _, _, spec_mask = self.encoder_1(spec, spec_len, g=g)        # [1, 192, spec_len]
        emb = self.vits.flow(latent, spec_mask, g=g)

        return emb


    def get_gsv_emb(self, wave, sr):
        ### Modified from the GPT-SoVITS model
        trans_32k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000).to(self.device)
        wave = trans_32k(wave)
        spec = spectrogram_torch(wave, n_fft=2048, sampling_rate=32000, hop_size=640, win_size=2048, center=False)
        spec_len = torch.tensor([spec.shape[2]]).to(self.device)          
        
        spec_mask = torch.unsqueeze(commons.sequence_mask(spec_len, spec.size(2)), 1).to(spec.dtype)
        ge = self.gsv.ref_enc(spec[:, :704] * spec_mask, spec_mask)
        latent, _, _, spec_mask = self.encoder_2(spec, spec_len, g=ge)        # [1, 192, spec_len]
        emb = self.gsv.flow(latent, spec_mask, g=ge)

        return emb


    def get_embeddings(self, wave, sr):
        # VITS Embeddings
        emb_1 = self.get_vits_emb(wave, sr)                         # [1, 192, spec_len]

        # GSV Embeddings
        emb_2 = self.get_gsv_emb(wave, sr)

        # Acoutic Features
        emb_3 = self.encoder_3(wave)

        # WavLM Embeddings
        emb_4 = self.encoder_4(input_values=wave, output_hidden_states=True).last_hidden_state
        emb_4 = emb_4.transpose(2, 1)                               # torch.Size([1, 768, shape])

        # CosyVoice Embeddings
        feature = kaldi.fbank(wave,
                              num_mel_bins=80,
                              dither=0,
                              sample_frequency=sr)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        emb_5 = self.encoder_5(feature)                              # [1, 192]

        # Style Encoder
        def preprocess(wave):
            to_mel = torchaudio.transforms.MelSpectrogram(
                n_mels=80, n_fft=2048, win_length=1200, hop_length=300).to(self.device)
            mean, std = -4, 4

            mel_tensor = to_mel(wave)
            mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
            return mel_tensor

        transform_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000).to(self.device)
        wave_24k = transform_24k(wave)
        mel_tensor = preprocess(wave_24k).to(self.device)
        emb_6 = self.encoder_6(mel_tensor)                             # [1, 128]
        
        embeddings = [emb_1, emb_2, emb_3, emb_4, emb_5, emb_6]
        return embeddings


    def compute_feature_loss(self, embs_1, embs_2):
        loss = 0.0

        encoder_weight_cosyvoice = 0.5
        for index, (emb_1, emb_2) in enumerate(zip(embs_1, embs_2)):
            try:
                score = self.criterion(emb_1, emb_2).mean()
            except:
                # For VITS, GSV, MFCC, WavLM
                min_len = min(emb_1.shape[2], emb_2.shape[2])
                emb_1 = emb_1[:, :, :min_len]
                emb_2 = emb_2[:, :, :min_len]
                score = self.criterion(emb_1, emb_2).mean()

            if (index + 1) == 5: # CosyVoice Encoder
                score = score * encoder_weight_cosyvoice

            loss = loss + score
            
        
        return loss
    

    def text2id(self, str, dict):
        ls = []
        for i in str:
            if i == ' ':
                ls.append(dict['|'])
            else:
                ls.append(dict[i])
        return ls
    

    def compute_asr_loss(self, adv_wave_16k, target_text, ori_text=None):
        """
            Compute the ASR loss with the target text
        """

        if self.asr_name.startswith("wav2vec2"):
            target_text = target_text.upper()

            with open('tts_models/configs/vocab.json', encoding='utf-8') as a:
                vocab = json.load(a)

            labels = torch.tensor(self.text2id(target_text, vocab))

            output = self.asr_model(adv_wave_16k, labels=labels)
            loss_asr = output.loss

        elif self.asr_name.startswith("whisper"):
            tokenizer = whisper.tokenizer.get_tokenizer(True, language="en")

            target_text = [*tokenizer.sot_sequence_including_notimestamps] + tokenizer.encode(target_text)
            target_labels = [target_text[1:] + [tokenizer.eot]]
            labels = target_labels

            if ori_text is not None:
                text_ = [*tokenizer.sot_sequence_including_notimestamps] + tokenizer.encode(ori_text)
            else:
                text_ = target_text

            dec_input_ids = [text_]

            label_lengths = [len(lab) for lab in labels]
            dec_input_ids_length = [len(e) for e in dec_input_ids]
            max_label_len = max(label_lengths + dec_input_ids_length)

            labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
            dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

            labels = torch.tensor(labels[0]).to(self.device).unsqueeze(0)
            dec_input_ids = torch.tensor(dec_input_ids[0]).to(self.device).unsqueeze(0)

            w_adv_wave = whisper.pad_or_trim(adv_wave_16k)
            if self.asr_name == "whisper-large":
                mel = whisper.log_mel_spectrogram(w_adv_wave, n_mels=128)
            else:
                mel = whisper.log_mel_spectrogram(w_adv_wave, n_mels=80)
            audio_features = self.asr_model.encoder(mel)

            out = self.asr_model.decoder(dec_input_ids, audio_features)
            loss_asr = self.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1))
        
        elif self.asr_name in ["conformer", "citrinet"]:
            tokenizer = self.asr_model.tokenizer
            target_ids = tokenizer.text_to_ids(target_text)
            target_ids = torch.LongTensor(target_ids).to(self.device).unsqueeze(0)
            target_length = torch.tensor([target_ids.shape[1]]).to(self.device)

            logits_hyps = self.asr_model.transcribe(adv_wave_16k.squeeze(0), batch_size=1, 
                                                    return_hypotheses=True, verbose=False)             
            logits = [hyp.alignments for hyp in logits_hyps][0]

            loss_asr = self.ctc_loss(log_probs=logits, targets=target_ids,
                                     input_lengths=[logits.shape[0]], target_lengths=[target_length])


        return loss_asr
    

    def select_target_speaker(self, ori_embs):
        """
            Select the best matched speaker from database for the input speech
        """

        speaker_database_path = "data/speakers_database"
        selected_speakers = os.listdir(speaker_database_path)
        min_loss = float('inf')
        best_speaker, best_embs = None, None

        for speaker in selected_speakers:
            speaker_path = os.path.join(speaker_database_path, speaker)

            wave, sr = librosa.load(speaker_path, sr=16000)
            wave = torch.from_numpy(wave).unsqueeze(0).to(self.device)
            embs = self.get_embeddings(wave, sr)

            loss = self.compute_feature_loss(ori_embs, embs)

            if loss < min_loss:
                min_loss = loss
                best_speaker = speaker_path
                best_embs = embs

        return best_speaker, best_embs
    

    def calculate_snr(self, wave, adv_wave):
        """
            Compute the SNR metric for each audio.
        """
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave)
        if isinstance(adv_wave, np.ndarray):
            adv_wave = torch.from_numpy(adv_wave)

        noise = adv_wave - wave
        
        signal_power = torch.sum(wave ** 2)
        noise_power = torch.sum(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power  + 1e-8))

        return snr


    def start_protect(self, audio_path, target_text=None, target_speaker=None, 
                      timbre_mode="untargeted", verbose=False):
        self.timbre_mode = timbre_mode
        wave, sr = torchaudio.load(audio_path) # [1, time]
        ori_length = wave.shape[1]
        if ori_length < sr * 1:
            wave = torch.cat((wave, torch.zeros((1, sr * 1 - ori_length))), dim=1)
        wave = wave.to(self.device)

        noise = torch.randn(wave.shape).to(self.device)
        max_item = noise.max().item()
        noise = noise / max_item * (self.epsilon)
        # noise = torch.zeros(wave.shape).to(self.device)
        adv_wave = torch.clamp(wave + noise, min=-1., max=1.)
        adv_wave.requires_grad = True

        transform_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)

        wave_16k = transform_16k(wave)
        wav_embs = self.get_embeddings(wave_16k, 16000)

        if timbre_mode == "targeted":
            print("Begin to select the target speaker!")
            target_speaker, target_embs = self.select_target_speaker(wav_embs)
            print(f"The target speaker path is:", target_speaker)

        ori_text = self._transcribe(audio_path)
        if target_text is None:
            if target_speaker is None:
                max_index = len(self.long_script) - len(ori_text) - 1
                start_index = random.randint(0, max_index)
                end_index = start_index + len(ori_text)

                target_text = self.long_script[start_index: end_index]
            else:
                ###### For the targeted attack, we utilize the select speaker's text ######
                target_text = self._transcribe(target_speaker)
                target_text = target_text[:len(ori_text)]
        else:
            target_text = target_text[:len(ori_text)]


        print("The original text is:", [ori_text.lower()])
        print("The target text is:  ", [target_text])

        opt_noise = optim.SGD([adv_wave], lr=self.lr_optim, weight_decay=0.95)

        # Original variable for PSY
        msker = Masker(device=self.device, sample_rate=sr)
        theta, original_max_psd = msker._compute_masking_threshold(wave[0].cpu().numpy())
        theta = torch.FloatTensor(theta.transpose(1, 0))
        original_max_psd = torch.FloatTensor([original_max_psd])
        theta_batch = theta.unsqueeze(0)
        original_max_psd_batch = original_max_psd.unsqueeze(0)


        for step in tqdm(range(self.max_items)):
        # for step in range(self.max_items):
            self.step = step
            adv_wave_16k = transform_16k(adv_wave)


            ###### Compute the Feature Losses ######
            adv_embs = self.get_embeddings(adv_wave_16k, 16000)

            if timbre_mode == "untargeted":
                loss_feature = self.compute_feature_loss(wav_embs, adv_embs)
            else:
                loss_feature = self.compute_feature_loss(target_embs, adv_embs) * -1.

            # ###### Compute ASR Losses ######
            loss_asr = self.compute_asr_loss(adv_wave_16k, target_text, ori_text=ori_text)


            ###### Compute Psychoacoustic Losses ######
            loss_psy = msker.batch_forward_2nd_stage(
                local_delta_rescale=(adv_wave - wave).squeeze(1),
                theta_batch=theta_batch.to(self.device),
                original_max_psd_batch=original_max_psd_batch.to(self.device),
            )

            loss_psy_max = 1.0e7
            if torch.isnan(loss_psy) or torch.isinf(loss_psy):
                loss_psy = torch.FloatTensor([loss_psy_max]).to(self.device)
            elif loss_psy > loss_psy_max:
                loss_psy = torch.FloatTensor([loss_psy_max]).to(self.device)

            ###### L2 norm for imperceptibility #######
            loss_l2 = torch.norm(adv_wave - wave, p=2)

            ###### Total Loss for Optimization #######
            weight_asr = 1
            if self.asr_name.startswith("wav2vec2"):
                weight_feature = 500
                weight_psy = 5e-3
                weight_l2 = 0.1
            elif self.asr_name.startswith("whisper"):
                weight_feature = 1
                weight_psy = 1e-5
                weight_l2 = 0.1
            elif self.asr_name == "conformer":
                weight_feature = 2
                weight_psy = 2e-5
                weight_l2 = 0.1
            elif self.asr_name == "citrinet":
                weight_feature = 2
                weight_psy = 2e-5
                weight_l2 = 0.1

            loss = loss_feature * weight_feature + loss_asr * weight_asr + loss_psy * weight_psy + loss_l2 * weight_l2
            loss_items = {"loss_emb": f"{loss_feature.item():.6f}",
                          "loss_asr": f"{loss_asr.item():.6f}",
                          "loss_psy": f"{loss_psy.item():.6f}",
                          "loss_l2": f"{loss_l2.item():.6f}"}

            
            if step % 50 == 0 and verbose:
                print(step, loss_items, self.lr)
            
            opt_noise.zero_grad()
            loss.backward()
            opt_noise.step()
        

            ### Optimize the learning rate
            self.last_ls.append(loss)
            self.last_ls = self.last_ls[-self.plateau_length:]
            if self.last_ls[-1] > self.last_ls[0] and len(self.last_ls) == self.plateau_length:
                if self.lr > self.min_lr:
                    self.lr = max(self.lr / self.plateau_drop, self.min_lr)
                self.last_ls = []


            ### Projected Gradient Descent
            if self.asr_name in ["conformer", "citrinet"]:
                delta = (self.epsilon / 10) * torch.sign(adv_wave.grad) * -1.
            else:
                delta = self.lr * torch.sign(adv_wave.grad) * -1.
            adv_wave = delta + adv_wave
            noise = torch.clamp(adv_wave.data - wave, min=-self.epsilon, max=self.epsilon)
            adv_wave = torch.clamp(wave + noise, min=-1., max=1.)
            adv_wave.requires_grad = True

        snr_24k = self.calculate_snr(wave, adv_wave.detach()).item()
        print("The SNR_24K is", snr_24k)

        adv_wave = adv_wave[:, :ori_length]

        return adv_wave.detach().cpu().numpy()[0], loss_items
    