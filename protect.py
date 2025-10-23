import os
import time
import torch
import librosa
import argparse

from E2E_VGuard import E2E_VGuard


def get_args():
    parser = argparse.ArgumentParser(description="The protect scipt for a single audio.")

    parser.add_argument("--input_wav", type=str, required=True, help="The file path of the input audio to be protected.")
    parser.add_argument("--ASR", type=str, default="wav2vec2-base", help="The targeted ASR system for text recognition.")
    parser.add_argument("--timbre_mode", type=str, default="untargeted", choices=["untargeted", "targeted"], 
                        help="The protective mode of timbre prevention.")
    parser.add_argument("--epsilon", type=int, default=8, help="The perturbation radius.")
    parser.add_argument("--epochs", type=int, default=500, help="The optimization epochs of generated perturbation.")

    args = parser.parse_args()
    return args



def main():
    TARGET_SR = 24000

    args = get_args()
    input_wav = args.input_wav

    epsilon = int(args.epsilon) / 255
    max_items = int(args.epochs)
    target_asr = args.ASR
    timbre_mode = args.timbre_mode

    device = torch.device("cuda")
    verbose = False
    mode = f"E2E-{target_asr}-{timbre_mode}"
    print(f"The protective mode of perturbation is {mode}.")

    input_wav_name = input_wav.rsplit('.', 1)[0]
    output_wav = input_wav_name + "_protected.wav" 

    defender = E2E_VGuard(target_asr, epsilon, max_items, device)
    defender.initialize()

    start_time = time.time()
    print("**************************************************************************************")
    _, sampling_rate = librosa.load(input_wav, sr=TARGET_SR)

    adv_wave, loss_items = defender.start_protect(input_wav, timbre_mode=timbre_mode, verbose=verbose)
    defender.save_audio(output_wav, adv_wave, sampling_rate)

    pred_text = defender._transcribe(output_wav)

    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
    print(f"[{formatted_time}] Sample: Loss info {loss_items}\n "
            f"                    Recognized text is [{pred_text}]")
    print("**************************************************************************************")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"The total process time is {total_time:.6}s.")

    print(f"The protected audio is saved to {output_wav}.")


if __name__ == "__main__":
    main()