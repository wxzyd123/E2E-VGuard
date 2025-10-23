import os
import yaml
import shutil

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    ###### Download checkpoints ######

    # First, you need to download the pre-trained VITS model (`pretrained_ljs.pth`) from https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
    # Then, move it to the folder path `checkpoints/VITS/pretrained_ljs.pth`.

    print("Downloading GPT-SoVITS...")
    repo_id = "lj1995/GPT-SoVITS"
    download_path = "checkpoints/GSV/base_models/"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    snapshot_download(repo_id=repo_id, local_dir=download_path, local_dir_use_symlinks=False)


    print("Downloading WavLM...")
    repo_id = "microsoft/wavlm-base-plus"
    download_path = "checkpoints/wavlm"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    snapshot_download(repo_id=repo_id, local_dir=download_path, local_dir_use_symlinks=False)

    print("Downloading CosyVoice Encoder...")
    repo_id = "FunAudioLLM/CosyVoice-300M"
    files = ["campplus.onnx"]
    download_path = "checkpoints/CosyVoice/base_models/CosyVoice-300M"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    download_files(repo_id, files, download_path)


    print("Downloading StyleTTS2 Encoder...")
    repo_id = "yl4579/StyleTTS2-LibriTTS"
    files = ["Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth"]
    download_path = "checkpoints/StyleTTS2/base_models"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    download_files(repo_id, files, download_path)
    # move file 
    shutil.move("checkpoints/StyleTTS2/base_models/Models/LibriTTS/config.yml", "checkpoints/StyleTTS2/base_models/config.yml")
    shutil.move("checkpoints/StyleTTS2/base_models/Models/LibriTTS/epochs_2nd_00020.pth", "checkpoints/StyleTTS2/base_models/epochs_2nd_00020.pth")

    with open("checkpoints/StyleTTS2/base_models/config.yml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    config['ASR_config'] = "tts_models/styletts2/Utils/ASR/config.yml"
    config['ASR_path'] = "tts_models/styletts2/Utils/ASR/epoch_00080.pth"
    config['F0_path'] = "tts_models/styletts2/Utils/JDC/bst.t7"
    config['PLBERT_dir'] = "tts_models/styletts2/Utils/PLBERT"    

    with open('checkpoints/StyleTTS2/base_models/config.yml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, allow_unicode=True, default_flow_style=False, sort_keys=False)



def download_files(repo_id, files, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for file in files:
        try:
            hf_hub_download(repo_id=repo_id, filename=file, local_dir=download_path)
            print(f"{file} has been download successfully!")
        except Exception as e:
            print(f"{file} with error: {e}")


if __name__ == "__main__":
    main()