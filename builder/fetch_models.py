import os
import wget
from runpod.serverless.utils import rp_cuda
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import platform
import signal
# Check if running on Windows and apply a workaround
if platform.system() == 'Windows':
    if not hasattr(signal, 'SIGKILL'):
        signal.SIGKILL = signal.SIGTERM  # or SIGINT
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from omegaconf import OmegaConf
from ctc_forced_aligner import (
    load_alignment_model
)
import torch

compute_type = "float16"
# or run on GPU with INT8
# compute_type = "int8_float16"
# or run on CPU with INT8
# compute_type = "int8"
device = "cuda" if rp_cuda.is_available() else "cpu"

whisper_model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
# Directory to store pre-trained models
PRETRAINED_MODEL_DIR = "/models"
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

def download_model(model_name, url):
    model_path = os.path.join(PRETRAINED_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        wget.download(url, model_path)
    return model_path

def load_diarizer_models():
    # URLs for downloading the models
    vad_model_url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/vad_multilingual_marblenet/1.10.0/files?redirect=true&path=vad_multilingual_marblenet.nemo"
    speaker_model_url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/titanet_large/v1/files?redirect=true&path=titanet-l.nemo"

    # Download and cache models
    vad_model_path = download_model("vad_multilingual_marblenet.nemo", vad_model_url)
    speaker_model_path = download_model("titanet-l.nemo", speaker_model_url)

    return vad_model_path, speaker_model_path

def load_whisper_model(selected_model):
    """
    Load Faster Whisper model.
    """
    for _attempt in range(5):
        while True:
            try:
                loaded_model = WhisperModel(
                    selected_model, device=device, compute_type=compute_type)
                break
            except (AttributeError, OSError):
                continue
    return selected_model, loaded_model

def load_punctuation_model():
    """
    Load Punctuation model.
    """
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    return 'punctuation_model', punct_model

def load_alignment_model_wrapper():
    """
    Load CTC Forced Aligner model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return "alignment_model", (alignment_model, alignment_tokenizer)

def create_diarizer_config(output_dir, vad_model_path, speaker_model_path):
    # Load YAML Diarizer Configuration
    DOMAIN_TYPE = "telephonic"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Model Paths
    config.diarizer.speaker_embeddings.model_path = speaker_model_path
    config.diarizer.vad.model_path = vad_model_path
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
    
    return config

models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_whisper_model, whisper_model_names):
        if model_name is not None:
            models[model_name] = model

    for model_name, model in executor.map(lambda fn: fn(), [
        load_punctuation_model,
        load_alignment_model_wrapper,
    ]):
        if model_name is not None:
            models[model_name] = model
