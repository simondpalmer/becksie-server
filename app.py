from beam import endpoint, Image, Volume, env
import base64
from omegaconf import OmegaConf
import requests
from tempfile import NamedTemporaryFile, TemporaryDirectory
from nemo.collections.asr.models import NeuralDiarizer
import os
import json
import wget

BEAM_VOLUME_PATH = "./cached_models"


# These packages will be installed in the remote container
if env.is_remote():
    from faster_whisper import WhisperModel, download_model


# This runs once when the container first starts
def load_models():
    model_path = download_model("large-v3", cache_dir=BEAM_VOLUME_PATH)
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    return model


@endpoint(
    on_start=load_models,
    name="faster-whisper",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=Image(
        base_image="nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        python_version="python3.10",
        python_packages=["git+https://github.com/SYSTRAN/faster-whisper.git", "nemo_toolkit[asr]", "requests"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def transcribe(context, **inputs):
    # Retrieve cached model from on_start
    model = context.on_start_value

    # Inputs passed to API
    language = inputs.get("language")
    audio_base64 = inputs.get("audio_file")
    url = inputs.get("url")

    if audio_base64 and url:
        return {"error": "Only a base64 audio file OR a URL can be passed to the API."}
    if not audio_base64 and not url:
        return {
            "error": "Please provide either an audio file in base64 string format or a URL."
        }

    binary_data = None

    if audio_base64:
        binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    elif url:
        resp = requests.get(url)
        binary_data = resp.content

    text = ""

    with TemporaryDirectory() as tempdir:
        audio_file_path = os.path.join(tempdir, "audio.wav")
        
        # Write the audio data to a file in the temporary directory
        with open(audio_file_path, 'wb') as audio_file:
            audio_file.write(binary_data)
            audio_file.flush()

        try:
            segments, _ = model.transcribe(audio_file_path, beam_size=5, language=language)
            word_timestamps = []
            for segment in segments:
                word_timestamps.append({'start': segment.start, 'end': segment.end, 'text':segment.text})
        except Exception as e:
            return {"error": f"Something went wrong with Whisper transcription: {e}"}
            # full_transcript = "".join(segment.text for segment in transcript_segments)
        try:
            temp_path = os.path.join(tempdir, "temp_outputs")
            # Perform speaker diarization (pseudo-implementation)
            msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
            # msdd_model.speaker_embeddings.parameters.save_embeddings=True
            msdd_model.diarize()
        except Exception as e:
            return {"error": f"Something went wrong with Diarization: {e}"}
        try:
            # Reading timestamps <> Speaker Labels mapping
            speaker_ts = []
            with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line_list = line.split(" ")
                    s = int(float(line_list[5]) * 1000)
                    e = s + int(float(line_list[8]) * 1000)
                    speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

            diarization_result = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

            return {"text": text, "diarization": diarization_result}
        
        except Exception as e:
            return {"error": f"Something went wrong speaker mapping: {e}"}
        
def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": os.path.join(output_dir, "pred_rttms", "mono_file.rttm"),
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.speaker_embeddings.parameters.save_embeddings=True

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config

def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping

def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s