"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import re
import tempfile
import torchaudio
import wget
import platform
import signal
# Check if running on Windows and apply a workaround
if platform.system() == 'Windows':
    if not hasattr(signal, 'SIGKILL'):
        signal.SIGKILL = signal.SIGTERM  # or SIGINT
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import numpy as np
import logging
import json
from omegaconf import OmegaConf
import torch
import faster_whisper
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.utils import format_timestamp
from ctc_forced_aligner import (
    generate_emissions, 
    preprocess_text,
    get_alignments, 
    get_spans, 
    postprocess_results,
    load_alignment_model,
)
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from faster_whisper.utils import format_timestamp
from deepmultilingualpunctuation import PunctuationModel
from runpod.serverless.utils import rp_cuda

from utils import (
    find_numeral_symbol_tokens,
    langs_to_iso,
    convert_and_save_audio,
    get_words_speaker_mapping,
    punct_model_langs,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript
)


device = "cuda" if rp_cuda.is_available() else "cpu"


class Predictor:
    """ A Predictor class for all the models """

    def __init__(self, vad_model_path, speaker_model_path):
        self.models = {}
        self.vad_model_path = vad_model_path
        self.speaker_model_path = speaker_model_path

    def create_diarizer_config(self, temp_path):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        model_config_path = os.path.join(temp_path, "diar_infer_telephonic.yaml")

        if not os.path.exists(model_config_path):
            wget.download(config_url, temp_path)

        config = OmegaConf.load(model_config_path)
        data_dir = os.path.join(temp_path, "data")
        os.makedirs(data_dir, exist_ok=True)

        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"

        # Update the config with model paths
        # config.diarizer.speaker_embeddings.model_path = self.speaker_model_path
        # config.diarizer.vad.model_path = self.vad_model_path
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.vad.model_path = pretrained_vad

        # Prepare input manifest for a specific audio file
        meta = {
            "audio_filepath": os.path.join(temp_path, "mono_file.wav"),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        config_file = os.path.join(data_dir, "input_manifest.json")
        config.diarizer.manifest_filepath = config_file
        config.diarizer.out_dir = data_dir
        
        return config

    def load_model(self, model_name, model_loader):
        model = model_loader(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        return (model_name, model)

    def setup(self):
            # Update models dictionary with models initialized under the keys
            model_name, model_instance = self.load_model('base', lambda name: WhisperModel(name, device=device, compute_type='int8'))
            self.models[model_name] = model_instance

            # model_name, model_instance = self.load_model('punctuation_model', lambda _: PunctuationModel(model="kredor/punctuate-all"))
            # self.models[model_name] = model_instance

            model_name, model_instance = self.load_model('alignment_model', lambda _: load_alignment_model(device))
            self.models[model_name] = model_instance

            # # Assume create_diarizer_config is a method or separate function that prepares configurations
            # diarizer_cfg = self.create_diarizer_config()
            # model_name, model_instance = self.load_model('neural_diarizer', lambda _: NeuralDiarizer(cfg=diarizer_cfg).to(device))
            # self.models[model_name] = model_instance

            # Load diaritization related paths
            self.models.update({
                'vad_model_path': self.vad_model_path,
                'speaker_model_path': self.speaker_model_path
            })

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False
    ):
        """
        Run a single prediction on the model
        """
        whisper_model: WhisperModel = self.models.get(model_name)
        if not whisper_model:
            raise ValueError(f"Model '{model_name}' not found.")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # OPTIONS
        # Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
        enable_stemming = True
        # replaces numerical digits with their pronounciation, increases diarization accuracy
        suppress_numerals = True
        batch_size = 8

        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        audio_waveform = faster_whisper.decode_audio(audio)
        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )
        if batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=batch_size,
                without_timestamps=True,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                without_timestamps=True,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)
        # clear gpu vram
        del whisper_pipeline
        torch.cuda.empty_cache()

        # Aligning the transcription with the original audio using Forced Alignment
        alignment_model, alignment_tokenizer = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        audio_waveform = (
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device)
        )

        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=batch_size
        )

        del alignment_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Convert audio to mono for NeMo combatibility 
        temp_dir = tempfile.mkdtemp()
        mono_file_path = os.path.join(temp_dir, "mono_file.wav")
        torchaudio.save(
            mono_file_path,
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True,
        )
        # convert_and_save_audio(audio_waveform, temp_dir)
        
        # Speaker Diarization using NeMo MSDD Model
        msdd_model = NeuralDiarizer(cfg=self.create_diarizer_config(temp_dir)).to(device)
        # Example if using DataLoader:
        if hasattr(msdd_model, 'data_loader'):
            msdd_model.data_loader.num_workers = 0
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        # Mapping Speakers to Sentences According to Timestamps
        speaker_ts = []
        with open(os.path.join(temp_dir, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        
        # Realligning Speech segments using Punctuation
        if info.language in punct_model_langs:
            # restoring punctuation in the transcript to help realign the sentences
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
        
        get_speaker_aware_transcript(ssm, f)
        transcription = segments = list(segments)

        # transcription = format_segments(transcription, segments)

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription,
            "translation":  None,
            "device": device,
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
            results["word_timestamps"] = word_timestamps


        return results


def serialize_segments(transcript):
    '''
    Serialize the segments to be returned in the API response.
    '''
    return [{
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob
    } for segment in transcript]
