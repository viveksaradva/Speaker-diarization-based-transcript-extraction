import os
import torchaudio
import tempfile
from transformers import pipeline
import glob
import json
import subprocess
import urllib.request
import logging
from pathlib import Path
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class SpeechProcessingPipeline:
    """
    A complete audio processing pipeline for:
    1. Converting audio to WAV format
    2. Transcribing speech using OpenAI Whisper
    3. Performing speaker diarization using NeMo
    4. Aligning transcription with diarization output
    """

    def __init__(self, input_audio: str, num_speakers: int = 2, model: str = "medium"):
        """
        Initializes the pipeline with input audio, number of speakers, and Whisper model.
        :param input_audio: Path to the input audio file.
        :param num_speakers: Estimated number of speakers.
        :param model: Whisper model type (tiny, base, small, medium, large, etc.).
        """
        self.input_audio = Path(input_audio)
        self.num_speakers = num_speakers
        self.model = model
        self.audio_stem = self.input_audio.stem
        self.wav_file = None
        self.transcript_json = None
        self.rttm_file = None
        self.diarized_transcript = None

    def convert_audio_to_wav(self):
        """Converts audio file to 16kHz mono WAV format if it's not already a WAV file."""
        output_path = f"{self.audio_stem}.wav"
        
        if self.input_audio.suffix == ".wav":
            self.wav_file = str(self.input_audio)
            logging.info(f"File is already in WAV format: {self.wav_file}")
            return


        logging.info(f"Converting {self.input_audio} to WAV format...")
        command = f"ffmpeg -i {self.input_audio} -ar 16000 -ac 1 {output_path} -y"
        subprocess.run(command, shell=True, check=True)
        
        self.wav_file = str(output_path)

    def parse_rttm(self):
        """Parses RTTM file and returns a list of speaker segments."""
        if not self.rttm_file:
            raise FileNotFoundError("RTTM file not found. Run diarization first.")
        
        segments = []
        with open(self.rttm_file, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
                speaker = parts[7]
                segments.append({"speaker": speaker, "start": start, "end": end})
        return segments
 
    @staticmethod
    def ensure_diarization_config():
        """Ensures diarization configuration file is available by downloading if necessary."""
        config_path = "diar_infer_telephonic.yaml"
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"

        if not os.path.exists(config_path):
            logging.info("Downloading diarization configuration file...")
            urllib.request.urlretrieve(config_url, config_path)
            logging.info("Download complete!")

        return config_path

    def perform_speaker_diarization(self):
        """Performs speaker diarization using NeMo and generates RTTM file."""
        if not self.wav_file:
            raise RuntimeError("WAV file not found. Ensure audio conversion was successful.")

        manifest_file = "manifest.json"
        config_path = self.ensure_diarization_config()

        metadata = {
            "audio_filepath": self.wav_file,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": self.num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None
        }
        with open(manifest_file, "w") as f:
            f.write(json.dumps(metadata) + "\n")

        # Load diarization config
        config = OmegaConf.load(config_path)
        config.diarizer.manifest_filepath = manifest_file
        config.diarizer.out_dir = "./"
        config.diarizer.speaker_embeddings.model_path = "titanet_large"

        logging.info("Running speaker diarization...")
        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()

    def find_rttm_file(self):
        """Finds the RTTM file corresponding to the audio file."""
        search_pattern = f"**/{self.audio_stem}.rttm"
        matching_files = glob.glob(search_pattern, recursive=True)

        if not matching_files:
            raise FileNotFoundError(f"No RTTM file found for `{self.audio_stem}`.")

        self.rttm_file = matching_files[0]
        logging.info(f"Found RTTM file: {self.rttm_file}")

    def transcribe_diarized_segments(self):
        """Transcribes individual speaker segments using Hugging Face Whisper ASR."""
        if not self.rttm_file:
            raise RuntimeError("Diarization must be completed before segment-level transcription.")

        # Load audio
        waveform, sample_rate = torchaudio.load(self.wav_file)

        # Parse speaker segments
        speaker_segments = self.parse_rttm()

        # Load ASR model
        logging.info("Loading Whisper ASR pipeline (transformers)...")
        asr = pipeline("automatic-speech-recognition", model=f"openai/whisper-{self.model}")

        results = []
        logging.info("Transcribing each speaker segment...")

        for segment in speaker_segments:
            speaker = segment["speaker"]
            start_time = segment["start"]
            end_time = segment["end"]

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_segment = waveform[:, start_sample:end_sample]

            # Save and transcribe temporary chunk
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                torchaudio.save(tmp.name, audio_segment, sample_rate)
                transcription_result = asr(tmp.name)
                transcription = transcription_result.get("text", "").strip()

            results.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "transcription": transcription
            })

        self.diarized_transcript = results

        return results

    def run_pipeline(self):
        """Runs the complete audio processing pipeline."""
        self.convert_audio_to_wav()
        self.perform_speaker_diarization()
        self.find_rttm_file()
        self.diarized_transcript = self.transcribe_diarized_segments()
        self.transcribe_diarized_segments()
        logging.info("Pipeline Complete! Diarized transcript is ready.")

        return self.diarized_transcript

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("input_audio", type=str, help="Path to input audio file")
    parser.add_argument("--num_speakers", type=int, default=2, help="Estimated number of speakers")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size")

    args = parser.parse_args()
    speech_pipeline = SpeechProcessingPipeline(args.input_audio, args.num_speakers, args.model)
    speech_pipeline.run_pipeline()
