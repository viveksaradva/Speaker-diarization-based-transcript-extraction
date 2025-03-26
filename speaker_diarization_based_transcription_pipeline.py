import os
import glob
import json
import shutil
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
        output_path = self.input_audio.with_suffix(".wav")

        if self.input_audio.suffix == ".wav":
            logging.info(f"File is already in WAV format: {self.input_audio}")
            self.wav_file = str(output_path)
            return
        
        logging.info(f"Converting {self.input_audio} to WAV format...")
        command = f"ffmpeg -i {self.input_audio} -ar 16000 -ac 1 {output_path} -y"
        subprocess.run(command, shell=True, check=True)
        
        self.wav_file = str(output_path)

    def transcribe_with_whisper(self):
        """Transcribes speech from the WAV file using Whisper and saves as JSON."""
        if not self.wav_file:
            raise RuntimeError("WAV file not found. Ensure audio conversion was successful.")

        output_dir = Path("whisper_output")
        output_dir.mkdir(exist_ok=True)
        original_json = output_dir / f"{self.audio_stem}.json"
        renamed_json = output_dir / f"{self.audio_stem}_transcript.json"

        logging.info(f"Running Whisper transcription with model `{self.model}`...")
        command = f"whisper {self.wav_file} --model {self.model} --output_format json --output_dir {output_dir}"
        subprocess.run(command, shell=True, check=True)

        if original_json.exists():
            shutil.move(original_json, renamed_json)
            logging.info(f"Transcription saved as: {renamed_json}")
            self.transcript_json = str(renamed_json)
        else:
            raise FileNotFoundError("Whisper did not generate the expected JSON file.")

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

    def align_transcription_with_speakers(self):
        """Aligns the Whisper transcript with speaker diarization results."""
        if not (self.transcript_json and self.rttm_file):
            raise RuntimeError("Ensure both transcription and diarization are complete before alignment.")

        output_txt = "diarized_transcript.txt"

        # Load Whisper transcript
        with open(self.transcript_json, "r") as file:
            whisper_data = json.load(file)

        whisper_segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in whisper_data["segments"]
        ]

        # Load RTTM file
        with open(self.rttm_file, "r") as file:
            rttm_contents = file.readlines()

        speaker_segments = []
        for line in rttm_contents:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            speaker = parts[7]
            speaker_segments.append({"start": start_time, "end": end_time, "speaker": speaker})

        # Assign speakers to transcript
        def assign_speakers():
            result = []
            for seg in whisper_segments:
                best_match = max(speaker_segments, key=lambda spk: max(0, min(seg["end"], spk["end"]) - max(seg["start"], spk["start"])), default=None)
                speaker = best_match["speaker"] if best_match else "Unknown"
                result.append(f"[{speaker}] {seg['text']}")
            return "\n".join(result)

        formatted_transcript = assign_speakers()

        with open(output_txt, "w") as f:
            f.write(formatted_transcript)

        logging.info(f"Diarized transcript saved to {output_txt}")
        self.diarized_transcript = output_txt

    def run_pipeline(self):
        """Runs the complete audio processing pipeline."""
        self.convert_audio_to_wav()
        self.transcribe_with_whisper()
        self.perform_speaker_diarization()
        self.find_rttm_file()
        self.align_transcription_with_speakers()
        logging.info("Pipeline Complete! Diarized transcript is ready.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("input_audio", type=str, help="Path to input audio file")
    parser.add_argument("--num_speakers", type=int, default=2, help="Estimated number of speakers")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size")

    args = parser.parse_args()
    pipeline = SpeechProcessingPipeline(args.input_audio, args.num_speakers, args.model)
    pipeline.run_pipeline()
