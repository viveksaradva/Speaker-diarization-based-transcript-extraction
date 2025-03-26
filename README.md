# Information about Speaker diarization based transcript extraction from the audio file!

- I have made a class name `SpeechProcessingPipeline` that initialized the instance by taking `input_audio`(the path of audio file), `num_speakers` and `model`(the model type).
- `convert_audio_to_wav` function allow to convert the audio file to `.wav` format that'll be further used by the `whisper` model.
- `transcribe_with_whisper` function just runs the shell command with the help of `subprocess` lib and converts the audio into the json file that contains a whole transcripted in single chunk and the segments too.
- `ensure_diarization_config` function check if the `diar_infer_telephonic.yaml` is there or not and if it is not then it will downloaded from [github](https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml).
- `perform_speaker_diarization` function uses **NeMo's ClusteringDiarizer** to analyze the audio and identify different speakers. It ensures that the necessary diarization configuration file (`diar_infer_telephonic.yaml`) is available, then creates a **manifest file** (`manifest.json`) containing metadata about the audio file. After configuring the diarizer with **speaker embedding model** (`titanet_large`), it runs the diarization process, which generates an **RTTM file** containing speaker timestamps.
- `find_rttm_file` function **searches for the RTTM file** corresponding to the processed audio. It scans the entire project directory to locate the file using the audio filename as a reference. If no RTTM file is found, an error is raised.
- `align_transcription_with_speakers` function combines the Whisper transcript with the speaker diarization results from the RTTM file. It performs the following steps:
    1. Loads the **Whisper JSON transcript** and extracts **speech segments** (start time, end time, text).
    2. Reads the **RTTM file** and extracts **speaker segments** (start time, end time, speaker ID).
    3. Matches each transcript segment with the **most overlapping speaker segment** to assign speaker labels.
    4. Formats the transcript by prefixing each speech segment with its assigned speaker.
5. Saves the final **diarized transcript** to `diarized_transcript.txt`.