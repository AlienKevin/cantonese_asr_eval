from datasets import load_dataset
from transformers import pipeline
import json
from tqdm import tqdm
import soundfile as sf

# Load the Common Voice 16.0 Cantonese dataset
dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue")

# Initialize the model
MODEL_NAME = "alvanlii/whisper-small-cantonese"
lang = "zh"
device = "mps"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

# Process the dataset and generate transcriptions
transcriptions = []
batch_size = 64
audio_paths = list(dataset['test'])

print(f'Number of test audios: {len(audio_paths)}')

for i in tqdm(range(0, len(audio_paths), batch_size)):
    batch_audio_paths = [sample['path'] for sample in audio_paths[i:i + batch_size]]
    batch_expected_texts = [sample['sentence'] for sample in audio_paths[i:i + batch_size]]
    batch_waveforms = [sf.read(path)[0] for path in batch_audio_paths]
    res = pipe(batch_audio_paths)
    for j, audio_path in enumerate(batch_audio_paths):
        transcriptions.append({"audio": audio_path, "transcription": res[j]["text"], "expected": batch_expected_texts[j]})

# Save the transcriptions to a JSON file
with open("cv16_transcriptions.json", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)
