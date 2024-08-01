from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import pipeline
import random

random.seed(42)

dataset = load_dataset("AlienKevin/guangzhou-cabin-speech")

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
audio_paths = random.sample(list(dataset['train']), 1000)

print(f'Number of test audios: {len(audio_paths)}')

for i in tqdm(range(0, len(audio_paths), batch_size)):
    batch_audios = [sample['audio'] for sample in audio_paths[i:i + batch_size]]
    batch_expected_texts = [sample['sentence'] for sample in audio_paths[i:i + batch_size]]
    res = pipe([audio['array'] for audio in batch_audios])
    for j, audio_path in enumerate(batch_audios):
        transcriptions.append({"audio": audio_path['path'], "transcription": res[j]["text"], "expected": batch_expected_texts[j]})

# Save the transcriptions to a JSON file
with open("cabin_transcriptions.json", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)
