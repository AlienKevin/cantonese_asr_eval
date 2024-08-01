from datasets import load_dataset
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json
from tqdm import tqdm
import torch
import random

random.seed(42)

dataset = load_dataset("AlienKevin/guangzhou-cabin-speech")

# Initialize the model
model_dir = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
)

# Process the dataset and generate transcriptions
transcriptions = []
batch_size = 64
audios = random.sample(list(dataset['train']), 1000)

print(f'Number of test audios: {len(audios)}')

for i in tqdm(range(0, len(audios), batch_size)):
    batch_audios = [sample['audio'] for sample in audios[i:i + batch_size]]
    batch_expected_texts = [sample['sentence'] for sample in audios[i:i + batch_size]]
    res = model.generate(
        input=[item['array'] for item in batch_audios],
        cache={},
        language="yue",
        use_itn=True,
        batch_size_s=batch_size,
        merge_vad=True,
        merge_length_s=15,
    )
    for j, audio in enumerate(batch_audios):
        text = rich_transcription_postprocess(res[j]["text"])
        transcriptions.append({"audio": audio['path'], "transcription": text, "expected": batch_expected_texts[j]})

# Save the transcriptions to a JSON file
with open("cabin_transcriptions.json", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)
