from datasets import load_dataset
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json
from tqdm import tqdm
import torch

# Load the Common Voice 16.0 Cantonese dataset
dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue")

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
audio_paths = list(dataset['test'])

print(f'Number of test audios: {len(audio_paths)}')

for i in tqdm(range(0, len(audio_paths), batch_size)):
    batch_audio_paths = [sample['path'] for sample in audio_paths[i:i + batch_size]]
    batch_expected_texts = [sample['sentence'] for sample in audio_paths[i:i + batch_size]]
    res = model.generate(
        input=batch_audio_paths,
        cache={},
        language="yue",
        use_itn=True,
        batch_size_s=batch_size,
        merge_vad=True,
        merge_length_s=15,
    )
    for j, audio_path in enumerate(batch_audio_paths):
        text = rich_transcription_postprocess(res[j]["text"])
        transcriptions.append({"audio": audio_path, "transcription": text, "expected": batch_expected_texts[j]})

# Save the transcriptions to a JSON file
with open("cv16_transcriptions.json", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)
