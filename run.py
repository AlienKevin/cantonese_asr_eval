from asr_models.sensevoice_model import SenseVoiceASRModel
from asr_models.whisper_model import WhisperASRModel, WhisperASRModelWithNgram
from asr_models.mimo_model import MiMoASRModel
from asr_models.fireredasr_model import FireRedASRModel
from asr_models.qwen3_asr_model import Qwen3ASRModel
from asr_datasets.common_voice import CommonVoiceDataset
from asr_datasets.guangzhou_daily_use import GuangzhouDailyUseDataset
from asr_datasets.guangzhou_cabin import GuangzhouCabinDataset
from asr_datasets.zoengjyutgaai_saamgwokjinji import ZoengjyutgaaiSaamgwokjinjiDataset
from asr_datasets.wordshk_hiujin import WordshkHiujinDataset
from asr_datasets.mixed_cantonese_and_english import MixedCantoneseAndEnglishDataset
import argparse
import torch
import torchaudio
import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="+", metavar="MODEL_NAME", help="Model name(s) to run (default: all)")
args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
batch_size = 1

MODEL_REGISTRY = {
    "sensevoice_small": lambda: SenseVoiceASRModel(device=device),
    "whisper_small_cantonese": lambda: WhisperASRModel(model_name="alvanlii/whisper-small-cantonese", device=device),
    "whisper_large_v2_cantonese_scrya": lambda: WhisperASRModel(model_name="Scrya/whisper-large-v2-cantonese", device=device),
    "whisper_large_v2_cantonese_scrya_ngram": lambda: WhisperASRModelWithNgram(
        model_name="Scrya/whisper-large-v2-cantonese",
        lm_model="words.txt_correct.arpa",
        device=device,
    ),
    "mimo_v2.5_asr": lambda: MiMoASRModel(device=device),
    "fireredasr_aed_l": lambda: FireRedASRModel(
        model_type="aed",
        model_dir="pretrained_models/FireRedASR-AED-L",
        use_gpu=torch.cuda.is_available(),
    ),
    "qwen3_asr_0.6b": lambda: Qwen3ASRModel(model_name="Qwen/Qwen3-ASR-0.6B", device=device),
    "qwen3_asr_1.7b": lambda: Qwen3ASRModel(model_name="Qwen/Qwen3-ASR-1.7B", device=device),
}

DATASETS = [
    lambda: ZoengjyutgaaiSaamgwokjinjiDataset(batch_size=batch_size),
    lambda: GuangzhouDailyUseDataset(batch_size=batch_size),
    lambda: GuangzhouCabinDataset(batch_size=batch_size),
    lambda: CommonVoiceDataset(batch_size=batch_size),
    lambda: WordshkHiujinDataset(batch_size=batch_size),
    lambda: MixedCantoneseAndEnglishDataset(batch_size=batch_size),
]

selected_models = args.model if args.model else list(MODEL_REGISTRY.keys())
unknown = [m for m in selected_models if m not in MODEL_REGISTRY]
if unknown:
    parser.error(f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(MODEL_REGISTRY.keys())}")


def load_audio(item):
    """Normalise a dataset audio item to a 16kHz mono tensor."""
    if isinstance(item, str):
        wav, sr = torchaudio.load(item)
    else:
        wav = torch.tensor(item["array"]).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        sr = item["sampling_rate"]
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav


for dataset_factory in DATASETS:
    dataset = dataset_factory()
    dataset_name = dataset.get_name()

    for model_name in selected_models:
        model = MODEL_REGISTRY[model_name]()

        if os.path.exists(f"results/{model_name}/{dataset_name}.json"):
            print(f"Results for {model_name} on {dataset_name} already exist. Skipping...")
            continue

        results = []
        for batch_audios, batch_sentences in tqdm(
            dataset,
            desc=f"{model_name} on {dataset_name}",
            total=len(dataset) // batch_size,
        ):
            transcriptions = model.generate([load_audio(item) for item in batch_audios])
            for transcription, sentence in zip(transcriptions, batch_sentences):
                results.append(
                    {"transcription": transcription["text"], "expected": sentence}
                )

        os.makedirs(f"results/{model_name}", exist_ok=True)
        with open(f"results/{model_name}/{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
