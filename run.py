from .asr_models.asr_model import ASRModel
from .asr_models.sensevoice_model import SenseVoiceASRModel
from .asr_models.whisper_model import WhisperASRModel
from .asr_datasets.common_voice import CommonVoiceDataset
from .asr_datasets.guangzhou_daily_use import GuangzhouDailyUseDataset
from .asr_datasets.guangzhou_cabin import GuangzhouCabinDataset
from .asr_datasets.asr_dataset import ASRDataset
import json

datasets: list[ASRDataset] = [
    CommonVoiceDataset(),
    GuangzhouDailyUseDataset(),
    GuangzhouCabinDataset(),
]

models: list[ASRModel] = [
    SenseVoiceASRModel(),
    WhisperASRModel()
]
import os

for dataset in datasets:
    dataset_name = dataset.get_name()
    for model in models:
        model_name = model.get_name()
        results = []
        for batch_audios, batch_sentences in dataset:
            transcriptions = model.generate(batch_audios)
            for transcription, sentence in zip(transcriptions, batch_sentences):
                results.append({"transcription": transcription["text"], "expected": sentence})

        # Create directory if it doesn't exist
        os.makedirs(f'results/{model_name}', exist_ok=True)
        
        # Save results to a JSON file
        with open(f'results/{model_name}/{dataset_name}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
