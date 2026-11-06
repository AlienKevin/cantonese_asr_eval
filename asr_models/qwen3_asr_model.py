import torch
from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel
from .asr_model import ASRModel


class Qwen3ASRModel(ASRModel):
    def __init__(self, model_name="Qwen/Qwen3-ASR-0.6B", device="cuda"):
        device_map = device if device != "cpu" else "cpu"
        self.model = _Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device_map,
            max_new_tokens=256,
        )
        self.model_name = model_name

    def generate(self, input):
        audio_inputs = []
        for wav in input:
            if wav.dim() > 1:
                wav = wav.squeeze(0)
            audio_inputs.append((wav.numpy(), 16000))
        results = self.model.transcribe(audio=audio_inputs, language="Cantonese")
        return [{"text": r.text} for r in results]

    def get_name(self):
        if "0.6B" in self.model_name:
            return "qwen3_asr_0.6b"
        elif "1.7B" in self.model_name:
            return "qwen3_asr_1.7b"
        return self.model_name.split("/")[-1].lower()
