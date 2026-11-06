import os
import sys
import torch
import torchaudio
from .asr_model import ASRModel

MIMO_REPO_PATH = os.environ.get("MIMO_REPO_PATH", "./MiMo-V2.5-ASR")
MIMO_MODEL_PATH = os.environ.get("MIMO_MODEL_PATH", "./models/MiMo-V2.5-ASR")
MIMO_TOKENIZER_PATH = os.environ.get("MIMO_TOKENIZER_PATH", "./models/MiMo-Audio-Tokenizer")


class MiMoASRModel(ASRModel):
    def __init__(
        self,
        model_path=MIMO_MODEL_PATH,
        tokenizer_path=MIMO_TOKENIZER_PATH,
        device=None,
    ):
        repo = os.path.abspath(MIMO_REPO_PATH)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.mimo_audio.mimo_audio import MimoAudio

        self.model = MimoAudio(
            model_path=model_path,
            mimo_audio_tokenizer_path=tokenizer_path,
            device=device,
        )

    def generate(self, input):
        results = []
        for audio in input:
            if isinstance(audio, torch.Tensor):
                waveform = audio.float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                waveform = torch.tensor(audio).float().unsqueeze(0)
            text = self.model.asr_sft(waveform, audio_tag="<chinese>")
            results.append({"text": text})
        return results

    def get_name(self):
        return "mimo_v2.5_asr"
