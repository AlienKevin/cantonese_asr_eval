import os
import sys
import tempfile
import torch
import torchaudio
from .asr_model import ASRModel

FIREREDASR_REPO_PATH = os.environ.get("FIREREDASR_REPO_PATH", "./FireRedASR")


class FireRedASRModel(ASRModel):
    def __init__(self, model_type="aed", model_dir="pretrained_models/FireRedASR-AED-L", use_gpu=True):
        repo = os.path.abspath(FIREREDASR_REPO_PATH)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from fireredasr.models.fireredasr import FireRedAsr
        self.model = FireRedAsr.from_pretrained(model_type, model_dir)
        self.model_type = model_type
        self._config = self._default_config(use_gpu)

    def _default_config(self, use_gpu):
        base = {"use_gpu": 1 if use_gpu else 0, "beam_size": 3, "decode_max_len": 0}
        if self.model_type == "aed":
            return {**base, "nbest": 1, "softmax_smoothing": 1.25, "aed_length_penalty": 0.6, "eos_penalty": 1.0}
        else:
            return {**base, "decode_min_len": 0, "repetition_penalty": 3.0, "llm_length_penalty": 1.0, "temperature": 1.0}

    def generate(self, input):
        tmp_paths = []
        try:
            for audio in input:
                if isinstance(audio, torch.Tensor):
                    waveform = audio.float().cpu()
                else:
                    waveform = torch.tensor(audio).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                torchaudio.save(tmp.name, waveform, 16000)
                tmp.close()
                tmp_paths.append(tmp.name)

            uttids = [str(i) for i in range(len(tmp_paths))]
            results = self.model.transcribe(uttids, tmp_paths, self._config)
            return [{"text": r["text"]} for r in results]
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

    def get_name(self):
        return f"fireredasr_{self.model_type}_l"
