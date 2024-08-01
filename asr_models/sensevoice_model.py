from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
from .asr_model import ASRModel

class SenseVoiceASRModel(ASRModel):
    def __init__(self, model_dir, device):
        self.model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device
        )

    def generate(self, input):
        res = self.model.generate(
            input=input,
            cache={},
            language="yue",
            use_itn=True,
            batch_size_s=64,
            merge_vad=True,
            merge_length_s=15,
        )
        return [{"text": rich_transcription_postprocess(r["text"])} for r in res]
