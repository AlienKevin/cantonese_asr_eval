from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from .asr_model import ASRModel

class SenseVoiceASRModel(ASRModel):
    def __init__(self, device, model_path="FunAudioLLM/SenseVoiceSmall"):
        self.model = AutoModel(
            model=model_path,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
        )

    def generate(self, input):
        res = self.model.generate(
            input=input,
            cache={},
            language="yue",
            use_itn=True,
            merge_vad=True,
            merge_length_s=15,
        )
        return [{"text": rich_transcription_postprocess(r["text"])} for r in res]

    def get_name(self):
        return "sensevoice_small"
