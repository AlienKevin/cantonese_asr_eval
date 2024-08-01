from transformers import pipeline
from .asr_model import ASRModel

class WhisperASRModel(ASRModel):
    def __init__(self, device, model_name="alvanlii/whisper-small-cantonese", lang = "zh"):
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

    def generate(self, input):
        results = self.pipe(input)
        return [{"text": result["text"]} for result in results]

    def get_name(self):
        return "whisper_small_cantonese"
