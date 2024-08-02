from transformers import pipeline
from .asr_model import ASRModel

class WhisperASRModel(ASRModel):
    def __init__(self, model_name, device):
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )

        self.model_name = model_name
        if model_name == "alvanlii/whisper-small-cantonese" or model_name == "Scrya/whisper-large-v2-cantonese":
            self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language="zh", task="transcribe")
            self.pipe.model.generation_config.suppress_tokens = None
        elif model_name == "openai/whisper-large-v3":
            pass
        else:
            raise ValueError(f"Invalid model_name {model_name}")

    def generate(self, input):
        if self.model_name == "alvanlii/whisper-small-cantonese" or self.model_name == "Scrya/whisper-large-v2-cantonese":
            results = self.pipe(input)
        elif self.model_name == "openai/whisper-large-v3":
            results = self.pipe(input, generate_kwargs={"language": "cantonese", task="transcribe"})
        else:
            raise ValueError(f"Invalid model_name {model_name}")
        return [{"text": result["text"]} for result in results]

    def get_name(self):
        if self.model_name == "alvanlii/whisper-small-cantonese":
            return "whisper_small_cantonese"
        elif self.model_name == "Scrya/whisper-large-v2-cantonese":
            return "whisper_large_v2_cantonese_scrya"
        elif self.model_name == "openai/whisper-large-v3":
            return "whisper_large_v3"
