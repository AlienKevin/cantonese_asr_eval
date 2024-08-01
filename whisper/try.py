import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

y, sr = librosa.load('../samples/yue.mp3', sr=16000)

MODEL_NAME = "alvanlii/whisper-small-cantonese"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

processed_in = processor(y, sampling_rate=sr, return_tensors="pt")
gout = model.generate(
    input_features=processed_in.input_features, 
    output_scores=True, return_dict_in_generate=True
)
transcription = processor.batch_decode(gout.sequences, skip_special_tokens=True)[0]
print(transcription)
