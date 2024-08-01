# Character Error Rate
| Model           | CER on CV16 |
|-----------------|-------|
| [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 7.72% |
| [whisper-small-cantonese](https://huggingface.co/alvanlii/whisper-small-cantonese)   | 8.12% |

Note: CV16 is the yue split of Common Voice 16.0.

# Postprocessing for Evaluation

1. Convert all Chinese characters.
2. Strip away all punctuations.
3. Remove any emotion or event tags (sensevoice only).
