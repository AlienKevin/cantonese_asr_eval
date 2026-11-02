import os
import tempfile
import wave

import numpy as np

from .asr_model import ASRModel


class MlxQwen3ASRModel(ASRModel):
    """Qwen3-ASR running under Apple MLX, quantised to 8-bit at load time.

    Apple Silicon only.  Requires `pip install mlx-qwen3-asr`.

    This is the same checkpoint as `Qwen3ASRModel`, but the weights are
    quantised in memory (`quantize_model`, group_size 64) instead of being run
    at bfloat16, so the model fits and decodes in real time on a laptop.  It is
    reported as a separate row rather than folded into the bfloat16 row, so the
    quantisation loss stays visible rather than assumed.

    `language` defaults to "Cantonese" to match `Qwen3ASRModel`, which keeps the
    row a like-for-like quantisation comparison.  Pass `language=None` for the
    detect-the-language configuration; both runs are in `results/`.

    `bits` is 4 or 8 to quantise, or None to run unquantised at `dtype`.

    The audio path (16-bit WAV, float16) is the one the committed
    `qwen3_asr_1.7b_mlx_8bit*` results were produced with; see Discussion
    point 6 in the README for how it differs from the bfloat16 row.
    """

    TARGET_SR = 16000
    DTYPES = ("float16", "bfloat16", "float32")

    def __init__(
        self,
        model_name="Qwen/Qwen3-ASR-1.7B",
        dtype="float16",
        bits=8,
        group_size=64,
        language="Cantonese",
    ):
        if bits not in (None, 4, 8):
            raise ValueError("bits must be 4, 8 or None (unquantised), got %r" % (bits,))
        if dtype not in self.DTYPES:
            raise ValueError("dtype must be one of %s, got %r" % (", ".join(self.DTYPES), dtype))

        import mlx.core as mx
        import mlx_qwen3_asr as asr

        mx_dtype = getattr(mx, dtype)

        if bits is not None:
            from mlx_qwen3_asr.convert import quantize_model

            model, _ = asr.load_model(model_name, dtype=mx_dtype)
            model = quantize_model(model, bits=int(bits), group_size=int(group_size))
            # Pass dtype here too: Session otherwise falls back to its own default.
            self.session = asr.Session(model, dtype=mx_dtype, tokenizer_model=model_name)
        else:
            self.session = asr.Session(model_name, dtype=mx_dtype)

        self.model_name = model_name
        self.dtype = dtype
        self.bits = bits
        self.language = language

    def _write_wav(self, arr):
        arr = np.asarray(arr, dtype="float64")
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.TARGET_SR)
            fh.writeframes(pcm.tobytes())
        return path

    def generate(self, input):
        results = []
        for wav in input:
            if hasattr(wav, "numpy"):
                wav = wav.squeeze(0) if wav.dim() > 1 else wav
                arr = wav.numpy()
            else:
                arr = np.asarray(wav)
            path = self._write_wav(arr)
            try:
                out = self.session.transcribe(
                    path, language=self.language, context=""
                )
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            text = getattr(out, "text", None)
            if text is None:
                text = str(out)
            results.append({"text": text.strip()})
        return results

    def get_name(self):
        # Matches the run.py registry keys, e.g. qwen3_asr_1.7b_mlx_8bit[_auto].
        size = "1.7b" if "1.7B" in self.model_name else "0.6b"
        precision = "%dbit" % self.bits if self.bits is not None else self.dtype
        name = "qwen3_asr_%s_mlx_%s" % (size, precision)
        return name if self.language else name + "_auto"
