#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

try:
    import whisper
except Exception as exc:
    print("OpenAI whisper not installed. Run: pip install -U openai-whisper")
    raise


def main():
    model_name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cuda")

    print(f"Loading whisper model: {model_name} on {device}")
    model = whisper.load_model(model_name, device=device)

    # 3s silence @16k for warm-up
    warmup_audio = np.zeros(3 * 16000, dtype=np.float32)
    runs = int(os.getenv("WHISPER_RUNS", "5"))
    print(f"Running warm-up transcription for {runs} runs...")
    last_text = ""
    for i in range(1, runs + 1):
        start = time.perf_counter()
        result = model.transcribe(
            warmup_audio,
            language="zh",
            task="transcribe",
            beam_size=3,
            fp16=(device == "cuda"),
        )
        elapsed = time.perf_counter() - start
        last_text = (result.get("text") or "").strip()
        print(f"Run {i}: {elapsed:.3f}s")

    print("Warm-up done.")
    print(f"Last transcript: {last_text!r}")


if __name__ == "__main__":
    sys.exit(main())
