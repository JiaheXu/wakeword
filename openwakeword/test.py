import soundfile as sf
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model
import time
# Input files
AUDIO_FILE = "recorded.wav"


# Load audio as float32 for resampling
audio, sr = sf.read(AUDIO_FILE, dtype="float32")
print(f"Loaded {AUDIO_FILE} with samplerate {sr} Hz, {len(audio)} samples")

# If stereo, keep only the first channel
if audio.ndim > 1:
    audio = audio[:, 0]

# Resample to 16kHz if needed (still float32 here)
if sr != 16000:
    audio = resample_poly(audio, up=16000, down=sr).astype(np.float32)
    sr = 16000
    print(f"Resampled audio to {sr} Hz, {len(audio)} samples")

# Convert to int16 PCM for saving & model
audio_int16 = (audio * 32767).astype(np.int16)

# Optional: playback (float32 is fine for sounddevice)

# ===== Wake Word Detection =====
oww_model = Model(
    wakeword_models=["alexa_v0.1"]   # use built-in model
)

FRAME_LENGTH = 2048
THRESHOLD = 0.5

print("🔍 Running wake word detection...")

# Process in 512-sample frames
for i in range(0, len(audio_int16) - FRAME_LENGTH, FRAME_LENGTH):
    frame = audio_int16[i:i + FRAME_LENGTH]
    # preds = oww_model.predict(frame)
    start_time = time.time()
    preds = oww_model.predict(frame)
    end_time = time.time()

    print("inf took: ", end_time - start_time)
    for ww, score in preds.items():
        if score > THRESHOLD:
            print(f"Wake word '{ww}' detected! frame={i//FRAME_LENGTH}, score={score:.3f}")

print("✅ Detection finished")
