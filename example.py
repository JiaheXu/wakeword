import soundfile as sf
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model
import time

# INPUT_FILE = "recorded.wav"

INPUT_FILE = "test_input_48k.wav"

# INPUT_FILE = 'alexa_test.wav'
OUTPUT_FILE = "resampled_16k.wav"

# ===== Wake Word Detection with openWakeWord =====
model = Model(
    wakeword_models=["alexa_v0.1"]   # use built-in model
)

samples, sr = sf.read(INPUT_FILE, dtype="float32")
print('sr: ', sr )
# Resample to 16kHz for processing
samples_resampled = resample_poly(samples, 16000, sr).astype(np.float32)
    
audio_int16 = (samples_resampled * 32767).astype(np.int16)

# Save resampled audio
sf.write(OUTPUT_FILE, audio_int16, 16000)
print(f"✅ Saved resampled audio to {OUTPUT_FILE}")

# start = time.time()
# preds = model.predict(audio_int16)
# start = time.time()
# for ww, score in preds.items():
#     print(f"{ww} score: {score:.3f}")


start = time.time()
preds = model.predict_clip(OUTPUT_FILE)
start = time.time()
for item in preds:
    val = list( item.values() )[0]
    if(val > 0.5):
        print(f"val: {val:.3f}")

end = time.time()

print("used time: ", end - start)
