import soundfile as sf
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model

# INPUT_FILE = "recorded.wav"
INPUT_FILE = "test_16k.wav"
OUTPUT_FILE = "recorded_16k.wav"



# ===== Wake Word Detection with openWakeWord =====
model = Model(
    wakeword_models=["alexa_v0.1"]   # use built-in model
)

result = model.predict_clip("./alexa_test.wav")
print("result: ", result)