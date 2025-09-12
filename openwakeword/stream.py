import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.model import Model
from collections import deque
import os
import time
# Audio constants
INPUT_SR = 44100         # mic sample rate
TARGET_SR = 16000        # required by openWakeWord
FRAME_LENGTH = 1 * TARGET_SR       # 2s = 32000 samples
STEP_SIZE = int(0.3 * TARGET_SR)  # 50ms = 800 samples
BLOCKSIZE = int(0.05* INPUT_SR)   # capture 50ms @44.1kHz
THRESHOLD = 0.5

# Detection audio saving
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)
ROLLING_DURATION = 3  # seconds of audio to save before detection
rolling_buffer = deque(maxlen=ROLLING_DURATION * TARGET_SR)  # 3s @16kHz
detection_count = 0

# Load wake word model
oww_model = Model(
    wakeword_models=["alexa_v0.1"]
)

# Buffer for accumulating resampled audio
resample_buffer = np.array([], dtype=np.int16)
count = 0

start = time.time()
def save_detection_audio(frame):
    """Save the last few seconds of audio when detection fires."""
    global detection_count
    detection_count += 1
    filename = os.path.join(SAVE_DIR, f"detection_{detection_count}.wav")
    audio_array = np.array(frame, dtype=np.int16)
    sf.write(filename, audio_array, TARGET_SR, subtype="PCM_16")
    print(f"💾 Saved detection audio: {filename}")

def audio_callback(indata, frames, timer, status):
    global resample_buffer
    global count
    if status:
        print(status)

    # audio_44k = np.squeeze(indata).astype(np.float32)
    # audio_16k = resample_poly(audio_44k, up=16000, down=INPUT_SR).astype(np.float32)
    # audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
    # resample_buffer = np.concatenate((resample_buffer, audio_16k_int16))
    # print('len: ', len(resample_buffer) )
    count+=1
    end = time.time()
    print('count: ', count)
    print('time: ', end - start)
    print("")
    # Process 2s sliding window, step forward 50ms
    # while len(resample_buffer) >= FRAME_LENGTH:
    #     frame = resample_buffer[:FRAME_LENGTH]
        
    #     # start_time = time.time()
    #     preds = oww_model.predict(frame)
    #     # end_time = time.time()

    #     print("timer")
    #     for ww, score in preds.items():
    #         print(f"{ww} score: {score:.3f}")
    #         if score > THRESHOLD:
    #             print(f"🚀 Wake word '{ww}' detected! score={score:.3f}")
    #             save_detection_audio(frame)
    #             break

    #     # Slide window forward by 50ms
    #     resample_buffer = resample_buffer[STEP_SIZE:]

# Open mic stream
with sd.InputStream(
    # samplerate=INPUT_SR,
    blocksize=BLOCKSIZE,
    dtype="float32",
    channels=1,
    callback=audio_callback
):
    print("🎙️ Listening for wake word... (say 'alexa')")
    while True:
        pass