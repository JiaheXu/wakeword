import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.model import Model
from collections import deque
import threading, queue, os, time


# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = 2 * TARGET_SR       # 2s window = 32000 samples
STEP_SIZE = int(0.4 * TARGET_SR)  # 50 ms step = 800 samples
THRESHOLD = 0.5

SAVE_DIR = "detections"
ROLLING_DURATION = 2  # seconds of context to save before detection


# =========================
# Wake Word Detector
# =========================
class WakeWordDetector:
    def __init__(self, model_name="alexa_v0.1"):
        self.model = Model(wakeword_models=[model_name])

        self.resample_buffer = np.array([], dtype=np.int16)
        self.rolling_buffer = deque(maxlen=ROLLING_DURATION * TARGET_SR)
        self.detection_count = 0
        os.makedirs(SAVE_DIR, exist_ok=True)

    def save_detection(self,frame):
        """Save rolling buffer when wake word fires."""
        self.detection_count += 1
        filename = os.path.join(SAVE_DIR, f"detection_{self.detection_count}.wav")
        audio_array = np.array(frame, dtype=np.int16)
        sf.write(filename, audio_array, TARGET_SR, subtype="PCM_16")
        print(f"💾 Saved detection audio: {filename}")

    def process_frame(self, frame):
        """Run openWakeWord on one frame of audio."""
        preds = self.model.predict(frame)
        for ww, score in preds.items():
            print(f"{ww} score: {score:.3f}")
            if score > THRESHOLD:
                print(f"🚀 Wake word '{ww}' detected! score={score:.3f}")
                self.save_detection(frame)


# =========================
# Audio Producer (mic → queue)
# =========================
def audio_callback(indata, frames, time, status, q: queue.Queue = None, input_sr=TARGET_SR):
    if status:
        print(status)

    audio = np.squeeze(indata).astype(np.float32)

    q.put(audio)

# =========================
# Consumer Thread (detection)
# =========================
def detection_loop(q: queue.Queue, detector: WakeWordDetector):
    while True:
        samples = q.get()
        if samples is None:
            break

        # Append to rolling buffer
        for s in samples:
            detector.rolling_buffer.append(s)

        # Append to resample buffer
        detector.resample_buffer = np.concatenate((detector.resample_buffer, samples))

        # Process 2s windows with 50ms step
        while len(detector.resample_buffer) >= FRAME_LENGTH:
            frame = detector.resample_buffer[:FRAME_LENGTH]
            detector.process_frame(frame)
            detector.resample_buffer = detector.resample_buffer[STEP_SIZE:]


# =========================
# Main
# =========================
def main():
    q = queue.Queue()
    detector = WakeWordDetector("alexa_v0.1")
    # detector = WakeWordDetector("./zh/xiao_ai.tflite")
    # Detect nearest supported sample rate
    try:
        input_sr = TARGET_SR
        sd.check_input_settings(samplerate=TARGET_SR, channels=1, dtype="float32")
        print(f"✅ Using direct {TARGET_SR} Hz input")
    except Exception:
        device_info = sd.query_devices(kind="input")
        input_sr = int(device_info["default_samplerate"])
        print(f"⚠️  Device does not support 16kHz, using {input_sr} Hz with resampling")

    blocksize = int(0.05 * input_sr)  # 50ms block at input SR

    # Start consumer thread
    consumer_thread = threading.Thread(target=detection_loop, args=(q, detector), daemon=True)
    consumer_thread.start()

    # Start audio stream
    with sd.InputStream(
        samplerate=input_sr,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=lambda indata, frames, time, status: audio_callback(indata, frames, time, status, q, input_sr)
    ):
        # print("🎙️ Listening for wake word... (say 'alexa')")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            q.put(None)  # stop consumer thread


if __name__ == "__main__":
    main()
