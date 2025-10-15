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
STEP_SIZE = int(0.05 * TARGET_SR)  # 50 ms step = 800 samples
THRESHOLD = 0.5

SAVE_DIR = "detections"
ROLLING_DURATION = 2  # seconds of context to save before detection


# =========================
# Wake Word Detector
# =========================
class WakeWordDetector:
    def __init__(self, model_name="alexa_v0.1"):
        self.model = Model(wakeword_models=[model_name])

        self.resample_buffer = np.array([], dtype=np.float32)
        self.rolling_buffer = deque(maxlen=ROLLING_DURATION * TARGET_SR)
        self.detection_count = 0
        os.makedirs(SAVE_DIR, exist_ok=True)

    def save_detection(self, frame):
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
def audio_callback(indata, frames, time_info, status, q: queue.Queue = None, input_sr=TARGET_SR):
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


# =========================
# Consumer Thread (detection)
# =========================
def detection_loop(q: queue.Queue, detector: WakeWordDetector, input_sr):
    while True:
        samples = q.get()
        if samples is None:
            break

        # 🔽 Downsample to 16kHz if needed
        if input_sr != TARGET_SR:
            samples = resample_poly(samples, TARGET_SR, input_sr).astype(np.float32)

        # Append to rolling buffer
        detector.rolling_buffer.extend(samples)

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
    # detector = WakeWordDetector("alexa_v0.1")
    detector = WakeWordDetector("./zh/xiaobai.tflite")

    input_sr = 48000
    blocksize = int(0.02 * input_sr)  # 50ms block at input SR

    # Start consumer thread
    consumer_thread = threading.Thread(
        target=detection_loop, args=(q, detector, input_sr), daemon=True
    )
    consumer_thread.start()

    # Start audio stream
    with sd.InputStream(
        samplerate=input_sr,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=lambda indata, frames, time_info, status: audio_callback(indata, frames, time_info, status, q, input_sr),
    ):
        print("🎙️ Listening for wake word... (say 'alexa')")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            q.put(None)  # stop consumer thread


if __name__ == "__main__":
    main()
