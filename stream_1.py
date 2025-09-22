import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.model import Model
import threading, queue, os, time
from collections import deque

# VAD
from utils.vad import load_vad


# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int( 1.5 * TARGET_SR )       # 2s window = 32000 samples



STEP_SIZE = int(0.1 * TARGET_SR)  # 50 ms step = 800 samples
WAKEWORD_THRESHOLD = 0.5

VAD_THRESHOLD = 0.2
VAD_START_LENGTH= int( 1.5 * TARGET_SR)       # 2s window = 32000 samples

MAX_UTTERANCE_SEC = 10  # force finalize if longer than this
ROLLBACK_SEC = 2        # save N seconds before wakeword

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Detector
# =========================
class WakeWordVADDetector:
    def __init__(self, model_name="alexa_v0.1", vad_model=None):
        self.model = Model(wakeword_models=[model_name])
        self.vad_model = vad_model
        self.mode = "wakeword"  # "wakeword" or "vad"
        # self.mode = "vad"  # "wakeword" or "vad"
        self.audio_buffer = []  # speech buffer
        self.wakeword_buffer = deque(maxlen=ROLLBACK_SEC * TARGET_SR)  # rolling buffer
        self.detection_count = 0
        self.start_time = None

    def save_wakeword(self):
        """Save the rolling buffer at wakeword trigger."""
        if len(self.wakeword_buffer) == 0:
            return
        filename = os.path.join(SAVE_DIR, f"wakeword_{self.detection_count}.wav")
        sf.write(filename, np.array(self.wakeword_buffer, dtype=np.int16), TARGET_SR, subtype="PCM_16")
        print(f"💾 Saved wakeword audio: {filename}")

    def save_segment(self):
        """Save captured utterance after VAD ends or timeout."""
        if len(self.audio_buffer) == 0:
            return
        filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.wav")

        samples = np.array(self.audio_buffer, dtype=np.int16)


        # samples = (samples * 32767).astype(np.int16)
        sf.write(filename, samples, TARGET_SR, subtype="PCM_16")
        print(f"💾 Saved utterance: {filename}")
        self.audio_buffer = []
        return samples
    def transcribe(self, audio):
    	return

    def process_wakeword(self):
        """Run wakeword detection on 2s sliding windows inside buffer."""
        while len(self.audio_buffer) >= FRAME_LENGTH:
            
            

            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)

            preds = self.model.predict(frame)
            for ww, score in preds.items():
                print(f"{ww} score: {score:.3f}")
                if score > WAKEWORD_THRESHOLD:
                    self.detection_count += 1
                    print(f"🚀 Wakeword '{ww}' detected! Switching to VAD mode")
                    # Save wakeword audio from rolling buffer
                    # self.save_wakeword()
                    # Keep remainder after the wakeword frame
                    self.audio_buffer = self.audio_buffer[FRAME_LENGTH:]
                    self.mode = "vad"
                    self.start_time = time.time()
                    return
            # slide window
            self.audio_buffer = self.audio_buffer[STEP_SIZE:]

    def handle_audio(self, samples):

        self.audio_buffer.extend(samples)

        if self.mode == "wakeword":
            # self.audio_buffer.extend(samples)
            # self.wakeword_buffer.extend(samples)
            # print('len: ', len(self.audio_buffer) )
            self.process_wakeword()

        elif self.mode == "vad":
            print("in vad")
            samples = (samples / 32767).astype(np.float32)

            if len(self.audio_buffer) < VAD_START_LENGTH:
                return
            # # check timeout
            # if time.time() - self.start_time > MAX_UTTERANCE_SEC:
            #     print("⏱️ Timeout reached, finalizing utterance")
            #     self.save_segment()
            #     self.mode = "wakeword"
            #     self.audio_buffer = []
            #     return

            # run VAD
            voice_prob = float(self.vad_model(samples, sr=TARGET_SR).flatten()[0])
            print(f"VAD prob: {voice_prob:.3f}")
            if voice_prob < VAD_THRESHOLD:
                # speech ended
                audio = self.save_segment()
                
                self.transcribe( samples )
                
                self.mode = "wakeword"
                self.audio_buffer = []


# =========================
# Audio Producer
# =========================
def audio_callback(indata, frames, time_info, status, q: queue.Queue = None, input_sr=TARGET_SR):
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


# =========================
# Consumer Thread (non-blocking)
# =========================
def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
    while True:
        if q.empty():
            time.sleep(0.01)  # avoid busy loop
            continue

        samples = q.get()
        if samples is None:
            break

        # 🔽 Downsample to 16kHz if needed
        if input_sr != TARGET_SR:
            samples = resample_poly(samples, TARGET_SR, input_sr).astype(np.float32)

        detector.handle_audio(samples)


# =========================
# Main
# =========================
def main():
    q = queue.Queue()

    # Load VAD
    vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
    vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)  # warmup

    detector = WakeWordVADDetector("alexa_v0.1", vad_model=vad_model)
    # detector = WakeWordVADDetector("./zh/xiao_ai.tflite", vad_model=vad_model)

    input_sr = 48000
    blocksize = int(0.1 * input_sr)  # 50ms block at input SR

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

