import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.model import Model
import threading, queue, os, time
from collections import deque
import argparse

# VAD
from utils.vad import load_vad

# Whisper
from faster_whisper import WhisperModel


# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int(1.5 * TARGET_SR)   # ~2s window
STEP_SIZE = int(0.1 * TARGET_SR)      # 100ms step
WAKEWORD_THRESHOLD = 0.5

VAD_THRESHOLD = 2.0
VAD_START_LENGTH = int(1.5 * TARGET_SR)

MAX_UTTERANCE_SEC = 10
ROLLBACK_SEC = 2

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Detector
# =========================
class WakeWordVADDetector:
    def __init__(self, wakeword_model=None, vad_model=None, whisper_model=None):
        self.model =  wakeword_model
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.mode = "wakeword"  # or "vad"
        self.audio_buffer = []  # speech buffer
        self.wakeword_buffer = deque(maxlen=ROLLBACK_SEC * TARGET_SR)
        self.detection_count = 0
        self.start_time = None

    def save_segment(self):
        """Save captured utterance after VAD ends or timeout."""
        if len(self.audio_buffer) == 0:
            return None
        filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.wav")
        samples = np.array(self.audio_buffer, dtype=np.int16)
        sf.write(filename, samples, TARGET_SR, subtype="PCM_16")
        print(f"💾 Saved utterance: {filename}")
        self.audio_buffer = []
        return samples

    def transcribe(self, samples: np.ndarray):
        """Transcribe detected speech with faster-whisper."""
        if self.whisper_model is None:
            print("⚠️ Whisper model not loaded, skipping transcription")
            return

        # Convert int16 → float32 normalized
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        # Save float32 audio for reference
        filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}_f32.wav")
        sf.write(filename, samples.astype(np.float32), 16000, subtype="FLOAT")

        # Run transcription
        start = time.time()
        segments, info = self.whisper_model.transcribe(samples.astype(np.float16), language="zh")

        end = time.time()
        took = end - start
        print('whisper took: ', f'{ took:.3f}' )

        txt_file = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.txt")

        with open(txt_file, "w", encoding="utf-8") as f:
            print(f"📝 Transcript saved: {txt_file}")
            for seg in segments:
                line = f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}"
                print(line)
                f.write(line + "\n")

    def process_wakeword(self):
        """Run wakeword detection on sliding windows inside buffer."""
        while len(self.audio_buffer) >= FRAME_LENGTH:
            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)
            start = time.time()

            # preds = self.model.predict_clip(frame)
            # end = time.time()
            # took = end - start
            # print('wakeword took: ', f'{ took:.3f}' )
            # for item in preds:
            #     score = list( item.values() )[0]
            #     print(f"score: {score:.3f}")
            #     if score > WAKEWORD_THRESHOLD:
            #         self.detection_count += 1
            #         print(f"🚀 Wakeword detected! Switching to VAD mode")
            #         self.audio_buffer = self.audio_buffer[FRAME_LENGTH:]
            #         self.mode = "vad"
            #         self.start_time = time.time()
            #         return

            preds = self.model.predict(frame)
            end = time.time()
            took = end - start
            print('wakeword took: ', f'{ took:.3f}' )
            for ww, score in preds.items():
                print(f"{ww} score: {score:.3f}")
                if score > WAKEWORD_THRESHOLD:
                    self.detection_count += 1
                    print(f"🚀 Wakeword '{ww}' detected! Switching to VAD mode")
                    self.audio_buffer = self.audio_buffer[FRAME_LENGTH:]
                    self.mode = "vad"
                    self.start_time = time.time()
                    return
            # slide window
            self.audio_buffer = self.audio_buffer[STEP_SIZE:]

    def handle_audio(self, samples):
        self.audio_buffer.extend(samples)

        if self.mode == "wakeword":
            self.process_wakeword()

        elif self.mode == "vad":
            
            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)
            samples_norm = (samples / 32767).astype(np.float32)

            if len(self.audio_buffer) < VAD_START_LENGTH:
                return

            start = time.time()
            
            voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
            end = time.time()
            took = end - start
            print('VAD took: ', f'{ took:.3f}' )

            print(f"VAD prob: {voice_prob:.3f}")

            if voice_prob < VAD_THRESHOLD:
                # speech ended
                utterance = self.save_segment()
                if utterance is not None:
                    self.transcribe(utterance)
                self.mode = "wakeword"
                self.audio_buffer = []


# =========================
# Audio Producer
# =========================
def audio_callback(indata, frames, time_info, status, q: queue.Queue = None, input_sr=TARGET_SR):
    # print("audio callbacl time: ", time.time())
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


# =========================
# Consumer Thread
# =========================
# def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
#     print("🔁 Detection loop started")
#     while True:
#         if q.empty():
#             time.sleep(0.05)
#             continue

#         samples = q.get()
#         if samples is None:
#             break

#         # 🔽 Downsample to 16kHz if needed
#         if input_sr != TARGET_SR:
#             samples = resample_poly(samples, TARGET_SR, input_sr).astype(np.float32)

#         detector.handle_audio(samples)
#     print("⏹️ Detection loop stopped")
def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
    print("🔁 Detection loop started")

    buffer = np.array([], dtype=np.int16)   # accumulate audio here
    target_len = int(0.1 * input_sr)        # 0.1s worth of samples

    while True:
        if q.empty():
            time.sleep(0.1)
            continue

        samples = q.get()
        if samples is None:
            break

        # Append to buffer
        buffer = np.concatenate((buffer, samples))

        # Process in 0.1s chunks
        while len(buffer) >= target_len:
            chunk = buffer[:target_len]
            buffer = buffer[target_len:]

            # 🔽 Downsample to 16kHz if needed
            if input_sr != TARGET_SR:
                start = time.time()
                chunk = resample_poly(chunk, TARGET_SR, input_sr).astype(np.float32)
                end = time.time()
                # print(f"🔽 Resample took {end - start:.4f}s")

            detector.handle_audio(chunk)

    print("⏹️ Detection loop stopped")

# =========================
# Device Selection
# =========================
def list_devices():
    print("🎤 Available audio devices:")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        print(f"[{idx}] {dev['name']} (inputs={dev['max_input_channels']}, outputs={dev['max_output_channels']})")

def find_device(name_substring=None):
    """Search for a device by substring, or fall back to default input device."""
    devices = sd.query_devices()

    if name_substring:
        for idx, dev in enumerate(devices):
            if name_substring.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                print(f"✅ Using matched input device {idx}: {dev['name']}")
                return idx
        print(f"⚠️ Device with name containing '{name_substring}' not found, falling back to default input.")

    # fallback: system default input
    default_input = sd.default.device[0]  # (input, output)
    if default_input is not None and default_input >= 0:
        print(f"✅ Using default input device {default_input}: {devices[default_input]['name']}")
        return default_input

    raise RuntimeError("❌ No valid input device found.")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="USB",
                        help="Name substring of the input device (default: USB PnP Sound Device)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    q = queue.Queue()

    # Load VAD
    vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
    vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)  # warmup

    # Load Whisper (Chinese transcription)
    whisper_model = WhisperModel("base")
    openwakeword_model = Model(wakeword_models=['alexa_v0.1'])
    
    test_frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
    openwakeword_model.predict(test_frame)

    detector = WakeWordVADDetector( wakeword_model = openwakeword_model, vad_model=vad_model, whisper_model=whisper_model)

    input_sr = 48000
    blocksize = int(0.02 * input_sr)  # 100ms block at input SR

    # 🔎 Find device
    device_index = find_device(args.device)

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
        device=device_index,
        callback=lambda indata, frames, time_info, status: audio_callback(indata, frames, time_info, status, q, input_sr),
    ):
        print(f"🎙️ Listening for wake word... (device={args.device})")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            q.put(None)  # stop consumer thread


if __name__ == "__main__":
    main()
