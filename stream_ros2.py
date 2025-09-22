#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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
FRAME_LENGTH = int(1.5 * TARGET_SR)
STEP_SIZE = int(0.1 * TARGET_SR)
WAKEWORD_THRESHOLD = 0.5
VAD_THRESHOLD = 2.0
VAD_START_LENGTH = int(1.5 * TARGET_SR)
ROLLBACK_SEC = 2
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)


class WakeWordVADDetector:
    def __init__(self, wakeword_model, vad_model, whisper_model, publisher):
        self.model = wakeword_model
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.publisher = publisher

        self.mode = "wakeword"
        self.audio_buffer = []
        self.wakeword_buffer = deque(maxlen=ROLLBACK_SEC * TARGET_SR)
        self.detection_count = 0
        self.start_time = None

    def save_segment(self):
        if len(self.audio_buffer) == 0:
            return None
        filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.wav")
        samples = np.array(self.audio_buffer, dtype=np.int16)
        sf.write(filename, samples, TARGET_SR, subtype="PCM_16")
        print(f"💾 Saved utterance: {filename}")
        self.audio_buffer = []
        return samples

    def transcribe(self, samples: np.ndarray):
        if self.whisper_model is None:
            print("⚠️ Whisper model not loaded, skipping transcription")
            return

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        segments, info = self.whisper_model.transcribe(
            samples.astype(np.float16), language="zh"
        )

        transcript_text = ""
        for seg in segments:
            line = f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}"
            print(line)
            transcript_text += seg.text.strip()

        if transcript_text:
            msg = String()
            msg.data = transcript_text
            self.publisher.publish(msg)
            print(f"📢 Published transcript: {transcript_text}")

    def process_wakeword(self):
        while len(self.audio_buffer) >= FRAME_LENGTH:
            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)
            preds = self.model.predict(frame)
            for ww, score in preds.items():
                if score > WAKEWORD_THRESHOLD:
                    self.detection_count += 1
                    print(f"🚀 Wakeword '{ww}' detected! Switching to VAD mode")
                    self.audio_buffer = self.audio_buffer[FRAME_LENGTH:]
                    self.mode = "vad"
                    self.start_time = time.time()
                    return
            self.audio_buffer = self.audio_buffer[STEP_SIZE:]

    def handle_audio(self, samples):
        self.audio_buffer.extend(samples)

        if self.mode == "wakeword":
            self.process_wakeword()

        elif self.mode == "vad":
            samples_norm = (samples / 32767).astype(np.float32)

            if len(self.audio_buffer) < VAD_START_LENGTH:
                return

            voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
            print(f"VAD prob: {voice_prob:.3f}")

            if voice_prob < VAD_THRESHOLD:
                utterance = self.save_segment()
                if utterance is not None:
                    self.transcribe(utterance)
                self.mode = "wakeword"
                self.audio_buffer = []


def audio_callback(indata, frames, time_info, status, q: queue.Queue, input_sr):
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
    buffer = np.array([], dtype=np.int16)
    target_len = int(0.1 * input_sr)

    while rclpy.ok():
        if q.empty():
            time.sleep(0.05)
            continue

        samples = q.get()
        if samples is None:
            break

        buffer = np.concatenate((buffer, samples))

        while len(buffer) >= target_len:
            chunk = buffer[:target_len]
            buffer = buffer[target_len:]
            if input_sr != TARGET_SR:
                chunk = resample_poly(chunk, TARGET_SR, input_sr).astype(np.float32)
            detector.handle_audio(chunk)


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")
        self.publisher_ = self.create_publisher(String, "user_speech", 10)

        # Load models
        vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
        vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
        whisper_model = WhisperModel("base")
        openwakeword_model = Model(wakeword_models=["alexa_v0.1"])
        test_frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
        openwakeword_model.predict(test_frame)

        self.detector = WakeWordVADDetector(
            wakeword_model=openwakeword_model,
            vad_model=vad_model,
            whisper_model=whisper_model,
            publisher=self.publisher_,
        )

        # Start audio
        self.q = queue.Queue()
        input_sr = 48000
        blocksize = int(0.02 * input_sr)
        device_index = sd.default.device[0]  # pick default input

        self.consumer_thread = threading.Thread(
            target=detection_loop, args=(self.q, self.detector, input_sr), daemon=True
        )
        self.consumer_thread.start()

        self.stream = sd.InputStream(
            samplerate=input_sr,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            device=device_index,
            callback=lambda indata, frames, time_info, status: audio_callback(
                indata, frames, time_info, status, self.q, input_sr
            ),
        )
        self.stream.start()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

