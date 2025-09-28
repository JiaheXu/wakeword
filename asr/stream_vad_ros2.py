#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pino_msgs.srv import Text   # custom service

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import threading, queue, os, time

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
VAD_THRESHOLD = 2.0
VAD_START_LENGTH = int(1.5 * TARGET_SR)
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)


class VADDetector:
    def __init__(self, vad_model, whisper_model, publisher, response_pub, client, node):
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.publisher = publisher
        self.response_pub = response_pub
        self.client = client
        self.node = node

        self.audio_buffer = []
        self.detection_count = 0

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
            # Publish transcript
            msg = String()
            msg.data = transcript_text
            self.publisher.publish(msg)
            print(f"📢 Published transcript: {transcript_text}")

            # Call llm_service
            req = Text.Request()
            req.text = transcript_text
            future = self.client.call_async(req)

            def _callback(fut):
                try:
                    resp = fut.result()
                    if resp.success:
                        print(f"✅ LLM Response: {resp.response}")
                        out = String()
                        out.data = resp.response
                        self.response_pub.publish(out)
                        print(f"📢 Published LLM response: {resp.response}")
                    else:
                        print(f"⚠️ LLM Service returned failure: {resp.response}")
                except Exception as e:
                    print(f"❌ Service call exception: {e}")

            future.add_done_callback(_callback)

    def handle_audio(self, samples):
        self.audio_buffer.extend(samples)

        samples_norm = (samples / 32767).astype(np.float32)

        if len(self.audio_buffer) < VAD_START_LENGTH:
            return

        voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
        print(f"VAD prob: {voice_prob:.3f}")

        if voice_prob < VAD_THRESHOLD:
            utterance = self.save_segment()
            if utterance is not None:
                self.transcribe(utterance)
            self.audio_buffer = []


def audio_callback(indata, frames, time_info, status, q: queue.Queue, input_sr):
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


def detection_loop(q: queue.Queue, detector: VADDetector, input_sr):
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


def find_device(name_substring=None):
    devices = sd.query_devices()
    if name_substring:
        for idx, dev in enumerate(devices):
            if name_substring.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                print(f"✅ Using matched input device {idx}: {dev['name']}")
                return idx
    default_input = sd.default.device[0]
    if default_input is not None and default_input >= 0:
        print(f"✅ Using default input device {default_input}: {devices[default_input]['name']}")
        return default_input
    raise RuntimeError("❌ No valid input device found.")


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")

        # Load models
        vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
        vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
        whisper_model = WhisperModel("large-v3", device='cuda')
        
        print("loaded VAD and ASR")
        
        # Publishers
        self.publisher_ = self.create_publisher(String, "user_speech", 10)
        self.response_pub = self.create_publisher(String, "llm_response", 10)

        # LLM client
        self.cli = self.create_client(Text, "llm_service")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("⏳ Waiting for llm_service...")
        print("found service !!!")

        self.detector = VADDetector(
            vad_model=vad_model,
            whisper_model=whisper_model,
            publisher=self.publisher_,
            response_pub=self.response_pub,
            client=self.cli,
            node=self,
        )

        # Audio stream
        self.q = queue.Queue()
        input_sr = 48000
        blocksize = int(0.02 * input_sr)
        device_index = find_device("USB PnP Sound Device")

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

