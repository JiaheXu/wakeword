#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# from pino_msgs.srv import Text   # custom service

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.model import Model
import threading, queue, os, time
from collections import deque
from opencc import OpenCC
# VAD
from utils.vad import load_vad
# Whisper
from faster_whisper import WhisperModel
from pathlib import Path

home_dir = str( Path.home() )

# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int(2.0 * TARGET_SR)
STEP_SIZE = int(0.15 * TARGET_SR)
WAKEWORD_THRESHOLD = 0.06
VAD_THRESHOLD = 0.6
VAD_START_LENGTH = int(1.5 * TARGET_SR)

VAD_LENGTH = 0.2

SILENT_LENGTH = 1.0
ROLLBACK_SEC = 2
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

class WakeWordVADDetector:
    def __init__(self, wakeword_model, vad_model, whisper_model, publisher, response_pub, client, node):
        self.model = wakeword_model
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.publisher = publisher       # publishes raw transcript
        self.response_pub = response_pub # publishes LLM responses
        self.client = client             # service client to llm_service
        self.node = node

        # self.noise_keywords = [ '字幕', '订阅' ,'谢谢大家' , 'CANADA']
        self.publisher_ = self.create_publisher(String, "raw_input", 10)
        self.cc = OpenCC('t2s')
        self.mode = "vad"
        self.audio_buffer = []
        self.wakeword_buffer = deque(maxlen=ROLLBACK_SEC * TARGET_SR)
        self.detection_count = 0
        self.start_time = None

        self.last_word = time.time()
        self.last_none_word = time.time()
        self.last_detect = time.time()
        self.last_speech_end = time.time()
        self.last_cmd_time = time.time() - 10.0

    def save_segment(self, save = False):
        if len(self.audio_buffer) == 0:
            return None
        samples = np.array(self.audio_buffer, dtype=np.int16)

        if(save):
            filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.wav")    
            sf.write(filename, samples, TARGET_SR, subtype="PCM_16")
            print(f"💾 Saved utterance: {filename}")
        self.audio_buffer = []
        return samples

    def traditional_to_simplified(self, text: str) -> str:
        """
        将输入的繁体中文字符串转换为简体中文
        """
        return self.cc.convert(text)

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
            transcript_text += self.traditional_to_simplified(seg.text.strip())

        if transcript_text:
            msg = String()
            msg.data = transcript_text
            # 现在发布到 raw_input
            self.publisher.publish(msg)
            print(f"📢 Published transcript to raw_input: {transcript_text}")

    def process_wakeword(self):
        while len(self.audio_buffer) >= FRAME_LENGTH:
            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)
            
            start = time.time()
            preds = self.model.predict(frame)
            # preds = self.model.predict(frame)
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
                    self.last_detect = time.time()
                    return
            self.audio_buffer = self.audio_buffer[STEP_SIZE:]

    def handle_audio(self, samples):

        samples_norm = (samples / 32767).astype(np.float32)

        start = time.time()
        voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
        end = time.time()
        print("VAD took: ", end - start)
        print(f"VAD prob: {voice_prob:.3f}")

        if voice_prob < VAD_THRESHOLD:
            self.last_none_word = time.time()

            if(self.last_none_word - self.last_word > SILENT_LENGTH):
                # start = time.time()
                utterance = self.save_segment()
                # end = time.time()
                # print("save audio took: ", end - start)
                if utterance is not None:
                    self.transcribe(utterance)
                # self.mode = "wakeword"
                self.audio_buffer = []
        else:
            self.audio_buffer.extend(samples)
            self.last_word = time.time()


def audio_callback(indata, frames, time_info, status, q: queue.Queue, input_sr):
    # print("audio_callback")
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
    buffer = np.array([], dtype=np.int16)
    target_len = int( VAD_LENGTH * input_sr)

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

class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")



        # Load models
        vad_model = load_vad(  home_dir + "/model_data/silero_vad.onnx")
        vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
        whisper_model = WhisperModel(home_dir + "/model_data/faster-whisper-large-v3", device='cuda')

        #openwakeword_model = Model(wakeword_models=["alexa_v0.1"])
        openwakeword_model = Model(wakeword_models=["./zh/xiaobai.tflite"])
        
        test_frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
        openwakeword_model.predict(test_frame)
        
        print('finished model loading!!!!')

        # Publisher for raw transcripts
        self.publisher_ = self.create_publisher(String, "user_speech", 10)

        # Publisher for LLM responses
        self.response_pub = self.create_publisher(String, "llm_response", 10)

        # Service client to LLM service
        self.cli = self.create_client(Text, "llm_service")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("⏳ Waiting for llm_service...")
        
        print("found service")

        self.detector = WakeWordVADDetector(
            wakeword_model=openwakeword_model,
            vad_model=vad_model,
            whisper_model=whisper_model,
            publisher=self.publisher_,      # topic: user_speech
            response_pub=self.response_pub, # topic: llm_response
            client=self.cli,                # service client
            node=self,
        )
        print("finished ros init part")
        # Start audio stream


        self.q = queue.Queue()
        input_sr = 48000
        blocksize = int(0.02 * input_sr)
        device_index = find_device('USB') #("USB PnP Sound Device")
        # device_index = sd.default.device[0]  # pick default input

        self.consumer_thread = threading.Thread(
            target=detection_loop, args=(self.q, self.detector, input_sr), daemon=True
        )
        self.consumer_thread.start()
        # print("start init stream")
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

