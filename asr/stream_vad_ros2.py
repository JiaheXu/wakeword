#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
# from pino_msgs.srv import Text   # custom service

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
# from openwakeword.model import Model
import threading, queue, os, time, sys, random
import json
from collections import deque
from opencc import OpenCC
from pino_msgs.msg import AudioMSG
# VAD
from utils.vad import load_vad
# Whisper
from faster_whisper import WhisperModel
from pathlib import Path

home_dir = str(Path.home())

# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int(2.0 * TARGET_SR)
STEP_SIZE = int(0.15 * TARGET_SR)
WAKEWORD_THRESHOLD = 0.06
VAD_THRESHOLD = 0.7
VAD_START_LENGTH = int(1.5 * TARGET_SR)

VAD_LENGTH = 0.2

SILENT_LENGTH = 0.5
ROLLBACK_SEC = 1
MAX_AUDIO_SEC = 5
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)
WARMUP_CONFIG_PATH = Path(__file__).with_name("online_warmup.json")
DEFAULT_ONLINE_WARMUP = [
    "好的，客官的这个问题问得我的大脑一阵空白，让我稍微冷静一下再回答你。",
    "卖锅的，这个问题问得真是刁钻，我得用我这聪明的脑子好好想一想。",
    "这个问题问得我快死机了,让我的cpu再加速运转一下,给您一个合理的答案!",
    "哎呀，额的神啊，真没想到你会问这个问题，这个我得查一下我的小本本！",
    "这个问题有点意思，看来您不是一般人啊，您得是二班的吧。哈哈哈",
    "听了您这个问题，我脑瓜子嗡嗡的，再多给我一点时间琢磨一下",
    "您这个问题问得太好了，您是第一个提出这个问题的游客，待我给您细细道来",
    "这位客官，请让我想想，好像唐明皇当年也提过同样的问题。",
    "恭喜您成为第八百八十八个提出这个问题的人，请稍等片刻，我得给你一个与众不同的答案。",
    "哎呀，这个问题可巧是问对人了，我敢说整个大唐芙蓉园也只有我能够给出最完美的答案了。"
]


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    oom_signals = (
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "std::bad_alloc",
        "cannot allocate memory",
    )
    return any(s in text for s in oom_signals)


def _handle_fatal_exception(exc_type, exc, tb):
    if issubclass(exc_type, MemoryError) or _is_oom_error(exc):
        print(f"❌ OOM detected ({exc_type.__name__}), exiting process.")
        os._exit(1)
    sys.__excepthook__(exc_type, exc, tb)


def _threading_excepthook(args):
    _handle_fatal_exception(args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _handle_fatal_exception
threading.excepthook = _threading_excepthook


def load_online_warmup(path: Path) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load warmup config from {path}: {e}; using defaults")
        return list(DEFAULT_ONLINE_WARMUP)

    if isinstance(data, dict):
        items = data.get("online_warmup", [])
    elif isinstance(data, list):
        items = data
    else:
        print(f"⚠️ Invalid warmup config root type in {path}; using defaults")
        return list(DEFAULT_ONLINE_WARMUP)

    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        print(f"⚠️ Warmup config in {path} is empty; using defaults")
        return list(DEFAULT_ONLINE_WARMUP)
    return cleaned


class WakeWordVADDetector:
    def __init__(self, wakeword_model, vad_model, whisper_model, publisher, response_pub, client, node, audio_pub):
        self.model = wakeword_model
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.publisher = publisher       # publishes raw transcript
        self.response_pub = response_pub # publishes LLM responses
        self.client = client             # service client to llm_service
        self.node = node
        self.audio_pub = audio_pub

        #self.publisher_ = self.node.create_publisher(String, "raw_input", 10)
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
        self.last_vad_log_time = 0.0
        self.online_warmup = load_online_warmup(WARMUP_CONFIG_PATH)

    def publish_online_warmup(self):
        if self.audio_pub is None or not self.online_warmup:
            return
        msg = AudioMSG()
        msg.cmd = "speak"
        msg.text = random.choice(self.online_warmup)
        msg.voice = "zf_xiaoyi"
        msg.volume = 3.0
        msg.speed = 0.9
        self.audio_pub.publish(msg)
        self.node.get_logger().info(f"🎙️ Published warmup AudioMSG: {msg.text}")

    def save_segment(self, save=False):
        if len(self.audio_buffer) == 0:
            return None
        samples = np.array(self.audio_buffer, dtype=np.int16)

        if save:
            filename = os.path.join(SAVE_DIR, f"speech_{self.detection_count}.wav")
            sf.write(filename, samples, TARGET_SR, subtype="PCM_16")
            print(f"💾 Saved utterance: {filename}")
        self.audio_buffer = []
        return samples

    def traditional_to_simplified(self, text: str) -> str:
        """繁体 → 简体"""
        return self.cc.convert(text)

    def transcribe(self, samples: np.ndarray):
        if self.whisper_model is None:
            print("⚠️ Whisper model not loaded, skipping transcription")
            return

        transcribe_start = time.time()

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        current_lang = "zh"
        print(f"🌐 Transcribing with language: {current_lang}")
        try:
            segments, info = self.whisper_model.transcribe(
                samples.astype(np.float32),
                language=current_lang,
                task="transcribe",
                beam_size=3,
            )
            detected_lang = current_lang

            transcript_text = ""
            for seg in segments:
                text = seg.text.strip()
                if detected_lang.startswith("zh"):
                    text = self.traditional_to_simplified(text)
                transcript_text += text

            if transcript_text:
                msg = String()
                msg.data = transcript_text
                self.publisher.publish(msg)
                print(f"📢 Published transcript to raw_input: {transcript_text}")
        except Exception as e:
            if _is_oom_error(e):
                print(f"❌ OOM during transcription: {e}")
                os._exit(1)
            raise
        finally:
            elapsed = time.time() - transcribe_start
            audio_sec = len(samples) / TARGET_SR
            print(f"⏱️ Transcribe time cost: {elapsed:.3f}s (audio={audio_sec:.2f}s)")

    def process_wakeword(self):
        while len(self.audio_buffer) >= FRAME_LENGTH:
            frame = np.array(self.audio_buffer[:FRAME_LENGTH], dtype=np.float32)
            preds = self.model.predict(frame)

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

        voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
        now = time.time()
        if voice_prob >= 0.5 and (now - self.last_vad_log_time) >= 5.0:
            print(f"VAD prob: {voice_prob:.3f}")
            self.last_vad_log_time = now
        if voice_prob < VAD_THRESHOLD:
            self.last_none_word = time.time()
            if self.last_none_word - self.last_word > SILENT_LENGTH:
                utterance = self.save_segment()
                if utterance is not None:
                    self.publish_online_warmup()
                    self.transcribe(utterance)
                self.audio_buffer = []
        else:
            self.audio_buffer.extend(samples)
            self.last_word = time.time()
            if len(self.audio_buffer) >= int(MAX_AUDIO_SEC * TARGET_SR):
                utterance = self.save_segment()
                if utterance is not None:
                    self.publish_online_warmup()
                    self.transcribe(utterance)
                self.audio_buffer = []


def audio_callback(indata, frames, time_info, status, q: queue.Queue, input_sr):
    if status:
        print(status)
    audio = np.squeeze(indata).astype(np.float32)
    q.put(audio)


def detection_loop(q: queue.Queue, detector: WakeWordVADDetector, input_sr):
    buffer = np.array([], dtype=np.int16)
    target_len = int(VAD_LENGTH * input_sr)

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
    devices = sd.query_devices()
    if name_substring:
        for idx, dev in enumerate(devices):
            if name_substring.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                print(f"✅ Using matched input device {idx}: {dev['name']}")
                return idx
        print(f"⚠️ Device with name containing '{name_substring}' not found, fallback to default")

    default_input = sd.default.device[0]  # (input, output)
    if default_input is not None and default_input >= 0:
        print(f"✅ Using default input device {default_input}: {devices[default_input]['name']}")
        return default_input

    raise RuntimeError("❌ No valid input device found.")


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")

        # Track speaker state
        self.speaker_playing = False
        self.create_subscription(Bool, "speaker_playing", self.speaker_cb, 10)

        # Load models
        try:
            vad_model = load_vad(home_dir + "/model_data/silero_vad.onnx")
            vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
            whisper_model = WhisperModel(home_dir + "/model_data/faster-whisper-large-v3", device='cuda')
        except Exception as e:
            if _is_oom_error(e):
                print(f"❌ OOM during model initialization: {e}")
                os._exit(1)
            raise
        # whisper_model = WhisperModel(home_dir + "/model_data/faster-distil-whisper-large-v3", device='cuda')
        # whisper_model = WhisperModel(home_dir + "/model_data/faster-whisper-base", device='cuda')        
        # Warm-up whisper model to reduce first-utterance latency
        try:
            warmup_audio = np.zeros(TARGET_SR, dtype=np.float32)  # 1s of silence @16k
            list(
                whisper_model.transcribe(
                    warmup_audio,
                    language="zh",
                    task="transcribe",
                    beam_size=1,
                )[0]
            )
            print("✅ Whisper model warm-up complete")
        except Exception as e:
            print(f"⚠️ Whisper model warm-up failed: {e}")
        # openwakeword_model = Model(wakeword_models=["./zh/xiaobai.tflite"])
        # openwakeword_model.predict(np.zeros(FRAME_LENGTH, dtype=np.float32))
        print("✅ Finished model loading")

        # Publishers
        self.publisher_ = self.create_publisher(String, "user_speech", 10)
        self.response_pub = self.create_publisher(String, "llm_response", 10)
        self.audio_pub = self.create_publisher(AudioMSG, "audio_cmd", 10)

        # Service client
        # self.cli = self.create_client(String, "llm_service")  # keep placeholder
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info("⏳ Waiting for llm_service...")
        print("✅ Found LLM service")

        self.detector = WakeWordVADDetector(
            wakeword_model=None,
            vad_model=vad_model,
            whisper_model=whisper_model,
            publisher=self.publisher_,
            response_pub=self.response_pub,
            client=None,
            node=self,
            audio_pub=self.audio_pub,
        )

        # Audio queue + stream
        self.q = queue.Queue()
        input_sr = 48000
        blocksize = int(0.02 * input_sr)
        device_index = find_device("USB")

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
            callback=lambda indata, frames, time_info, status: self.audio_cb(
                indata, frames, time_info, status, self.q, input_sr
            ),
        )
        self.stream.start()

    def speaker_cb(self, msg: Bool):
        """Callback for /speaker_playing Bool topic."""
        self.speaker_playing = msg.data
        if msg.data:
            # Flush buffers
            with self.q.mutex:
                dropped = len(self.q.queue)
                self.q.queue.clear()
            self.detector.audio_buffer = []
            self.get_logger().info(f"🔇 Speaker playing → mic input disabled, flushed {dropped} chunks")
        
        return

    def audio_cb(self, indata, frames, time_info, status, q: queue.Queue, input_sr):
        """Audio callback that respects speaker_playing state."""
        if self.speaker_playing:
            return  # ignore while speaker is active
        if status:
            print(status)
        audio = np.squeeze(indata).astype(np.float32)
        q.put(audio)


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
