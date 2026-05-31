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
import subprocess
import re
import json
import shutil
from collections import deque
from opencc import OpenCC
from pino_msgs.msg import AudioMSG
# VAD
from utils.vad import load_vad
# FunASR
from funasr import AutoModel
from pathlib import Path

home_dir = str(Path.home())

# =========================
# Configuration
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int(2.0 * TARGET_SR)
STEP_SIZE = int(0.15 * TARGET_SR)
WAKEWORD_THRESHOLD = 0.06
VAD_THRESHOLD = 0.85
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


class NoUsableInputDeviceError(RuntimeError):
    pass


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
    def __init__(
        self,
        wakeword_model,
        vad_model,
        asr_model,
        publisher,
        response_pub,
        client,
        node,
        audio_pub,
        transcribe_state_cb=None,
    ):
        self.model = wakeword_model
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.publisher = publisher       # publishes raw transcript
        self.response_pub = response_pub # publishes LLM responses
        self.client = client             # service client to llm_service
        self.node = node
        self.audio_pub = audio_pub
        self.transcribe_state_cb = transcribe_state_cb

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
        self.is_transcribing = False

    def _set_transcribing(self, state: bool):
        self.is_transcribing = state
        if self.transcribe_state_cb is None:
            return
        try:
            self.transcribe_state_cb(state)
        except Exception as e:
            self.node.get_logger().warn(f"Failed to update transcribe state: {e}")

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

    @staticmethod
    def count_chinese_characters(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    def transcribe(self, samples: np.ndarray):
        if self.asr_model is None:
            print("⚠️ FunASR model not loaded, skipping transcription")
            return

        transcribe_start = time.time()
        self._set_transcribing(True)

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        # print("🌐 Transcribing with FunASR (Fun-ASR-Nano-2512)")
        try:
            res = self.asr_model.generate(
                input=samples.astype(np.float32),
                batch_size_s=300,
                language="zh",
            )

            transcript_text = ""
            if res and len(res) > 0:
                raw_text = res[0].get("text", "").strip()
                transcript_text = self.traditional_to_simplified(raw_text)

            transcript_len = len(transcript_text)
            if transcript_text:
                chinese_char_count = self.count_chinese_characters(transcript_text)
                if chinese_char_count >= 8:
                    self.publish_online_warmup()
                else:
                    self.node.get_logger().info(
                        f"Skipping warmup AudioMSG: transcript has {chinese_char_count} Chinese chars (<8)"
                    )
                msg = String()
                msg.data = transcript_text
                self.publisher.publish(msg)
                print(
                    f"📢 Published transcript to raw_input (len={transcript_len}): {transcript_text}"
                )
            else:
                self.node.get_logger().info(
                    "FunASR returned empty transcript (len=0); skipping publish and warmup"
                )
        except Exception as e:
            if _is_oom_error(e):
                print(f"❌ OOM during transcription: {e}")
                os._exit(1)
            raise
        finally:
            elapsed = time.time() - transcribe_start
            audio_sec = len(samples) / TARGET_SR
            print(f"⏱️ Transcribe time cost: {elapsed:.3f}s (audio={audio_sec:.2f}s)")
            self._set_transcribing(False)

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

        try:
            voice_prob = float(self.vad_model(samples_norm, sr=TARGET_SR).flatten()[0])
        except Exception as e:
            print(f"[VAD] Fatal error: {e}")
            sys.exit(1)
        now = time.time()
        if voice_prob >= VAD_THRESHOLD and (now - self.last_vad_log_time) >= 5.0:
            print(f"VAD prob: {voice_prob:.3f}")
            self.last_vad_log_time = now
        if voice_prob < VAD_THRESHOLD:
            self.last_none_word = time.time()
            if self.last_none_word - self.last_word > SILENT_LENGTH:
                utterance = self.save_segment()
                if utterance is not None:
                    self.transcribe(utterance)
                self.audio_buffer = []
        else:
            self.audio_buffer.extend(samples)
            self.last_word = time.time()
            if len(self.audio_buffer) >= int(MAX_AUDIO_SEC * TARGET_SR):
                utterance = self.save_segment()
                if utterance is not None:
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


def start_pulse_capture(
    q: queue.Queue,
    source: str,
    input_sr: int,
    node: Node,
):
    chunk_frames = max(256, int(0.02 * input_sr))
    chunk_bytes = chunk_frames * 2  # int16 mono
    cmd = [
        "parec",
        "-d",
        source,
        "--rate",
        str(input_sr),
        "--channels",
        "1",
        "--format",
        "s16le",
        "--latency-msec",
        "20",
        "--raw",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise NoUsableInputDeviceError("USE_BT=true requires 'parec' (pulseaudio-utils).") from e

    def _reader():
        while rclpy.ok() and proc.poll() is None:
            if proc.stdout is None:
                break
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            q.put(samples)
        if proc.poll() not in (0, None):
            err = ""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    err = ""
            node.get_logger().error(f"❌ parec exited unexpectedly code={proc.poll()} err='{err}'")

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    return proc, th


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


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_bt_mac_from_autoconnect_script() -> str | None:
    script_path = _workspace_root() / "auto_connect_bt.sh"
    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return None

    m = re.search(r'^\s*DEVICE="([0-9A-Fa-f:]{17})"', text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).lower()


def _get_bt_device_name_from_mac(bt_mac: str) -> str | None:
    if shutil.which("bluetoothctl") is None:
        return None
    try:
        out = subprocess.run(
            ["bluetoothctl", "info", bt_mac],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        ).stdout
    except Exception:
        return None

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name: "):
            return line.split("Name: ", 1)[1].strip()
    return None


def _get_default_pulse_source() -> str | None:
    if shutil.which("pactl") is None:
        return None
    try:
        out = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        ).stdout.strip()
    except Exception:
        return None
    return out or None


def _parse_forced_device_selection() -> int | str | None:
    raw = os.getenv("ASR_INPUT_DEVICE", "").strip()
    if not raw:
        return None
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _is_blocked_input_device_name(name: str) -> bool:
    return "orin" in name.lower()


def _resolve_forced_input_device(selection: int | str):
    devices = sd.query_devices()
    if isinstance(selection, int):
        if (
            0 <= selection < len(devices)
            and devices[selection]["max_input_channels"] > 0
            and not _is_blocked_input_device_name(devices[selection]["name"])
        ):
            print(f"✅ Using forced input device index {selection}: {devices[selection]['name']}")
            return selection
        raise NoUsableInputDeviceError(
            f"❌ ASR_INPUT_DEVICE index {selection} is not a valid input device "
            "(or device name contains 'Orin')."
        )

    selection_lower = selection.lower()
    for idx, dev in enumerate(devices):
        if (
            dev["max_input_channels"] > 0
            and selection_lower in dev["name"].lower()
            and not _is_blocked_input_device_name(dev["name"])
        ):
            print(f"✅ Using forced input device name match {idx}: {dev['name']}")
            return idx
    raise NoUsableInputDeviceError(
        f"❌ ASR_INPUT_DEVICE='{selection}' did not match any allowed input device "
        "(devices containing 'Orin' are blocked)."
    )


def find_bluetooth_input_device(bt_mac: str | None):
    forced = _parse_forced_device_selection()
    if forced is not None:
        return _resolve_forced_input_device(forced)

    devices = sd.query_devices()
    default_input = sd.default.device[0]
    disable_bt_probe = os.getenv("ASR_DISABLE_BT_PROBE", "0").strip().lower() in ("1", "true", "yes", "on")
    if disable_bt_probe:
        bt_mac = None
    bt_name = _get_bt_device_name_from_mac(bt_mac) if bt_mac else None
    default_source = _get_default_pulse_source()

    mac_underscore = bt_mac.replace(":", "_").lower() if bt_mac else None
    mac_compact = bt_mac.replace(":", "").lower() if bt_mac else None
    source_tokens = []
    if default_source:
        source_tokens = [t for t in default_source.lower().split(".") if t]

    def _is_match(dev_name: str) -> bool:
        name = dev_name.lower()
        if mac_underscore and mac_underscore in name:
            return True
        if mac_compact and mac_compact in name:
            return True
        if bt_name and bt_name.lower() in name:
            return True
        if any(tok in name for tok in source_tokens):
            return True
        return False

    if bt_mac:
        for idx, dev in enumerate(devices):
            if (
                dev["max_input_channels"] > 0
                and _is_match(dev["name"])
                and not _is_blocked_input_device_name(dev["name"])
            ):
                print(
                    f"✅ Using Bluetooth input device {idx}: {dev['name']} "
                    f"(mac={bt_mac}, default_source={default_source})"
                )
                return idx
        print(f"⚠️ No Bluetooth input matched mac={bt_mac}, trying PulseAudio input")

        # On many remote Jetson setups, PortAudio exposes Bluetooth mic via "pulse"
        # instead of a "bluez_*" device name.
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if (
                dev["max_input_channels"] > 0
                and "pulse" in name
                and not _is_blocked_input_device_name(dev["name"])
            ):
                print(f"✅ Using PulseAudio input device {idx}: {dev['name']}")
                return idx

        print(f"⚠️ PulseAudio input device not found, fallback to default input")

    if (
        default_input is not None
        and default_input >= 0
        and not _is_blocked_input_device_name(devices[default_input]["name"])
        and devices[default_input]["max_input_channels"] > 0
    ):
        print(f"✅ Using default input device {default_input}: {devices[default_input]['name']}")
        return default_input

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and not _is_blocked_input_device_name(dev["name"]):
            print(f"✅ Using first allowed input device {idx}: {dev['name']}")
            return idx

    raise NoUsableInputDeviceError(
        "❌ No valid input device found (all candidates may be blocked by 'Orin' filter)."
    )


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")
        self.stream = None
        self.pulse_proc = None
        self.pulse_thread = None

        # Track speaker state
        self.speaker_playing = False
        self.transcribing = False
        self.last_audio_debug_time = 0.0
        self.create_subscription(Bool, "speaker_playing", self.speaker_cb, 10)

        # Load models
        try:
            vad_model = load_vad(home_dir + "/model_data/silero_vad.onnx")
            vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
            asr_model = AutoModel(
                # model="paraformer-zh",
                # model = "FunAudioLLM/Fun-ASR-Nano-2512",  # requires latest funasr (pip install -U funasr)
                model = "iic/SenseVoiceSmall",
                # trust_remote_code = True,
                device="cuda",
            )
        except Exception as e:
            if _is_oom_error(e):
                print(f"❌ OOM during model initialization: {e}")
                os._exit(1)
            raise

        # Warm-up FunASR model to reduce first-utterance latency
        try:
            warmup_audio = np.zeros(TARGET_SR, dtype=np.float32)  # 1s of silence @16k
            asr_model.generate(input=warmup_audio, batch_size_s=300)
            print("✅ FunASR model warm-up complete")
        except Exception as e:
            print(f"⚠️ FunASR model warm-up failed: {e}")
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
            asr_model=asr_model,
            publisher=self.publisher_,
            response_pub=self.response_pub,
            client=None,
            node=self,
            audio_pub=self.audio_pub,
            transcribe_state_cb=self.on_transcribe_state_change,
        )

        # Audio queue + stream
        self.q = queue.Queue()
        input_sr = 48000
        blocksize = int(0.02 * input_sr)
        use_bt = _env_flag("USE_BT", default=False)
        forced_device = os.getenv("ASR_INPUT_DEVICE", "").strip()
        disable_bt_probe = _env_flag("ASR_DISABLE_BT_PROBE", default=False)
        usb_device_hint = os.getenv("USB_INPUT_DEVICE", "USB").strip()

        self.get_logger().info(f"🔀 USE_BT={'true' if use_bt else 'false'}")
        if forced_device:
            self.get_logger().info(f"🎯 ASR_INPUT_DEVICE override active: '{forced_device}'")
        if disable_bt_probe:
            self.get_logger().info("⏭️ ASR_DISABLE_BT_PROBE active: skipping Bluetooth/Pulse probing")

        if use_bt:
            bt_mac = _load_bt_mac_from_autoconnect_script()
            if bt_mac:
                self.get_logger().info(f"🔎 auto_connect_bt.sh target MAC: {bt_mac}")
            else:
                self.get_logger().warn("⚠️ Could not read DEVICE from auto_connect_bt.sh; using default input selection")
            pulse_source = os.getenv("BT_PULSE_SOURCE", "").strip() or _get_default_pulse_source()
            if not pulse_source:
                raise NoUsableInputDeviceError("❌ USE_BT=true but no Pulse default source found.")
            self.get_logger().info(f"🎙️ Bluetooth Pulse source='{pulse_source}'")
            input_sr = 16000
            self.pulse_proc, self.pulse_thread = start_pulse_capture(
                self.q, pulse_source, input_sr, self
            )
            self.get_logger().info("✅ Started Bluetooth capture via parec")
        else:
            self.get_logger().info(f"🎧 USB input mode active (hint='{usb_device_hint}')")
            device_index = find_device(usb_device_hint or None)
            dev_info = sd.query_devices(device_index)
            self.get_logger().info(f"🎤 Input device={device_index} name='{dev_info['name']}'")

        self.consumer_thread = threading.Thread(
            target=detection_loop, args=(self.q, self.detector, input_sr), daemon=True
        )
        self.consumer_thread.start()

        if not use_bt:
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

    def on_transcribe_state_change(self, is_transcribing: bool):
        self.transcribing = is_transcribing
        if is_transcribing:
            with self.q.mutex:
                dropped = len(self.q.queue)
                self.q.queue.clear()
            self.detector.audio_buffer = []
            self.get_logger().info(f"📝 Transcribing → mic input disabled, flushed {dropped} chunks")

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
        if self.speaker_playing or self.transcribing:
            return  # ignore while speaker is active or transcription is running
        if status:
            print(status)
        raw = np.squeeze(indata).astype(np.int16)
        now = time.time()
        if (now - self.last_audio_debug_time) >= 1.0:
            self.last_audio_debug_time = now
            head = raw[:16].tolist()
            peak = int(np.max(np.abs(raw))) if raw.size else 0
            # self.get_logger().info(f"🎧 raw audio head={head} peak={peak} frames={frames}")
        audio = raw.astype(np.float32)
        q.put(audio)

    def destroy_node(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.pulse_proc is not None and self.pulse_proc.poll() is None:
            try:
                self.pulse_proc.terminate()
                self.pulse_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.pulse_proc.kill()
                except Exception:
                    pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SpeechNode()
        rclpy.spin(node)
    except NoUsableInputDeviceError as e:
        print(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
