# .utils/wakeword_vad_detector.py
import numpy as np
import soundfile as sf
import time, os
from collections import deque
from std_msgs.msg import String
from pino_msgs.srv import Text
from opencc import OpenCC

# Import config constants
from utils.config import (
    TARGET_SR,
    FRAME_LENGTH,
    STEP_SIZE,
    WAKEWORD_THRESHOLD,
    VAD_THRESHOLD,
    VAD_START_LENGTH,
    SILENT_LENGTH,
    ROLLBACK_SEC,
    SAVE_DIR,
)


class WakeWordVADDetector:
    def __init__(self, wakeword_model, vad_model, whisper_model, publisher, response_pub, client, node):
        self.model = wakeword_model
        self.vad_model = vad_model
        self.whisper_model = whisper_model
        self.publisher = publisher       # publishes raw transcript
        self.response_pub = response_pub # publishes LLM responses
        self.client = client             # service client to llm_service
        self.node = node

        self.noise_keywords = ['字幕', '订阅', '谢谢大家', 'CANADA']

        self.cc = OpenCC('t2s')
        self.mode = "wakeword"
        self.audio_buffer = []
        self.wakeword_buffer = deque(maxlen=ROLLBACK_SEC * TARGET_SR)
        self.detection_count = 0
        self.start_time = None

        self.last_word = time.time()
        self.last_none_word = time.time()
        self.last_detect = time.time()
        self.last_speech_end = time.time()
        self.last_cmd_time = time.time() - 10.0

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
        """繁体转简体"""
        return self.cc.convert(text)

    def transcribe(self, samples: np.ndarray):
        if self.whisper_model is None:
            print("⚠️ Whisper model not loaded, skipping transcription")
            return

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        start = time.time()

        segments, info = self.whisper_model.transcribe(
            samples.astype(np.float16), language="zh"
        )
        end = time.time()
        print("whisper took:", end - start)

        transcript_text = ""
        for seg in segments:
            line = f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}"
            print(line)
            transcript_text += self.traditional_to_simplified(seg.text.strip())

        if any(k in transcript_text for k in self.noise_keywords):
            return
        if time.time() - self.last_cmd_time < 5.0:
            return

        if transcript_text:
            # publish transcript
            msg = String()
            msg.data = transcript_text
            self.publisher.publish(msg)
            print(f"📢 Published transcript: {transcript_text}")

            # call LLM service
            req = Text.Request()
            req.text = transcript_text
            self.last_cmd_time = time.time()
            future = self.client.call_async(req)

            def _callback(fut):
                try:
                    resp = fut.result()
                    if resp.success:
                        print(f"✅ LLM Response: {resp.response}")
                        out = String()
                        out.data = resp.response
                        self.response_pub.publish(out)
                    else:
                        print(f"⚠️ LLM failure: {resp.response}")
                except Exception as e:
                    print(f"❌ Service call exception: {e}")

            future.add_done_callback(_callback)

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
                self.last_none_word = time.time()
                if self.last_none_word - self.last_word > SILENT_LENGTH:
                    utterance = self.save_segment()
                    if utterance is not None:
                        self.transcribe(utterance)
                    self.mode = "wakeword"
                    self.audio_buffer = []
            else:
                self.last_word = time.time()
