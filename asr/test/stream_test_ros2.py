#!/usr/bin/env python3
import os
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from openwakeword.model import Model
from stream import WakeWordVADDetector, TARGET_SR, FRAME_LENGTH, load_vad
from faster_whisper import WhisperModel


class SpeechTestNode(Node):
    def __init__(self, audio_file, block_dur=0.1):
        super().__init__("speech_test_node")
        self.publisher_ = self.create_publisher(String, "user_speech", 10)

        # Load audio
        samples, sr = sf.read(audio_file, dtype="float32")
        if samples.ndim > 1:  # stereo → mono
            samples = np.mean(samples, axis=1)
        if sr != 48000:
            raise ValueError(f"Expected 48kHz file, got {sr}")

        # Resample to 16kHz
        samples = resample_poly(samples, TARGET_SR, sr).astype(np.float32)

        # Load models
        vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
        vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)
        whisper_model = WhisperModel("base")
        openwakeword_model = Model(wakeword_models=["alexa_v0.1"])
        test_frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
        openwakeword_model.predict(test_frame)

        # Create detector with ROS publisher
        self.detector = WakeWordVADDetector(
            wakeword_model=openwakeword_model,
            vad_model=vad_model,
            whisper_model=whisper_model,
            publisher=self.publisher_
        )

        self.samples = samples
        self.block_size = int(block_dur * TARGET_SR)
        self.num_blocks = len(samples) // self.block_size
        self.current_block = 0

        self.get_logger().info(
            f"▶️ Running test on {audio_file}, {self.num_blocks} blocks of {self.block_size} samples"
        )

        # Timer to simulate streaming
        self.timer = self.create_timer(block_dur, self.timer_callback)

    def timer_callback(self):
        if self.current_block >= self.num_blocks:
            self.get_logger().info("✅ Finished playback.")
            rclpy.shutdown()
            return

        block = self.samples[
            self.current_block * self.block_size:(self.current_block + 1) * self.block_size
        ]
        self.detector.handle_audio(block)
        self.current_block += 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str, help="Path to 48kHz WAV test file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Test file not found: {args.audio_file}")

    rclpy.init()
    node = SpeechTestNode(args.audio_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

