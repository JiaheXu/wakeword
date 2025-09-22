import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import String
import numpy as np
import time
from collections import deque
from queue import Queue
from dataclasses import dataclass
from whisper_trt.vad import load_vad

def audio_numpy_from_bytes(audio_bytes: bytes):
    return np.frombuffer(audio_bytes, dtype=np.int16)

def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, num_channels: int = 6):
    return audio_numpy[channel_index::num_channels]

def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768

@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None

@dataclass
class AudioSegment:
    chunks: list

class VADNode(Node):
    def __init__(self):
        super().__init__('VADNode')

        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.sample_rate = 48000
        self.use_channel = 0
        self.speech_threshold = 0.2
        self.chunk_size = 4096
        self.audio_queue = Queue()
        self.speech_chunks = []
        self.prev_is_voice = False
        self.max_filter_window = deque(maxlen=20)
        self.current_chunk = None

        # VAD model
        self.vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
        self.vad_model(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)

        # Publisher: publish speech segments
        self.speech_pub = self.create_publisher(String, "/speech_audio", 10)

        # Subscriber: raw audio
        self.audio_sub = self.create_subscription(
            AudioStamped,
            "/audio",
            self.audio_callback,
            qos_profile
        )

        self.get_logger().info("VADNode started")
        self.create_timer(0.05, self.run_vad)

    def audio_callback(self, msg):
        if self.current_chunk is None:
            self.current_chunk = np.array(msg.audio.audio_data.int16_data, np.int16)
        else:
            self.current_chunk = np.append(
                self.current_chunk,
                np.array(msg.audio.audio_data.int16_data, np.int16)
            )

        if self.current_chunk.shape[0] > self.chunk_size:
            audio_raw = self.current_chunk.tobytes()
            num_channels = msg.audio.info.channels
            audio_numpy = audio_numpy_from_bytes(audio_raw)
            audio_numpy = np.stack([audio_numpy_slice_channel(audio_numpy, i, num_channels) for i in range(num_channels)])
            audio_numpy_normalized = audio_numpy_normalize(audio_numpy)

            chunk = AudioChunk(
                audio_raw=audio_raw,
                audio_numpy=audio_numpy,
                audio_numpy_normalized=audio_numpy_normalized
            )
            self.audio_queue.put(chunk)
            self.current_chunk = None

    def run_vad(self):
        if self.audio_queue.empty():
            return

        audio_chunk = self.audio_queue.get()
        voice_prob = float(
            self.vad_model(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0]
        )
        
        print("voice_prob: ", voice_prob)

        audio_chunk.voice_prob = voice_prob
        self.max_filter_window.append(audio_chunk)
        is_voice = any(c.voice_prob > self.speech_threshold for c in self.max_filter_window)

        if is_voice > self.prev_is_voice:
            self.speech_chunks = [c for c in self.max_filter_window]
            self.speech_chunks.append(audio_chunk)
        elif is_voice < self.prev_is_voice:
            segment = AudioSegment(chunks=self.speech_chunks)
            self.speech_pub.publish(String(data=str([c.audio_raw.hex() for c in segment.chunks])))
            print("published voice")
        elif is_voice:
            self.speech_chunks.append(audio_chunk)

        self.prev_is_voice = is_voice

def main(args=None):
    rclpy.init(args=args)
    node = VADNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
