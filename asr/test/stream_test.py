import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import time
import os
from openwakeword.model import Model

from stream import WakeWordVADDetector, TARGET_SR, FRAME_LENGTH, load_vad  # import from your main script

from faster_whisper import WhisperModel

def run_test(audio_file, block_dur=0.1):
    # Load 48kHz test audio
    samples, sr = sf.read(audio_file, dtype="float32")
    if samples.ndim > 1:  # stereo → mono
        samples = np.mean(samples, axis=1)
    assert sr == 48000, f"Expected 48kHz file, got {sr}"

    # Resample to 16kHz for processing
    samples = resample_poly(samples, TARGET_SR, sr).astype(np.float32)
    # print("float audio: ", samples.nbytes)
    # samples = (samples * 32767).astype(np.int16)
    # print("int16 audio: ", samples.nbytes)

    # Load VAD model
    vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
    vad_model(np.zeros(1536, dtype=np.float32), sr=TARGET_SR)  # warmup

    # Load Whisper model (Chinese capable)
    whisper_model = WhisperModel("base")
    openwakeword_model = Model(wakeword_models=['alexa_v0.1'])

    test_frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
    openwakeword_model.predict(test_frame)

    detector = WakeWordVADDetector( wakeword_model = openwakeword_model, vad_model=vad_model, whisper_model=whisper_model)


    block_size = int(block_dur * TARGET_SR)
    num_blocks = len(samples) // block_size
    print("samples: ", len(samples))
    print(f"▶️ Running test on {audio_file}, {num_blocks} blocks of {block_size} samples")
    # detector.mode = "vad"
    for i in range(num_blocks):
        block = samples[i * block_size:(i + 1) * block_size]
        detector.handle_audio(block)
        time.sleep(0.1)  # simulate real-time pacing

    # flush any final segment
    if detector.mode == "vad":
        print("⚠️ File ended during speech, finalizing...")
        detector.save_segment()


if __name__ == "__main__":
    test_file = "test_input_48k.wav"  # replace with your own 48kHz file
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    run_test(test_file)

