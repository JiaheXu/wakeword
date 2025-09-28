import soundfile as sf
import numpy as np
from whisper_trt.vad import load_vad

# ========= Config =========
INPUT_FILE = "resampled_16k.wav"   # your test audio file
SAMPLE_RATE = 16000
CHUNK_SIZE = 1536                  # Silero VAD expects 1536 samples
1600
CHUNK_SIZE = int(0.1 * SAMPLE_RATE)
SPEECH_THRESHOLD = 0.2
MIN_SILENCE_FRAMES = int(0.1 * SAMPLE_RATE)  # ~200ms

# ========= Load audio =========
samples, sr = sf.read(INPUT_FILE, dtype="float32")
if sr != SAMPLE_RATE:
    raise ValueError(f"❌ Input must be {SAMPLE_RATE}Hz, got {sr}")

# ========= Load VAD model =========
vad_model = load_vad("/home/developer/model_data/silero_vad.onnx")
# warmup
_ = vad_model(np.zeros(CHUNK_SIZE, dtype=np.float32), sr=SAMPLE_RATE)

# ========= Process chunks =========
speech_segments = []
is_speaking = False
current_segment = []
silence_counter = 0

for i in range(0, len(samples) - CHUNK_SIZE, CHUNK_SIZE):
    chunk = samples[i:i+CHUNK_SIZE]

    # Run VAD
    voice_prob = float(vad_model(chunk, sr=SAMPLE_RATE).flatten()[0])
    print(f"Chunk {i//CHUNK_SIZE}: voice_prob={voice_prob:.3f}")

    if voice_prob > SPEECH_THRESHOLD:
        if not is_speaking:
            print(f"▶ Speech started at {i/SAMPLE_RATE:.2f}s")
            is_speaking = True
            current_segment = []
        current_segment.extend(chunk.tolist())
        silence_counter = 0
    else:
        if is_speaking:
            silence_counter += 1
            if silence_counter > MIN_SILENCE_FRAMES:
                print(f"⏹ Speech ended at {i/SAMPLE_RATE:.2f}s")
                is_speaking = False
                segment = np.array(current_segment, dtype=np.float32)
                speech_segments.append(segment)
                out_file = f"speech_segment_{len(speech_segments)-1}.wav"
                sf.write(out_file, segment, SAMPLE_RATE)
                print(f"💾 Saved {out_file} ({len(segment)/SAMPLE_RATE:.2f}s)")

# ========= Save last ongoing segment =========
if is_speaking and current_segment:
    segment = np.array(current_segment, dtype=np.float32)
    speech_segments.append(segment)
    out_file = f"speech_segment_{len(speech_segments)-1}.wav"
    sf.write("./detections/" + out_file, segment, SAMPLE_RATE)
    print(f"💾 Saved {out_file} ({len(segment)/SAMPLE_RATE:.2f}s)")

