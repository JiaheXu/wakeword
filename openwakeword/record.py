import sounddevice as sd
import soundfile as sf

# Settings
SAMPLE_RATE = 16000   # match your mic (44.1 kHz)
CHANNELS = 1          # mono
DURATION = 5          # seconds
OUTPUT_FILE = "recorded.wav"

print(f"🎙️ Recording {DURATION} seconds at {SAMPLE_RATE} Hz ...")

# Record audio
audio = sd.rec(int(DURATION * SAMPLE_RATE), 
               samplerate=SAMPLE_RATE, 
               channels=CHANNELS, 
               dtype="int16")

sd.wait()  # wait until recording is finished

# Save to WAV
sf.write(OUTPUT_FILE, audio, SAMPLE_RATE, subtype="PCM_16")

print(f"✅ Saved recording to {OUTPUT_FILE}")
