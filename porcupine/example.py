import pvporcupine
import sounddevice as sd
import struct

# Get your key from https://console.picovoice.ai/
ACCESS_KEY = ""

# Built-in wake words: "porcupine", "picovoice", "bumblebee", etc.
# Or replace with a `.ppn` file path for custom wake word
keyword_paths = ["./小白.ppn"]
model_path = "./porcupine_params_zh.pv"

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=keyword_paths,
    model_path=model_path
)
# Open microphone
def audio_callback(indata, frames, time, status):
    pcm = struct.unpack_from("h" * frames, indata)
    result = porcupine.process(pcm)
    if result >= 0:
        print("Wake word detected!")

with sd.RawInputStream(samplerate=porcupine.sample_rate,
                       blocksize=porcupine.frame_length,
                       dtype='int16',
                       channels=1,
                       callback=audio_callback):
    print("Listening for wake word...")
    while True:
        pass
