import pvporcupine
import sounddevice as sd
import struct

# Get your key from https://console.picovoice.ai/
ACCESS_KEY = ""


porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=["bumblebee"]
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
