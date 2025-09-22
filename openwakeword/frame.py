def main():
    mic = MicrophoneStream()

    while True:
        audio_chunk = mic.read()

        # --- Wake word detection ---
        if wakeword.detect(audio_chunk):
            print("Wake word detected!")

            # --- Start VAD session ---
            speech_segments = []
            while True:
                vad_chunk = mic.read()
                if vad.is_speech(vad_chunk):
                    speech_segments.append(vad_chunk)
                else:
                    if speech_segments:
                        # End of speech
                        break

            # --- Send to ASR ---
            text = asr.transcribe(b"".join(speech_segments))
            print("User said:", text)
