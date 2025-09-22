python3 transcribe.py base example.wav --backend whisper


python3 transcribe.py base test_input_48k.wav --backend faster_whisper

whisper speech.wav --model medium

whisper speech.wav --model medium --language English --output_format srt --output_dir ./subs

