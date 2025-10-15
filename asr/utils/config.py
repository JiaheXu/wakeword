import os

# =========================
# Audio / ASR / VAD Config
# =========================
TARGET_SR = 16000
FRAME_LENGTH = int(2.0 * TARGET_SR)
STEP_SIZE = int(0.15 * TARGET_SR)

# Wakeword detection
WAKEWORD_THRESHOLD = 0.06

# VAD detection
VAD_THRESHOLD = 0.5
VAD_START_LENGTH = int(1.5 * TARGET_SR)  # how much audio before starting VAD
SILENT_LENGTH = 1.0                      # seconds of silence to consider speech ended

# Buffer handling
ROLLBACK_SEC = 2

# Directory for saving audio segments
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

