#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


class NoUsableInputDeviceError(RuntimeError):
    pass


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""


def load_mac_from_autoconnect(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r'^\s*DEVICE="([0-9A-Fa-f:]{17})"', text, flags=re.MULTILINE)
    return m.group(1).lower() if m else None


def get_bt_name(mac: str | None) -> str | None:
    if not mac:
        return None
    out = run_cmd(["bluetoothctl", "info", mac])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name: "):
            return line.split("Name: ", 1)[1].strip()
    return None


def pick_input_device(mac: str | None, device_arg: str | None):
    devices = sd.query_devices()
    def is_blocked(name: str) -> bool:
        return "orin" in name.lower()

    if device_arg:
        if device_arg.isdigit():
            idx = int(device_arg)
            if (
                0 <= idx < len(devices)
                and devices[idx]["max_input_channels"] > 0
                and not is_blocked(devices[idx]["name"])
            ):
                return idx
            raise NoUsableInputDeviceError(
                f"Invalid input device index: {device_arg} (or device name contains 'Orin')"
            )
        key = device_arg.lower()
        for idx, dev in enumerate(devices):
            if (
                dev["max_input_channels"] > 0
                and key in dev["name"].lower()
                and not is_blocked(dev["name"])
            ):
                return idx
        raise NoUsableInputDeviceError(
            f"No allowed input device contains: {device_arg} (devices containing 'Orin' are blocked)"
        )

    default_source = run_cmd(["pactl", "get-default-source"]).lower()
    bt_name = (get_bt_name(mac) or "").lower()
    mac_underscore = mac.replace(":", "_").lower() if mac else ""
    mac_compact = mac.replace(":", "").lower() if mac else ""
    source_tokens = [t for t in default_source.split(".") if t]

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if mac_underscore and mac_underscore in n:
            s += 5
        if mac_compact and mac_compact in n:
            s += 5
        if bt_name and bt_name in n:
            s += 4
        if "bluez" in n:
            s += 3
        if any(t in n for t in source_tokens):
            s += 2
        if "pulse" in n:
            s += 1
        return s

    best_idx = None
    best_score = -1
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        if is_blocked(dev["name"]):
            continue
        s = score(dev["name"])
        if s > best_score:
            best_idx = idx
            best_score = s

    if best_idx is not None and best_score > 0:
        return best_idx

    default_input = sd.default.device[0]
    if (
        default_input is not None
        and default_input >= 0
        and default_input < len(devices)
        and devices[default_input]["max_input_channels"] > 0
        and not is_blocked(devices[default_input]["name"])
    ):
        return int(default_input)

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and not is_blocked(dev["name"]):
            return idx

    raise NoUsableInputDeviceError(
        "No usable input device found (all candidates may be blocked by 'Orin' filter)"
    )


def save_wav(path: Path, audio_i16: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def record_with_pulse(seconds: float, source: str, sr: int, chunk_frames: int = 320) -> np.ndarray:
    cmd = [
        "parec",
        "-d",
        source,
        "--rate",
        str(sr),
        "--channels",
        "1",
        "--format",
        "s16le",
        "--latency-msec",
        "20",
        "--raw",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise NoUsableInputDeviceError("parec not found. Install pulseaudio-utils.") from e

    raw = bytearray()
    chunk_bytes = chunk_frames * 2
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            if proc.stdout is None:
                break
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            raw.extend(buf)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)

    if not raw:
        raise NoUsableInputDeviceError(f"No audio captured from Pulse source '{source}'")
    return np.frombuffer(bytes(raw), dtype=np.int16).copy()


def main():
    parser = argparse.ArgumentParser(description="Test Bluetooth microphone capture")
    parser.add_argument("--seconds", type=float, default=5.0, help="record duration")
    parser.add_argument("--device", type=str, default=None, help="input device index or substring")
    parser.add_argument(
        "--autoconnect",
        type=Path,
        default=Path.cwd() / "auto_connect_bt.sh",
        help="path to auto_connect_bt.sh",
    )
    parser.add_argument("--mac", type=str, default=None, help="Bluetooth MAC override")
    parser.add_argument("--source", type=str, default=None, help="Pulse source override for BT mode")
    parser.add_argument("--out", type=Path, default=Path("bt_test.wav"), help="output wav path")
    args = parser.parse_args()
    use_bt = env_flag("USE_BT", default=False)

    mac = (args.mac or load_mac_from_autoconnect(args.autoconnect) or "").lower() or None
    if mac:
        print(f"[INFO] target MAC: {mac}")
    else:
        print("[WARN] no MAC found, using best available input")

    if use_bt:
        source = (args.source or run_cmd(["pactl", "get-default-source"])).strip()
        if not source:
            raise NoUsableInputDeviceError("USE_BT=true but no Pulse default source found")
        sr = 16000
        print(f"[INFO] USE_BT=true, Pulse source: {source}")
        print(f"[INFO] samplerate={sr}, seconds={args.seconds}")
        print("[INFO] recording...")
        audio = record_with_pulse(args.seconds, source, sr)
    else:
        idx = pick_input_device(mac, args.device)
        dev = sd.query_devices(idx)
        sr = int(dev.get("default_samplerate") or 16000)
        blocksize = max(256, int(0.02 * sr))
        print(f"[INFO] USE_BT=false, device {idx}: {dev['name']}")
        print(f"[INFO] samplerate={sr}, blocksize={blocksize}, seconds={args.seconds}")

        chunks: list[np.ndarray] = []

        def cb(indata, frames, time_info, status):
            if status:
                print(f"[WARN] {status}")
            chunks.append(indata[:, 0].copy())

        print("[INFO] recording...")
        start = time.time()
        with sd.InputStream(
            samplerate=sr,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            device=idx,
            callback=cb,
        ):
            while (time.time() - start) < args.seconds:
                time.sleep(0.05)

        if not chunks:
            raise RuntimeError("No audio frames captured")
        audio = np.concatenate(chunks).astype(np.int16)

    rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
    peak = int(np.max(np.abs(audio)))
    duration = len(audio) / sr
    print(f"[INFO] captured_samples={len(audio)}, duration={duration:.2f}s, rms={rms:.6f}, peak={peak}")

    save_wav(args.out, audio, sr)
    print(f"[OK] wrote {args.out}")
    if peak < 200:
        print("[WARN] very low level; check headset profile is HFP/HSP and speak near mic")


if __name__ == "__main__":
    try:
        main()
    except NoUsableInputDeviceError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)
