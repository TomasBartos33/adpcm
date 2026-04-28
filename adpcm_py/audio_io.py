from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file and return mono float samples in [-1, 1]."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        fs = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return np.clip(data, -1.0, 1.0), fs


def write_wav_int16(path: str | Path, samples: np.ndarray, fs: int) -> None:
    """Write normalized float samples or int-like samples as 16-bit mono PCM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        data = np.zeros(1, dtype=np.float64)
    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        data = data / peak
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(fs))
        wav.writeframes(pcm.tobytes())
