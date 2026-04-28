from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .audio_io import write_wav_int16


@dataclass
class AdpcmParameters:
    nbits: int = 4
    alpha: float = 0.8
    deltamin: float = 16.0
    deltamax: float = 1600.0


@dataclass
class AdpcmResult:
    x: np.ndarray
    xhat: np.ndarray
    error: np.ndarray
    codewords: np.ndarray
    codeword_signs: np.ndarray
    deltas: np.ndarray
    p_table: np.ndarray
    snr_db: float
    x_mean: float
    x_sigma: float
    error_mean: float
    error_sigma: float
    autocorr_1: float
    effective_nbits: int


def validate_parameters(params: AdpcmParameters) -> None:
    if params.nbits not in (2, 3, 4, 5, -4, -5):
        raise ValueError("nbits must be 2, 3, 4, 5, -4, or -5")
    if not -1.0 <= params.alpha <= 1.0:
        raise ValueError("alpha must be between -1 and 1")
    if not 1 <= params.deltamin <= 64:
        raise ValueError("deltamin must be between 1 and 64")
    if not 400 <= params.deltamax <= 3200:
        raise ValueError("deltamax must be between 400 and 3200")
    if params.deltamin >= params.deltamax:
        raise ValueError("deltamin must be smaller than deltamax")


def p_table_for_nbits(nbits: int) -> tuple[int, np.ndarray]:
    if nbits == 2:
        return 2, np.array([0.8, 1.6], dtype=np.float64)
    if nbits == 3:
        return 3, np.array([0.9, 0.9, 1.25, 1.75], dtype=np.float64)
    if nbits == 4:
        return 4, np.array([0.9, 0.9, 0.9, 0.9, 1.2, 1.6, 2.0, 2.4], dtype=np.float64)
    if nbits == -4:
        return 4, np.array([0.9, 0.9, 0.9, 0.9, 1.2, 1.6, 4.0, 9.6], dtype=np.float64)
    if nbits == 5:
        return 5, np.array(
            [0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3],
            dtype=np.float64,
        )
    if nbits == -5:
        return 5, np.array(
            [0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 6.6],
            dtype=np.float64,
        )
    raise ValueError("Unsupported nbits value")


def quantize_difference(
    d: float,
    delta_current: float,
    nbits: int,
    p_table: np.ndarray,
    deltamin: float,
    deltamax: float,
) -> tuple[float, int, int, float]:
    sign = 1 if d >= 0 else -1
    codeword = int(np.floor(abs(d) / delta_current))
    max_codeword = 2 ** (nbits - 1) - 1
    codeword = min(codeword, max_codeword)
    dhat = delta_current * (codeword + 0.5)
    if sign == -1:
        dhat = -dhat
    delta_next = float(delta_current * p_table[codeword])
    delta_next = min(max(delta_next, deltamin), deltamax)
    return dhat, sign, codeword, delta_next


def encode_adpcm(samples: np.ndarray, params: AdpcmParameters) -> AdpcmResult:
    validate_parameters(params)
    nbits, p_table = p_table_for_nbits(params.nbits)

    x = np.asarray(samples, dtype=np.float64).copy()
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Input signal must contain at least two mono samples")

    x_mean = float(np.mean(x))
    x_sigma = float(np.sqrt(np.mean(x * x) - x_mean**2))
    denom = np.sqrt(np.sum(x[:-1] ** 2) * np.sum(x[1:] ** 2))
    autocorr_1 = float(np.sum(x[:-1] * x[1:]) / denom) if denom else 0.0

    xhat = np.zeros_like(x)
    codewords = np.zeros(x.size, dtype=np.int32)
    codeword_signs = np.ones(x.size, dtype=np.int32)
    deltas = np.zeros(x.size, dtype=np.float64)
    delta_current = float(params.deltamin)
    deltas[0] = delta_current

    for n in range(1, x.size):
        xtilde = params.alpha * xhat[n - 1]
        d = x[n] - xtilde
        dhat, sign, codeword, delta_next = quantize_difference(
            d, delta_current, nbits, p_table, params.deltamin, params.deltamax
        )
        xhat[n] = xtilde + dhat
        codewords[n] = codeword
        codeword_signs[n] = sign
        deltas[n] = delta_next
        delta_current = delta_next

    error = xhat - x
    error_mean = float(np.mean(x - xhat))
    error_sigma = float(np.sqrt(np.mean((x - xhat) ** 2) - error_mean**2))
    snr_db = float(20.0 * np.log10(x_sigma / error_sigma)) if error_sigma > 0 else float("inf")

    return AdpcmResult(
        x=x,
        xhat=xhat,
        error=error,
        codewords=codewords,
        codeword_signs=codeword_signs,
        deltas=deltas,
        p_table=p_table,
        snr_db=snr_db,
        x_mean=x_mean,
        x_sigma=x_sigma,
        error_mean=error_mean,
        error_sigma=error_sigma,
        autocorr_1=autocorr_1,
        effective_nbits=nbits,
    )


def power_spectrum(signal: np.ndarray, fs: int, nfft: int = 1024, nwin: int = 512) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=np.float64)
    if data.size < nwin:
        data = np.pad(data, (0, nwin - data.size))
    shift = nwin // 2
    window = np.hamming(nwin)
    spectra = []
    for start in range(0, data.size - nwin + 1, shift):
        frame = data[start : start + nwin] * window
        spectra.append(np.abs(np.fft.rfft(frame, nfft)) ** 2)
    if not spectra:
        spectra.append(np.abs(np.fft.rfft(data[:nwin] * window, nfft)) ** 2)
    power = np.mean(np.vstack(spectra), axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, power


def save_outputs(result: AdpcmResult, fs: int, output_dir: str | Path, params: AdpcmParameters) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    encoded_wav = output / "sp_encoded.wav"
    error_wav = output / "adpcm_error.wav"
    encode_txt = output / "adpcm_encode.txt"
    speech_txt = output / "adpcm_speech.txt"
    encode_mat = output / "adpcm_encode.mat"

    write_wav_int16(encoded_wav, result.xhat, fs)
    write_wav_int16(error_wav, result.error, fs)

    with encode_txt.open("w", encoding="utf-8") as fh:
        fh.write("     n      x xtilde      d   dhat   xhat      c  delta\n")
        for n in range(1, min(10, result.x.size)):
            xtilde = params.alpha * result.xhat[n - 1]
            d = result.x[n] - xtilde
            dhat = result.xhat[n] - xtilde
            fh.write(
                f"{n + 1:6.1f} {result.x[n]:6.1f} {xtilde:6.1f} {d:6.1f} "
                f"{dhat:6.1f} {result.xhat[n]:6.1f} {result.codewords[n]:6.1f} {result.deltas[n]:6.1f}\n"
            )

    with speech_txt.open("w", encoding="utf-8") as fh:
        fh.write("n    x(n)   c(n)  xhat(n)\n")
        for n, (x, code, xhat) in enumerate(zip(result.x, result.codewords, result.xhat)):
            fh.write(f"{n:d} {x:6.1f}    {code:6.1f}    {xhat:6.1f}\n")

    savemat(
        encode_mat,
        {
            "cs": result.codewords,
            "csigns": result.codeword_signs,
            "alpha": params.alpha,
            "deltamin": params.deltamin,
            "deltamax": params.deltamax,
            "nbits": result.effective_nbits,
            "P": result.p_table,
            "fs": fs,
        },
    )

    return {
        "encoded_wav": encoded_wav,
        "error_wav": error_wav,
        "encode_txt": encode_txt,
        "speech_txt": speech_txt,
        "encode_mat": encode_mat,
    }
