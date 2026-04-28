from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .adpcm_core import AdpcmParameters, AdpcmResult, encode_adpcm, power_spectrum, save_outputs
from .audio_io import read_wav_mono


DEFAULT_WAV = ROOT / "data" / "hcdr05.wav"
OUTPUT_DIR = ROOT / "output"


class PlotCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(9, 7), tight_layout=True)
        super().__init__(self.figure)
        self.axes = self.figure.subplots(3, 1)

    def draw_empty(self) -> None:
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle="--", alpha=0.35)
        self.axes[0].set_title("Signal and error power spectrum")
        self.axes[1].set_title("Error histogram")
        self.axes[2].set_title("Waveforms")
        self.draw()

    def draw_result(self, result: AdpcmResult, fs: int, view_start: int = 0, view_end: int | None = None) -> None:
        ax_spectrum, ax_hist, ax_wave = self.axes
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle="--", alpha=0.35)

        freqs, signal_power = power_spectrum(result.x, fs)
        _, error_power = power_spectrum(result.error, fs)
        eps = np.finfo(float).eps
        ax_spectrum.plot(freqs, 10 * np.log10(signal_power + eps), color="#c62828", linewidth=1.4, label="signal")
        ax_spectrum.plot(freqs, 10 * np.log10(error_power + eps), color="#1565c0", linewidth=1.4, label="error")
        ax_spectrum.set_xlabel("Frequency [Hz]")
        ax_spectrum.set_ylabel("Log magnitude [dB]")
        ax_spectrum.legend(loc="best")

        ax_hist.hist(result.error, bins=101, color="#455a64", edgecolor="white", linewidth=0.35)
        ax_hist.set_xlabel("Error signal")
        ax_hist.set_ylabel("Count")

        end = view_end if view_end is not None else result.x.size
        start = max(0, min(view_start, result.x.size - 1))
        end = max(start + 1, min(end, result.x.size))
        t = np.arange(start, end) / fs
        ax_wave.plot(t, result.x[start:end], color="#c62828", linewidth=1.0, label="x")
        ax_wave.plot(t, result.xhat[start:end], color="#2e7d32", linewidth=1.0, label="xhat")
        ax_wave.plot(t, result.error[start:end], color="#1565c0", linewidth=0.9, label="error")
        ax_wave.set_xlabel(f"Time [s], fs={fs} Hz")
        ax_wave.set_ylabel("Waveform value")
        ax_wave.legend(loc="best")

        self.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADPCM Speech Coder - PyQt6")
        self.resize(1280, 820)

        self.samples_float: np.ndarray | None = None
        self.samples_adpcm: np.ndarray | None = None
        self.fs = 8000
        self.current_file: Path | None = None
        self.result: AdpcmResult | None = None
        self.saved_paths: dict[str, Path] = {}
        self.players = {
            "original": QSoundEffect(self),
            "encoded": QSoundEffect(self),
            "error": QSoundEffect(self),
        }

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        title = QLabel("ADPCM Speech Coder")
        title.setObjectName("title")
        subtitle = QLabel("Python/PyQt6 port of the Matlab speech-processing GUI")
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        splitter = QSplitter()
        root_layout.addWidget(splitter, 1)

        controls = self._build_controls()
        self.canvas = PlotCanvas()
        self.canvas.draw_empty()
        splitter.addWidget(controls)
        splitter.addWidget(self.canvas)
        splitter.setSizes([390, 890])

        self.statusBar().showMessage("Ready")
        self._apply_style()
        if DEFAULT_WAV.exists():
            self.load_file(DEFAULT_WAV)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)
        layout = QVBoxLayout(panel)

        file_group = QGroupBox("Input WAV")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        load_button = QPushButton("Open WAV")
        load_button.clicked.connect(self.open_wav)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(load_button)
        layout.addWidget(file_group)

        params_group = QGroupBox("ADPCM parameters")
        params_form = QFormLayout(params_group)
        self.nbits = QSpinBox()
        self.nbits.setRange(-5, 5)
        self.nbits.setValue(4)
        self.nbits.setToolTip("Allowed values: 2, 3, 4, 5, -4, -5")
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(-1.0, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setDecimals(2)
        self.alpha.setValue(0.8)
        self.deltamin = QDoubleSpinBox()
        self.deltamin.setRange(1, 64)
        self.deltamin.setDecimals(1)
        self.deltamin.setValue(16)
        self.deltamax = QDoubleSpinBox()
        self.deltamax.setRange(400, 3200)
        self.deltamax.setDecimals(1)
        self.deltamax.setSingleStep(100)
        self.deltamax.setValue(1600)
        params_form.addRow("nbits", self.nbits)
        params_form.addRow("alpha", self.alpha)
        params_form.addRow("deltamin", self.deltamin)
        params_form.addRow("deltamax", self.deltamax)
        layout.addWidget(params_group)

        action_grid = QGridLayout()
        run_button = QPushButton("Run ADPCM")
        run_button.clicked.connect(self.run_adpcm)
        play_original = QPushButton("Play original")
        play_original.clicked.connect(lambda: self.play("original"))
        play_encoded = QPushButton("Play encoded")
        play_encoded.clicked.connect(lambda: self.play("encoded"))
        play_error = QPushButton("Play error")
        play_error.clicked.connect(lambda: self.play("error"))
        action_grid.addWidget(run_button, 0, 0, 1, 2)
        action_grid.addWidget(play_original, 1, 0)
        action_grid.addWidget(play_encoded, 1, 1)
        action_grid.addWidget(play_error, 2, 0, 1, 2)
        layout.addLayout(action_grid)

        view_group = QGroupBox("Waveform view")
        view_layout = QFormLayout(view_group)
        self.view_start = QSpinBox()
        self.view_start.setRange(0, 1)
        self.view_end = QSpinBox()
        self.view_end.setRange(1, 1)
        self.view_start.valueChanged.connect(self.refresh_view)
        self.view_end.valueChanged.connect(self.refresh_view)
        view_layout.addRow("start sample", self.view_start)
        view_layout.addRow("end sample", self.view_end)
        layout.addWidget(view_group)

        stats_group = QGroupBox("Results")
        stats_layout = QVBoxLayout(stats_group)
        self.stats = QLabel("SNR: -")
        self.stats.setWordWrap(True)
        stats_layout.addWidget(self.stats)
        layout.addWidget(stats_group)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.log, 1)

        close_line = QFrame()
        close_line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(close_line)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f6f7f9; color: #1f2933; font-size: 14px; }
            QLabel#title { font-size: 28px; font-weight: 700; padding: 8px 4px 0 4px; }
            QLabel#subtitle { color: #52606d; padding: 0 4px 8px 4px; }
            QGroupBox { border: 1px solid #ccd3db; border-radius: 6px; margin-top: 10px; padding: 10px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: #1f6feb; color: white; border: 0; border-radius: 5px; padding: 8px 10px; }
            QPushButton:hover { background: #195fc8; }
            QSpinBox, QDoubleSpinBox, QTextEdit { background: white; border: 1px solid #c7d0d9; border-radius: 4px; padding: 4px; }
            """
        )

    def open_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open WAV file", str(ROOT), "WAV files (*.wav)")
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> None:
        try:
            samples, fs = read_wav_mono(path)
        except Exception as exc:
            QMessageBox.critical(self, "WAV load error", str(exc))
            return
        self.current_file = path
        self.samples_float = samples
        self.samples_adpcm = samples * 32768.0
        self.fs = fs
        self.result = None
        self.saved_paths = {}
        self.file_label.setText(f"{path.name}\n{samples.size} samples, fs={fs} Hz")
        self.view_start.blockSignals(True)
        self.view_end.blockSignals(True)
        self.view_start.setRange(0, max(0, samples.size - 1))
        self.view_end.setRange(1, samples.size)
        self.view_start.setValue(0)
        self.view_end.setValue(samples.size)
        self.view_start.blockSignals(False)
        self.view_end.blockSignals(False)
        self.players["original"].setSource(QUrl.fromLocalFile(str(path)))
        self.stats.setText("SNR: -")
        self.canvas.draw_empty()
        self._log(f"Loaded {path} ({samples.size} samples, fs={fs} Hz)")

    def current_parameters(self) -> AdpcmParameters:
        return AdpcmParameters(
            nbits=int(self.nbits.value()),
            alpha=float(self.alpha.value()),
            deltamin=float(self.deltamin.value()),
            deltamax=float(self.deltamax.value()),
        )

    def run_adpcm(self) -> None:
        if self.samples_adpcm is None:
            QMessageBox.warning(self, "Missing input", "Load a WAV file first.")
            return
        params = self.current_parameters()
        try:
            self.result = encode_adpcm(self.samples_adpcm, params)
            self.saved_paths = save_outputs(self.result, self.fs, OUTPUT_DIR, params)
        except Exception as exc:
            QMessageBox.critical(self, "ADPCM error", str(exc))
            return

        self.players["encoded"].setSource(QUrl.fromLocalFile(str(self.saved_paths["encoded_wav"])))
        self.players["error"].setSource(QUrl.fromLocalFile(str(self.saved_paths["error_wav"])))
        self.refresh_view()
        self.stats.setText(
            f"SNR: {self.result.snr_db:.2f} dB\n"
            f"xbar: {self.result.x_mean:.2f}, sigmax: {self.result.x_sigma:.2f}\n"
            f"ebar: {self.result.error_mean:.2f}, sigmae: {self.result.error_sigma:.2f}\n"
            f"ac: {self.result.autocorr_1:.2f}"
        )
        self._log(
            "ADPCM finished: "
            f"nbits={self.result.effective_nbits}, alpha={params.alpha:.2f}, "
            f"deltamin={params.deltamin:.1f}, deltamax={params.deltamax:.1f}, "
            f"SNR={self.result.snr_db:.2f} dB"
        )
        self._log(f"Saved outputs to {OUTPUT_DIR}")
        self.statusBar().showMessage(f"ADPCM finished, SNR {self.result.snr_db:.2f} dB")

    def refresh_view(self) -> None:
        if self.result is None:
            return
        start = int(self.view_start.value())
        end = int(self.view_end.value())
        if end <= start:
            end = start + 1
            self.view_end.blockSignals(True)
            self.view_end.setValue(end)
            self.view_end.blockSignals(False)
        self.canvas.draw_result(self.result, self.fs, start, end)

    def play(self, key: str) -> None:
        player = self.players[key]
        if player.source().isEmpty():
            QMessageBox.information(self, "Nothing to play", "Run ADPCM or load a WAV file first.")
            return
        player.stop()
        player.play()

    def _log(self, message: str) -> None:
        self.log.append(message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
