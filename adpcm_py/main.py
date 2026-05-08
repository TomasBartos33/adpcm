from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ["QT_LOGGING_RULES"] = "qt.multimedia.ffmpeg.debug=false;qt.multimedia.ffmpeg.info=false"

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
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
    QScrollArea,
    QSizePolicy,
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
        self.figure = Figure(figsize=(9, 7), tight_layout=True, facecolor="#ffffff")
        super().__init__(self.figure)
        self.axes = self.figure.subplots(3, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def draw_empty(self) -> None:
        for ax in self.axes:
            ax.clear()
            self._style_axis(ax)
        self.axes[0].set_title("Signal and error power spectrum")
        self.axes[1].set_title("Error histogram")
        self.axes[2].set_title("Waveforms")
        self.draw()

    def _style_axis(self, ax) -> None:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle=":", color="#d3d7dc", linewidth=0.8)
        ax.tick_params(colors="#2f343a", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#c8cdd2")
            spine.set_linewidth(0.8)
        ax.title.set_color("#1f2328")
        ax.xaxis.label.set_color("#2f343a")
        ax.yaxis.label.set_color("#2f343a")

    def draw_result(self, result: AdpcmResult, fs: int, view_start: int = 0, view_end: int | None = None) -> None:
        ax_spectrum, ax_hist, ax_wave = self.axes
        for ax in self.axes:
            ax.clear()
            self._style_axis(ax)

        freqs, signal_power = power_spectrum(result.x, fs)
        _, error_power = power_spectrum(result.error, fs)
        eps = np.finfo(float).eps
        ax_spectrum.plot(freqs, 10 * np.log10(signal_power + eps), color="#d62728", linewidth=1.4, label="signal")
        ax_spectrum.plot(
            freqs,
            10 * np.log10(error_power + eps),
            color="#1f77b4",
            linewidth=1.2,
            linestyle="--",
            label="error",
        )
        ax_spectrum.set_xlabel("Frequency [Hz]")
        ax_spectrum.set_ylabel("Log magnitude [dB]")
        ax_spectrum.legend(loc="best", frameon=False)

        ax_hist.hist(result.error, bins=101, color="#6f7f90", edgecolor="#ffffff", linewidth=0.35)
        ax_hist.set_xlabel("Error signal")
        ax_hist.set_ylabel("Count")

        end = view_end if view_end is not None else result.x.size
        start = max(0, min(view_start, result.x.size - 1))
        end = max(start + 1, min(end, result.x.size))
        t = np.arange(start, end) / fs
        ax_wave.plot(t, result.x[start:end], color="#d62728", linewidth=1.0, label="x")
        ax_wave.plot(t, result.xhat[start:end], color="#2ca02c", linewidth=1.0, linestyle="--", label="xhat")
        ax_wave.plot(t, result.error[start:end], color="#1f77b4", linewidth=0.9, linestyle=":", label="error")
        ax_wave.set_xlabel(f"Time [s], fs={fs} Hz")
        ax_wave.set_ylabel("Waveform value")
        ax_wave.legend(loc="best", frameon=False)

        self.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADPCM Speech Coder - PyQt6")
        self.resize(1320, 860)
        self.setMinimumSize(900, 620)

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
        root_layout.setContentsMargins(18, 16, 18, 14)
        root_layout.setSpacing(12)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        title = QLabel("ADPCM Speech Coder")
        title.setObjectName("title")
        subtitle = QLabel("Python/PyQt6 speech-processing GUI")
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        splitter = QSplitter()
        root_layout.addWidget(splitter, 1)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(460)
        self.canvas = PlotCanvas()
        self.canvas.draw_empty()
        splitter.addWidget(scroll)
        splitter.addWidget(self.canvas)
        splitter.setSizes([420, 900])

        self.statusBar().showMessage("Ready")
        self._apply_style()
        if DEFAULT_WAV.exists():
            self.load_file(DEFAULT_WAV)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("controlPanel")
        panel.setMinimumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(12)

        file_group = QGroupBox("Input WAV")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        load_button = QPushButton("Open WAV")
        load_button.setObjectName("secondaryButton")
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
        self.nbits.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(-1.0, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setDecimals(2)
        self.alpha.setValue(0.8)
        self.alpha.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.deltamin = QDoubleSpinBox()
        self.deltamin.setRange(1, 64)
        self.deltamin.setDecimals(1)
        self.deltamin.setValue(16)
        self.deltamin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.deltamax = QDoubleSpinBox()
        self.deltamax.setRange(400, 3200)
        self.deltamax.setDecimals(1)
        self.deltamax.setSingleStep(100)
        self.deltamax.setValue(1600)
        self.deltamax.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        params_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        params_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        params_form.setVerticalSpacing(10)
        params_form.addRow("nbits", self.nbits)
        params_form.addRow("alpha", self.alpha)
        params_form.addRow("deltamin", self.deltamin)
        params_form.addRow("deltamax", self.deltamax)
        layout.addWidget(params_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(8)
        run_button = QPushButton("Run ADPCM")
        run_button.setObjectName("primaryButton")
        run_button.clicked.connect(self.run_adpcm)
        play_original = QPushButton("Play original")
        play_original.setObjectName("secondaryButton")
        play_original.clicked.connect(lambda: self.play("original"))
        play_encoded = QPushButton("Play encoded")
        play_encoded.setObjectName("secondaryButton")
        play_encoded.clicked.connect(lambda: self.play("encoded"))
        play_error = QPushButton("Play error")
        play_error.setObjectName("secondaryButton")
        play_error.clicked.connect(lambda: self.play("error"))
        action_grid.addWidget(run_button, 0, 0, 1, 2)
        action_grid.addWidget(play_original, 1, 0)
        action_grid.addWidget(play_encoded, 1, 1)
        action_grid.addWidget(play_error, 2, 0, 1, 2)
        actions_layout.addLayout(action_grid)
        layout.addWidget(actions_group)

        view_group = QGroupBox("Waveform view")
        view_layout = QFormLayout(view_group)
        self.view_start = QSpinBox()
        self.view_start.setRange(0, 1)
        self.view_start.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.view_end = QSpinBox()
        self.view_end.setRange(1, 1)
        self.view_end.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.view_start.valueChanged.connect(self.refresh_view)
        self.view_end.valueChanged.connect(self.refresh_view)
        view_layout.setVerticalSpacing(10)
        view_layout.addRow("start sample", self.view_start)
        view_layout.addRow("end sample", self.view_end)
        layout.addWidget(view_group)

        stats_group = QGroupBox("Results")
        stats_layout = QVBoxLayout(stats_group)
        self.stats = QLabel("SNR: -")
        self.stats.setObjectName("statsLabel")
        self.stats.setWordWrap(True)
        stats_layout.addWidget(self.stats)
        layout.addWidget(stats_group)

        log_group = QGroupBox("Activity log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        log_layout.addWidget(self.log)
        layout.addWidget(log_group)

        close_line = QFrame()
        close_line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(close_line)
        close_button = QPushButton("Close")
        close_button.setObjectName("secondaryButton")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        layout.addStretch(1)
        return panel

    def _apply_style(self) -> None:
        # Write arrow SVGs so the stylesheet can reference them
        assets = ROOT / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "arrow_up.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6">'
            '<polygon points="5,0 10,6 0,6" fill="#1f2328"/></svg>'
        )
        (assets / "arrow_down.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6">'
            '<polygon points="0,0 10,0 5,6" fill="#1f2328"/></svg>'
        )
        up_url   = (assets / "arrow_up.svg").as_posix()
        down_url = (assets / "arrow_down.svg").as_posix()

        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: #ffffff;
                color: #1f2328;
                font-size: 14px;
            }}
            QLabel#title {{
                font-size: 30px;
                font-weight: 800;
                letter-spacing: 0px;
                padding: 0;
            }}
            QLabel#subtitle {{
                color: #59636e;
                padding: 0 0 4px 1px;
            }}
            QWidget#controlPanel {{
                background: #ffffff;
            }}
            QGroupBox {{
                border: 1px solid #d0d7de;
                border-radius: 6px;
                margin-top: 12px;
                padding: 14px 12px 12px 12px;
                background: #ffffff;
                font-weight: 700;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #ffffff;
            }}
            QLabel#fileLabel, QLabel#statsLabel {{
                font-family: Menlo, Monaco, Consolas, monospace;
                font-size: 12px;
                line-height: 1.3;
            }}
            QPushButton {{
                border-radius: 5px;
                padding: 9px 10px;
                min-height: 18px;
                font-weight: 700;
            }}
            QPushButton#primaryButton {{
                background: #1f2328;
                color: #ffffff;
                border: 1px solid #1f2328;
            }}
            QPushButton#primaryButton:hover {{
                background: #3b424a;
            }}
            QPushButton#secondaryButton {{
                background: #ffffff;
                color: #1f2328;
                border: 1px solid #c9d1d9;
            }}
            QPushButton#secondaryButton:hover {{
                background: #f6f8fa;
            }}
            QSpinBox, QDoubleSpinBox {{
                background: #ffffff;
                color: #1f2328;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
                padding: 6px 22px 6px 6px;
                selection-background-color: #1f2328;
                selection-color: #ffffff;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #c9d1d9;
                border-bottom: 1px solid #c9d1d9;
                border-top-right-radius: 4px;
                background: #f6f8fa;
                image: url({up_url});
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
                background: #e8ecf0;
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
                background: #d0d7de;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-left: 1px solid #c9d1d9;
                border-top: 1px solid #c9d1d9;
                border-bottom-right-radius: 4px;
                background: #f6f8fa;
                image: url({down_url});
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: #e8ecf0;
            }}
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background: #d0d7de;
            }}
            QTextEdit {{
                background: #ffffff;
                color: #1f2328;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #1f2328;
                selection-color: #ffffff;
                font-family: Menlo, Monaco, Consolas, monospace;
                font-size: 12px;
            }}
            QSplitter::handle {{
                background: #d0d7de;
                width: 1px;
            }}
            QScrollArea {{
                border: 0;
            }}
            QScrollBar:vertical {{
                background: #ffffff;
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #8c959f;
                min-height: 28px;
                border-radius: 4px;
            }}
            QStatusBar {{
                background: #ffffff;
                color: #1f2328;
                border-top: 1px solid #d0d7de;
            }}
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