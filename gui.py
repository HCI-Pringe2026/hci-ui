"""
lsl_test_V5.py  —  EEG viewer with clean thread separation

Thread layout:
  DataThread   — pulls LSL chunks, writes log file, feeds display deques
                 and corr ring; runs at full LSL rate, never touches GUI.
  CorrThread   — waits for corr jobs on a queue, computes Spearman,
                 writes corr file, pushes results back to GUI via a result queue.
  GUI thread   — Qt timer at 25 fps redraws plots and processes corr results.
  Sound thread — daemon, runs beep loop when condition is met.
"""

import sys
import time
import csv
import threading
import queue
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pylsl import StreamInlet, resolve_streams

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QMessageBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QScrollArea, QGroupBox, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

CHANNEL_NAMES = [
    "FP1-A1", "FP2-A2",
    "F3-A1",  "F4-A2",
    "C3-A1",  "C4-A2",
    "P3-A1",  "P4-A2",
    "O1-A1",  "O2-A2",
    "F7-A1",  "F8-A2",
    "T3-A1",  "T4-A2",
    "T5-A1",  "T6-A2",
    "ECG"
]
N_CH = len(CHANNEL_NAMES)

GUI_FPS       = 25          # частота перерисовки GUI
DISPLAY_SEC   = 5           # сколько секунд показываем на графике
LOG_FLUSH_N   = 50          # flush лог-файла раз в N строк


def ch_color(i):
    return pg.intColor(i, hues=N_CH)

# ──────────────────────────────────────────────
# TIMEZOOM VIEWBOX
# ──────────────────────────────────────────────

class TimeZoomViewBox(pg.ViewBox):
    """
    Wheel / touchpad scroll  -> zoom X, anchored to right edge
    Ctrl + Wheel             -> zoom Y for current plot only
    Double click             -> reset Y autorange
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.setMouseMode(self.PanMode)
        self.setDefaultPadding(0.0)

    def wheelEvent(self, ev, axis=None):
        mods = QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier:
            self.enableAutoRange(axis='y', enable=False)
            super().wheelEvent(ev, axis=1)
        else:
            # Zoom X anchored to the right edge
            # ViewBox receives QGraphicsSceneWheelEvent which uses delta(), not angleDelta()
            delta = ev.delta()
            if delta == 0:
                ev.accept()
                return

            zoom_factor = 1.15 if delta > 0 else 1.0 / 1.15

            x_min, x_max = self.viewRange()[0]
            current_width = x_max - x_min
            new_width = current_width / zoom_factor

            # Anchor: keep x_max (right edge, = 0) fixed
            new_x_min = x_max - new_width
            self.enableAutoRange(axis='x', enable=False)
            self.setXRange(new_x_min, 0, padding=0)
            ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        self.enableAutoRange(axis='y', enable=True)
        ev.accept()

# ──────────────────────────────────────────────
# DATA THREAD  (LSL + file IO)
# ──────────────────────────────────────────────

class DataThread(threading.Thread):
    """
    Живёт пока running=True.
    Забирает чанки из LSL inlet, пишет в лог-файл и обновляет
    display-deques (общие с GUI, защищены lock'ом).
    Если корреляция активна — кладёт сэмплы в corr_ring и при
    накоплении step_size новых сэмплов кладёт снимок в corr_job_q.
    """

    def __init__(self, inlet, fs, log_file, log_writer,
                 disp_lock, disp_buffers,
                 corr_job_q,
                 active_channels_getter,
                 shown_channels_getter):
        super().__init__(daemon=True)
        self.inlet                  = inlet
        self.fs                     = fs
        self.log_file               = log_file
        self.log_writer             = log_writer
        self.disp_lock              = disp_lock
        self.disp_buffers           = disp_buffers
        self.corr_job_q             = corr_job_q
        self.active_channels_getter = active_channels_getter   # recorded channels
        self.shown_channels_getter  = shown_channels_getter    # displayed channels

        self.running = True

        # корреляция
        self.corr_active  = False
        self.window_size  = 500
        self.step_size    = 50
        self.corr_ring    = deque()
        self.samples_step = 0

        # RGB ring
        self.rgb_active = False
        self.rgb_rings  = [deque() for _ in range(6)]
        self.rgb_ch_idx = list(range(6))

        self._flush_cnt = 0

    def run(self):
        while self.running:
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.02, max_samples=256)
            if not chunk:
                continue

            active_channels = self.active_channels_getter()   # channels to record
            shown_channels  = self.shown_channels_getter()    # channels to display
            active_indices  = [CHANNEL_NAMES.index(ch) for ch in active_channels]

            rows = []
            disp_batch   = {ch: [] for ch in shown_channels}

            for sample, ts in zip(chunk, timestamps):
                t = ts if ts else time.time()

                # лог: только записываемые каналы
                row = [f"{t:.6f}"]
                for i in active_indices:
                    if i < len(sample):
                        row.append(f"{sample[i]:.4f}")
                    else:
                        row.append("0.0000")
                rows.append(row)

                # display batch — все показываемые каналы
                for i, ch in enumerate(CHANNEL_NAMES):
                    if ch in disp_batch and i < len(sample):
                        disp_batch[ch].append(sample[i])

                # corr ring
                if self.corr_active:
                    self.corr_ring.append(sample)
                    self.samples_step += 1
                    if self.samples_step >= self.step_size:
                        self.samples_step = 0
                        if len(self.corr_ring) >= self.window_size:
                            try:
                                self.corr_job_q.put_nowait((list(self.corr_ring), t))
                            except queue.Full:
                                pass

                # RGB
                if self.rgb_active:
                    for k, ci in enumerate(self.rgb_ch_idx):
                        if ci < len(sample):
                            self.rgb_rings[k].append(sample[ci])

            for row in rows:
                self.log_writer.writerow(row)
            self._flush_cnt += len(rows)
            if self._flush_cnt >= LOG_FLUSH_N:
                self.log_file.flush()
                self._flush_cnt = 0

            with self.disp_lock:
                for ch in shown_channels:
                    if ch in self.disp_buffers:
                        self.disp_buffers[ch].extend(disp_batch[ch])

    def stop(self):
        self.running = False

    def set_corr(self, active, window_size=500, step_size=50):
        self.window_size = window_size
        self.step_size   = step_size
        self.corr_ring   = deque(maxlen=window_size)
        self.samples_step = 0
        self.corr_active = active

    def set_rgb(self, active, ch_indices, window_size=500):
        self.rgb_ch_idx = ch_indices
        self.rgb_rings  = [deque(maxlen=window_size) for _ in range(6)]
        self.rgb_active = active


# ──────────────────────────────────────────────
# CORRELATION THREAD
# ──────────────────────────────────────────────

class CorrThread(threading.Thread):
    """
    Ждёт снимки из job_q, считает Spearman, пишет файл,
    кладёт (r, t) в result_q для GUI.

    sound_callback(r) вызывается СРАЗУ после вычисления — без ожидания
    GUI-тика, поэтому задержка звука минимальна.
    """

    def __init__(self, i1, i2, job_q, result_q, corr_file, corr_writer, lock,
                 sound_callback=None):
        super().__init__(daemon=True)
        self.i1             = i1
        self.i2             = i2
        self.job_q          = job_q
        self.result_q       = result_q
        self.corr_file      = corr_file
        self.corr_writer    = corr_writer
        self.lock           = lock
        self.running        = True
        self.sound_callback = sound_callback

    def run(self):
        while self.running:
            try:
                window, t = self.job_q.get(timeout=0.1)
            except queue.Empty:
                continue

            i1, i2 = self.i1, self.i2
            try:
                x = np.array([s[i1] for s in window if i1 < len(s)], dtype=np.float64)
                y = np.array([s[i2] for s in window if i2 < len(s)], dtype=np.float64)
            except Exception:
                continue

            if len(x) < 2 or len(y) < 2:
                continue

            if np.all(x == x[0]) or np.all(y == y[0]):
                r = 0.0
            else:
                r, _ = spearmanr(x, y)
                if not np.isfinite(r):
                    r = 0.0

            with self.lock:
                if self.corr_writer:
                    self.corr_writer.writerow([f"{t:.6f}", f"{r:.6f}"])
                    self.corr_file.flush()

            try:
                self.result_q.put_nowait((r, t))
            except queue.Full:
                pass

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────
# CORRELATION WINDOW
# ──────────────────────────────────────────────

class CorrelationWindow(QWidget):
    MAXLEN = 50   # хранить и рисовать только 50 последних окон

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spearman correlation (online)")
        self.resize(600, 300)

        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setYRange(-1, 1)
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("left", "R")
        self.plot.setLabel("bottom", "Window #")
        # Отключаем автомасштаб оси X — рисуем фиксированное окно
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.curve = self.plot.plot(pen=pg.mkPen("y", width=2))
        layout.addWidget(self.plot)

        self._thr_pos = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', width=1, style=Qt.DashLine))
        self._thr_neg = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', width=1, style=Qt.DashLine))
        self.plot.addItem(self._thr_pos)
        self.plot.addItem(self._thr_neg)
        self._thr_pos.setVisible(False)
        self._thr_neg.setVisible(False)

        # Хранение — простые numpy-массивы фиксированного размера (ring-buffer)
        self._ys  = np.full(self.MAXLEN, np.nan, dtype=np.float32)
        self._ptr = 0      # указатель записи
        self._cnt = 0      # сколько точек уже добавлено
        self._idx = 0      # глобальный индекс окна (для оси X)

    def add_point(self, r):
        self._ys[self._ptr] = r
        self._ptr = (self._ptr + 1) % self.MAXLEN
        self._cnt = min(self._cnt + 1, self.MAXLEN)
        self._idx += 1

        # Упорядоченный срез от старых к новым
        if self._cnt < self.MAXLEN:
            ys = self._ys[:self._cnt]
            xs = np.arange(self._idx - self._cnt, self._idx, dtype=np.int32)
        else:
            ys = np.roll(self._ys, -self._ptr)
            xs = np.arange(self._idx - self.MAXLEN, self._idx, dtype=np.int32)

        self.curve.setData(xs, ys)

    def set_threshold_lines(self, mode, val1, val2=None):
        if mode == "abs":
            self._thr_pos.setValue(val1)
            self._thr_neg.setValue(-val1)
            self._thr_pos.setVisible(True)
            self._thr_neg.setVisible(True)
        elif mode == "range":
            self._thr_pos.setValue(val2)
            self._thr_neg.setValue(val1)
            self._thr_pos.setVisible(True)
            self._thr_neg.setVisible(True)
        else:
            self._thr_pos.setVisible(False)
            self._thr_neg.setVisible(False)

    def reset(self):
        self._ys[:] = np.nan
        self._ptr = 0
        self._cnt = 0
        self._idx = 0
        self.curve.setData([], [])


# ──────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────

class EEGViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL EEG — realtime + correlations")
        self.resize(1600, 900)

        # LSL
        self.inlet   = None
        self.streams = []
        self.fs      = 250

        # Display deques — shared with DataThread (under disp_lock)
        self.disp_lock    = threading.Lock()
        self.disp_buffers = {}   # filled in setup_plots()

        # Background threads
        self.data_thread = None
        self.corr_thread = None

        # Log
        self.log_file     = None
        self.log_writer   = None
        self.log_filename = None

        # Active Channels
        self._syncing_x = False
        self.channel_checkboxes = {}
        self.channel_show   = {ch: True for ch in CHANNEL_NAMES}
        self.channel_record = {ch: True for ch in CHANNEL_NAMES}
        self.available_channels = CHANNEL_NAMES[:]

        # Correlation
        self.corr_active  = False
        self.corr_job_q   = queue.Queue(maxsize=4)   # не больше 4 задач в очереди
        self.corr_result_q = queue.Queue()
        self.corr_file    = None
        self.corr_writer  = None
        self.corr_lock    = threading.Lock()
        self.corr_window  = None

        # RGB (вычисляется в DataThread или отдельно — здесь в GUI-потоке раз в 200 мс)
        self.rgb_active    = False
        self.rgb_file      = None
        self.rgb_writer    = None
        self._rgb_tick_cnt = 0

        self.plots  = {}
        self.curves = {}
        self.gains  = {}

        self._build_ui()
        self._setup_timers()
        self.refresh_streams()

    # ──────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────

    def _build_ui(self):
        main = QHBoxLayout(self)

        # LEFT scroll panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setSpacing(5)

        # LSL streams
        left.addWidget(QLabel("LSL streams"))
        self.stream_list = QListWidget()
        self.stream_list.setMaximumHeight(90)
        left.addWidget(self.stream_list)

        row = QHBoxLayout()
        self.btn_refresh    = QPushButton("Refresh")
        self.btn_connect    = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        for b in (self.btn_refresh, self.btn_connect, self.btn_disconnect):
            row.addWidget(b)
        left.addLayout(row)

        self.btn_refresh.clicked.connect(self.refresh_streams)
        self.btn_connect.clicked.connect(self.connect_stream)
        self.btn_disconnect.clicked.connect(self.disconnect_stream)

        # CHANNEL TABLE  (Channel | Show | Record)
        channel_box = QGroupBox("Channels")
        from PyQt5.QtWidgets import QGridLayout
        grid = QGridLayout(channel_box)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        # Header row — labels
        for col, label in enumerate(("Channel", "Show", "Rec")):
            hdr = QLabel(label)
            hdr.setAlignment(Qt.AlignCenter)
            grid.addWidget(hdr, 0, col)

        # Helper: wrap a bare QCheckBox in a centered container so the
        # indicator pixel-aligns regardless of implicit label spacing.
        def _centered_cb(cb):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setAlignment(Qt.AlignHCenter)
            h.addWidget(cb)
            return w

        # "All" toggle row
        all_lbl = QLabel("All")
        all_lbl.setAlignment(Qt.AlignCenter)
        grid.addWidget(all_lbl, 1, 0)

        self.cb_show_all = QCheckBox()
        self.cb_show_all.setChecked(True)
        self.cb_show_all.setTristate(False)
        self.cb_show_all.stateChanged.connect(self._toggle_all_show)
        grid.addWidget(_centered_cb(self.cb_show_all), 1, 1)

        self.cb_rec_all = QCheckBox()
        self.cb_rec_all.setChecked(True)
        self.cb_rec_all.setTristate(False)
        self.cb_rec_all.stateChanged.connect(self._toggle_all_record)
        grid.addWidget(_centered_cb(self.cb_rec_all), 1, 2)

        self.show_checkboxes   = {}
        self.record_checkboxes = {}

        for row_i, ch in enumerate(CHANNEL_NAMES, start=2):
            name_lbl = QLabel(ch)
            name_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            cb_show = QCheckBox()
            cb_show.setChecked(True)
            cb_show.stateChanged.connect(lambda state, c=ch: self.toggle_show(c, state))

            cb_rec = QCheckBox()
            cb_rec.setChecked(True)
            cb_rec.stateChanged.connect(lambda state, c=ch: self.toggle_record(c, state))

            grid.addWidget(name_lbl,              row_i, 0)
            grid.addWidget(_centered_cb(cb_show), row_i, 1)
            grid.addWidget(_centered_cb(cb_rec),  row_i, 2)

            self.show_checkboxes[ch]   = cb_show
            self.record_checkboxes[ch] = cb_rec

        # Keep channel_checkboxes pointing at show boxes for legacy compat
        self.channel_checkboxes = self.show_checkboxes

        left.addWidget(channel_box)

        # Spearman 2ch
        corr_box = QGroupBox("Spearman (2 channels, online)")
        cl = QVBoxLayout(corr_box)

        cl.addWidget(QLabel("Channel 1"))
        self.ch1 = QComboBox(); self.ch1.addItems(CHANNEL_NAMES)
        cl.addWidget(self.ch1)

        cl.addWidget(QLabel("Channel 2"))
        self.ch2 = QComboBox(); self.ch2.addItems(CHANNEL_NAMES)
        self.ch2.setCurrentIndex(1)
        cl.addWidget(self.ch2)

        cl.addWidget(QLabel("Window (samples)"))
        self.win = QSpinBox(); self.win.setRange(50, 5000); self.win.setValue(500)
        cl.addWidget(self.win)

        cl.addWidget(QLabel("Step (samples)"))
        self.step = QSpinBox(); self.step.setRange(1, 2000); self.step.setValue(50)
        cl.addWidget(self.step)

        # Sound condition
        sg = QGroupBox("Sound condition")
        sl = QVBoxLayout(sg)

        sl.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Threshold  |r| > T", "Range  r in [min, max]"])
        sl.addWidget(self.mode_combo)

        self.invert_check = QCheckBox("Invert condition")
        sl.addWidget(self.invert_check)

        sl.addWidget(QLabel("Threshold T"))
        self.thr = QDoubleSpinBox(); self.thr.setRange(0, 1); self.thr.setValue(0.7); self.thr.setSingleStep(0.05)
        sl.addWidget(self.thr)

        sl.addWidget(QLabel("Range min"))
        self.range_min = QDoubleSpinBox(); self.range_min.setRange(-1, 1); self.range_min.setValue(-0.3); self.range_min.setSingleStep(0.05)
        sl.addWidget(self.range_min)

        sl.addWidget(QLabel("Range max"))
        self.range_max = QDoubleSpinBox(); self.range_max.setRange(-1, 1); self.range_max.setValue(0.8); self.range_max.setSingleStep(0.05)
        sl.addWidget(self.range_max)

        cl.addWidget(sg)

        self.btn_start_corr = QPushButton("Start correlation")
        self.btn_stop_corr  = QPushButton("Stop correlation")
        cl.addWidget(self.btn_start_corr)
        cl.addWidget(self.btn_stop_corr)

        self.indicator = QLabel("●")
        self.indicator.setStyleSheet("font-size:24px; color:gray;")
        cl.addWidget(self.indicator)

        self.btn_offline = QPushButton("Offline correlations from log")
        cl.addWidget(self.btn_offline)

        left.addWidget(corr_box)

        self.btn_start_corr.clicked.connect(self.start_correlation)
        self.btn_stop_corr.clicked.connect(self.stop_correlation)
        self.btn_offline.clicked.connect(self.load_offline_file)

        # RGB
        rgb_box = QGroupBox("RGB Spearman (6 channels)")
        rl = QVBoxLayout(rgb_box)
        self.rgb_ch = []
        for i in range(6):
            cb = QComboBox(); cb.addItems(CHANNEL_NAMES)
            if i < N_CH: cb.setCurrentIndex(i)
            self.rgb_ch.append(cb)
            rl.addWidget(QLabel(f"Channel {i+1}")); rl.addWidget(cb)
        self.btn_rgb_start = QPushButton("Start RGB")
        self.btn_rgb_stop  = QPushButton("Stop RGB")
        rl.addWidget(self.btn_rgb_start); rl.addWidget(self.btn_rgb_stop)
        self.rgb_bar = QLabel()
        self.rgb_bar.setFixedHeight(25)
        self.rgb_bar.setStyleSheet("background-color:black;")
        rl.addWidget(self.rgb_bar)
        left.addWidget(rgb_box)

        self.btn_rgb_start.clicked.connect(self.start_rgb)
        self.btn_rgb_stop.clicked.connect(self.stop_rgb)

        left.addStretch()
        left_scroll.setWidget(left_panel)
        main.addWidget(left_scroll)

        # RIGHT plots
        self.plot_widget = GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, stretch=1)

    # ──────────────────────────────────────────
    # ZOOMERS
    # ──────────────────────────────────────────

    def update_zoom(self):
        gain = self.gain_spin.value()
        sec = self.time_spin.value()
        disp_size = int(self.fs * sec)
        with self.disp_lock:
            for ch in CHANNEL_NAMES:
                if ch in self.disp_buffers:
                    old = self.disp_buffers[ch]
                    new = deque(old, maxlen=disp_size)
                    self.disp_buffers[ch] = new
        for ch, g in self.gains.items():
            g.setValue(gain)

    # ──────────────────────────────────────────
    # TIMERS
    # ──────────────────────────────────────────

    def _setup_timers(self):
        # Основной таймер GUI — только рисует
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self._gui_tick)
        self.draw_timer.start(1000 // GUI_FPS)   # 40 мс

    # ──────────────────────────────────────────
    # GUI TICK  (только отрисовка + corr results)
    # ──────────────────────────────────────────

    def _gui_tick(self):
        self._redraw_plots()
        self._process_corr_results()

    def _redraw_plots(self):
        if not self.plots:
            return

        shown_channels = self.get_shown_channels()

        with self.disp_lock:
            snapshots = {
                ch: list(self.disp_buffers[ch])
                for ch in shown_channels
                if ch in self.disp_buffers
            }

        for ch in CHANNEL_NAMES:
            if ch not in self.curves:
                continue

            if ch not in shown_channels:
                self.curves[ch].setData([], [])
                continue

            data_list = snapshots.get(ch, [])
            if not data_list:
                self.curves[ch].setData([], [])
                continue

            gain = self.gains[ch].value()
            arr = np.asarray(data_list, dtype=np.float32) * gain
            n = len(arr)
            x = np.arange(-(n - 1), 1, dtype=np.float32)
            self.curves[ch].setData(x, arr)

    def _process_corr_results(self):
        """Забираем ВСЕ готовые r из CorrThread и добавляем каждую точку на график.
        Звук уже сработал в CorrThread — здесь только индикатор и график."""
        last_r = None
        while True:
            try:
                r, _ = self.corr_result_q.get_nowait()
                last_r = r
                # Добавляем каждую точку — не пропускаем
                if self.corr_window:
                    self.corr_window.add_point(r)
            except queue.Empty:
                break

        if last_r is not None:
            cond  = self._check_sound_condition(last_r)
            color = "lime" if cond else "gray"
            self.indicator.setStyleSheet(f"font-size:24px; color:{color};")

    # ──────────────────────────────────────────
    # LSL
    # ──────────────────────────────────────────

    def refresh_streams(self):
        self.stream_list.clear()
        self.streams = resolve_streams()
        for s in self.streams:
            self.stream_list.addItem(
                f"{s.name()} | {s.channel_count()} ch | {s.nominal_srate()} Hz"
            )

    def connect_stream(self):
        idx = self.stream_list.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "LSL", "Select a stream first")
            return

        info = self.streams[idx]
        self.inlet = StreamInlet(info, max_buflen=2)
        self.fs = int(info.nominal_srate())

        for ch in CHANNEL_NAMES:
            if ch in self.show_checkboxes:
                self.channel_show[ch]   = self.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.record_checkboxes[ch].isChecked()

        self._setup_plots()
        self._start_log()
        self._start_data_thread()

    def disconnect_stream(self):
        self._stop_data_thread()
        self.inlet = None
        self._stop_log()

        for ch in CHANNEL_NAMES:
            if ch in self.show_checkboxes:
                self.channel_show[ch]   = self.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.record_checkboxes[ch].isChecked()

    # ──────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────

    def _clear_plots(self):
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.gains.clear()

        with self.disp_lock:
            self.disp_buffers.clear()

    def _setup_plots(self):
        self._clear_plots()

        disp_size = max(int(self.fs * DISPLAY_SEC), 256)

        with self.disp_lock:
            for ch in self.available_channels:
                self.disp_buffers[ch] = deque(maxlen=disp_size)

        for i, ch in enumerate(self.available_channels):
            vb = TimeZoomViewBox()
            p = self.plot_widget.addPlot(row=i, col=0, viewBox=vb)

            p.setLabel("left", ch)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.hideButtons()
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=True)
            p.enableAutoRange(x=False, y=True)
            p.setXRange(-(disp_size - 1), 0, padding=0)

            if i < len(self.available_channels) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "Samples")

            c = p.plot(pen=pg.mkPen(ch_color(i), width=1))

            self.plots[ch] = p
            self.curves[ch] = c

            g = QDoubleSpinBox()
            g.setRange(0.1, 1000)
            g.setValue(1.0)
            self.gains[ch] = g

            vb.sigXRangeChanged.connect(self._sync_x_ranges)

            p.setVisible(self.channel_show.get(ch, True))

    # ──────────────────────────────────────────
    # DATA THREAD
    # ──────────────────────────────────────────

    def _start_data_thread(self):
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.stop()
            self.data_thread.join(timeout=1.0)

        while not self.corr_job_q.empty():
            try:
                self.corr_job_q.get_nowait()
            except queue.Empty:
                break

        self.data_thread = DataThread(
            inlet=self.inlet,
            fs=self.fs,
            log_file=self.log_file,
            log_writer=self.log_writer,
            disp_lock=self.disp_lock,
            disp_buffers=self.disp_buffers,
            corr_job_q=self.corr_job_q,
            active_channels_getter=self.get_active_channels,
            shown_channels_getter=self.get_shown_channels,
        )
        self.data_thread.start()

    def _stop_data_thread(self):
        if self.data_thread:
            self.data_thread.stop()
            self.data_thread.join(timeout=1.0)
            self.data_thread = None

    # ──────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────

    def _start_log(self):
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/log_{ts}.txt"
        self.log_file = open(self.log_filename, "w", newline="", buffering=65536)
        self.log_writer = csv.writer(self.log_file, delimiter=' ')

        active_channels = self.get_active_channels()
        self.log_writer.writerow(["timestamp"] + active_channels)

        print("Log started:", self.log_filename)

    def _stop_log(self):
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = self.log_writer = None
            print("Log stopped")

    # ──────────────────────────────────────────
    # CORRELATION
    # ──────────────────────────────────────────

    def start_correlation(self):
        if not self.inlet or not self.data_thread:
            QMessageBox.warning(self, "LSL", "Not connected!"); return

        win  = self.win.value()
        stp  = self.step.value()
        ch1  = self.ch1.currentText()
        ch2  = self.ch2.currentText()
        i1   = CHANNEL_NAMES.index(ch1)
        i2   = CHANNEL_NAMES.index(ch2)

        # файл корреляции
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"corr_{ch1}_{ch2}_win{win}_step{stp}_{ts}.txt"
        self.corr_file   = open(fname, "w", newline="", buffering=8192)
        self.corr_writer = csv.writer(self.corr_file, delimiter=' ')
        self.corr_writer.writerow(["timestamp", f"spearman_{ch1}_{ch2}"])

        # очищаем очереди
        while not self.corr_job_q.empty():
            try: self.corr_job_q.get_nowait()
            except queue.Empty: break
        while not self.corr_result_q.empty():
            try: self.corr_result_q.get_nowait()
            except queue.Empty: break

        # запускаем CorrThread
        if self.corr_thread and self.corr_thread.is_alive():
            self.corr_thread.stop()
            self.corr_thread.join(timeout=0.5)

        # Кэшируем параметры звука ДО старта CorrThread
        self._cache_sound_params()

        self.corr_thread = CorrThread(
            i1=i1, i2=i2,
            job_q          = self.corr_job_q,
            result_q       = self.corr_result_q,
            corr_file      = self.corr_file,
            corr_writer    = self.corr_writer,
            lock           = self.corr_lock,
        )
        self.corr_thread.start()

        # говорим DataThread начать накапливать
        self.data_thread.set_corr(True, win, stp)
        self.corr_active = True

        # окно графика
        if self.corr_window is None:
            self.corr_window = CorrelationWindow()
        self.corr_window.reset()

        mode = self.mode_combo.currentIndex()
        if mode == 0:
            self.corr_window.set_threshold_lines("abs", self.thr.value())
        else:
            self.corr_window.set_threshold_lines("range", self.range_min.value(), self.range_max.value())

        self.corr_window.show(); self.corr_window.raise_()
        self.indicator.setStyleSheet("font-size:24px; color:gray;")
        print(f"Correlation started: {ch1} vs {ch2} win={win} step={stp}")

    def stop_correlation(self):
        self.corr_active = False

        if self.data_thread:
            self.data_thread.set_corr(False)

        if self.corr_thread:
            self.corr_thread.stop()
            self.corr_thread.join(timeout=0.5)
            self.corr_thread = None

        with self.corr_lock:
            if self.corr_file:
                self.corr_file.flush()
                self.corr_file.close()
                self.corr_file = self.corr_writer = None

        if self.corr_window:
            self.corr_window.close()
            self.corr_window = None

        self.indicator.setStyleSheet("font-size:24px; color:gray;")
        print("Correlation stopped")

    # ──────────────────────────────────────────
    # CHANNEL SWITCHER
    # ──────────────────────────────────────────

    def get_shown_channels(self):
        return [ch for ch in self.available_channels if self.channel_show.get(ch, False)]

    def get_recorded_channels(self):
        return [ch for ch in self.available_channels if self.channel_record.get(ch, False)]

    # Keep legacy name used by DataThread active_channels_getter
    def get_active_channels(self):
        return self.get_recorded_channels()

    def _sync_x_ranges(self, changed_vb, x_range):
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            x0, x1 = x_range
            for ch, p in self.plots.items():
                if not p.isVisible():
                    continue
                vb = p.getViewBox()
                if vb is changed_vb:
                    continue
                # Do NOT blockSignals — that suppresses the internal axis/grid
                # repaint. The _syncing_x guard above prevents infinite recursion.
                vb.setXRange(x0, x1, padding=0)
        finally:
            self._syncing_x = False

    def _toggle_all_show(self, state):
        checked = (state == Qt.Checked)
        for ch, cb in self.show_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.channel_show[ch] = checked
            if ch in self.plots:
                self.plots[ch].setVisible(checked)
            if not checked:
                with self.disp_lock:
                    if ch in self.disp_buffers:
                        self.disp_buffers[ch].clear()

    def _toggle_all_record(self, state):
        checked = (state == Qt.Checked)
        for ch, cb in self.record_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.channel_record[ch] = checked
        if self.inlet and self.data_thread:
            self._restart_log_for_active_channels()

    def toggle_show(self, ch, state):
        self.channel_show[ch] = (state == Qt.Checked)

        if ch in self.plots:
            self.plots[ch].setVisible(self.channel_show[ch])

        with self.disp_lock:
            if ch in self.disp_buffers and not self.channel_show[ch]:
                self.disp_buffers[ch].clear()

    def toggle_record(self, ch, state):
        self.channel_record[ch] = (state == Qt.Checked)

        if self.inlet and self.data_thread:
            self._restart_log_for_active_channels()

    def _restart_log_for_active_channels(self):
        self._stop_data_thread()
        self._stop_log()
        self._start_log()
        self._start_data_thread()

    def _cache_sound_params(self):
        self._snd_mode = self.mode_combo.currentIndex()
        self._snd_invert = self.invert_check.isChecked()
        self._snd_thr = self.thr.value()
        self._snd_rmin = self.range_min.value()
        self._snd_rmax = self.range_max.value()

    def _check_sound_condition(self, r):
        if self._snd_mode == 0:
            cond = abs(r) > self._snd_thr
        else:
            cond = self._snd_rmin <= r <= self._snd_rmax

        if self._snd_invert:
            cond = not cond
        return cond

    # ──────────────────────────────────────────
    # RGB
    # ──────────────────────────────────────────

    def start_rgb(self):
        if not self.data_thread:
            QMessageBox.warning(self, "LSL", "Not connected!"); return

        w   = self.win.value()
        idx = []
        for cb in self.rgb_ch:
            ch = cb.currentText()
            idx.append(CHANNEL_NAMES.index(ch) if ch in CHANNEL_NAMES else 0)

        self.data_thread.set_rgb(True, idx, w)
        self.rgb_active = True

        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rgb_win{w}_step{self.step.value()}_{ts}.txt"
        self.rgb_file   = open(fname, "w", newline="", buffering=8192)
        self.rgb_writer = csv.writer(self.rgb_file, delimiter=' ')
        self.rgb_writer.writerow(["timestamp", "R", "G", "B"])
        self._rgb_tick_cnt = 0
        print("RGB started:", fname)

        # RGB вычисляется в отдельном таймере — реже, чтобы не нагружать GUI
        self._rgb_timer = QTimer()
        self._rgb_timer.timeout.connect(self._compute_rgb_tick)
        self._rgb_timer.start(200)   # 5 раз/сек достаточно для цвета

    def stop_rgb(self):
        self.rgb_active = False
        if hasattr(self, '_rgb_timer'):
            self._rgb_timer.stop()
        if self.data_thread:
            self.data_thread.set_rgb(False, [], 0)
        if self.rgb_file:
            self.rgb_file.flush(); self.rgb_file.close()
            self.rgb_file = self.rgb_writer = None
        self.rgb_bar.setStyleSheet("background-color:black;")
        print("RGB stopped")

    def _compute_rgb_tick(self):
        if not self.data_thread or not self.rgb_active:
            return

        w    = self.win.value()
        rings = self.data_thread.rgb_rings

        if any(len(rb) < w for rb in rings):
            return

        rgb_vals = []
        for i in range(0, 6, 2):
            x = np.array(rings[i],     dtype=np.float64)[-w:]
            y = np.array(rings[i + 1], dtype=np.float64)[-w:]
            if np.all(x == x[0]) or np.all(y == y[0]):
                rv = 0.0
            else:
                rv, _ = spearmanr(x, y)
                if not np.isfinite(rv): rv = 0.0
            rgb_vals.append(abs(rv))

        R = int(np.clip(rgb_vals[0], 0, 1) * 255)
        G = int(np.clip(rgb_vals[1], 0, 1) * 255)
        B = int(np.clip(rgb_vals[2], 0, 1) * 255)
        self.rgb_bar.setStyleSheet(f"background-color: rgb({R},{G},{B});")

        if self.rgb_writer:
            self.rgb_writer.writerow([self._rgb_tick_cnt, rgb_vals[0], rgb_vals[1], rgb_vals[2]])
        self._rgb_tick_cnt += 1

    # ──────────────────────────────────────────
    # OFFLINE CORRELATION
    # ──────────────────────────────────────────

    def load_offline_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select log file", "", "Text/CSV (*.txt *.csv)"
        )
        if not fname:
            return

        try:
            with open(fname, "r") as f:
                header = f.readline().strip().split()

            if len(header) < 2 or header[0] != "timestamp":
                raise ValueError("Invalid log header")

            names = header[1:]
            data = np.loadtxt(fname, skiprows=1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] < 2:
            QMessageBox.critical(self, "Error", "Not enough rows.")
            return

        sig = data[:, 1:]
        n_ch = sig.shape[1]

        if n_ch != len(names):
            QMessageBox.critical(self, "Error", "Header/data column mismatch.")
            return

        w = self.win.value()
        stp = self.step.value()
        out = fname.rsplit(".", 1)[0] + "_offline_corr.txt"

        with open(out, "w") as f:
            hdr = ["window"] + [
                f"{names[i]}_{names[j]}"
                for i in range(n_ch) for j in range(i + 1, n_ch)
            ]
            f.write(" ".join(hdr) + "\n")

            idx = 0
            row = 0
            while idx + w <= sig.shape[0]:
                vals = []
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        x, y = sig[idx:idx + w, i], sig[idx:idx + w, j]
                        if np.all(x == x[0]) or np.all(y == y[0]):
                            rv = 0.0
                        else:
                            rv, _ = spearmanr(x, y)
                            if not np.isfinite(rv):
                                rv = 0.0
                        vals.append(rv)

                f.write(str(row) + " " + " ".join(f"{v:.6f}" for v in vals) + "\n")
                idx += stp
                row += 1

        print("Offline corr saved:", out)
        QMessageBox.information(self, "Done", f"Saved:\n{out}")

    # ──────────────────────────────────────────
    # CLOSE
    # ──────────────────────────────────────────

    def closeEvent(self, event):
        self.stop_correlation()
        self._stop_data_thread()
        self._stop_log()
        self.stop_rgb()
        # Удаляем временный WAV файл
        try:
            import os
            os.unlink(self._tone_path)
        except Exception:
            pass
        event.accept()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    w = EEGViewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
