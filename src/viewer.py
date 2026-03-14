import csv
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import override

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_streams
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsProxyWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import GraphicsLayoutWidget
from scipy.stats import spearmanr

from .config import CHANNEL_NAMES, DISPLAY_SEC, GUI_FPS, N_CH, ch_color
from .corr_window import CorrelationWindow
from .threads import CorrThread, DataThread
from .viewbox import TimeZoomViewBox


class EEGViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL EEG — realtime + correlations")
        self.resize(1600, 900)

        self.inlet = None
        self.streams = []
        self.fs = 250

        self.disp_lock = threading.Lock()
        self.disp_buffers = {}

        self.data_thread = None
        self.corr_thread = None

        self.log_file = None
        self.log_writer = None
        self.log_filename = None

        self._syncing_x = False
        self.channel_checkboxes = {}
        self.channel_show = dict.fromkeys(CHANNEL_NAMES, True)
        self.channel_record = dict.fromkeys(CHANNEL_NAMES, True)
        self.available_channels = CHANNEL_NAMES[:]

        self.corr_active = False
        self.corr_job_q = queue.Queue(maxsize=4)
        self.corr_result_q = queue.Queue()
        self.corr_file = None
        self.corr_writer = None
        self.corr_lock = threading.Lock()
        self.corr_window = None

        self.rgb_active = False
        self.rgb_file = None
        self.rgb_writer = None
        self._rgb_tick_cnt = 0

        self.plots = {}
        self.curves = {}
        self.gains = {}
        self.y_scale_spinboxes = {}

        self._build_ui()
        self._setup_timers()
        self.refresh_streams()

    # UI

    def _build_ui(self):
        main = QHBoxLayout(self)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setSpacing(5)

        left.addWidget(QLabel("LSL streams"))
        self.stream_list = QListWidget()
        self.stream_list.setMaximumHeight(90)
        left.addWidget(self.stream_list)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        for b in (self.btn_refresh, self.btn_connect, self.btn_disconnect):
            row.addWidget(b)
        left.addLayout(row)

        self.btn_refresh.clicked.connect(self.refresh_streams)
        self.btn_connect.clicked.connect(self.connect_stream)
        self.btn_disconnect.clicked.connect(self.disconnect_stream)

        channel_box = QGroupBox("Channels")
        grid = QGridLayout(channel_box)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        for col, label in enumerate(("Channel", "Show", "Rec")):
            hdr = QLabel(label)
            hdr.setAlignment(Qt.AlignCenter)
            grid.addWidget(hdr, 0, col)

        def _centered_cb(cb):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setAlignment(Qt.AlignHCenter)
            h.addWidget(cb)
            return w

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

        self.show_checkboxes = {}
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

            grid.addWidget(name_lbl, row_i, 0)
            grid.addWidget(_centered_cb(cb_show), row_i, 1)
            grid.addWidget(_centered_cb(cb_rec), row_i, 2)

            self.show_checkboxes[ch] = cb_show
            self.record_checkboxes[ch] = cb_rec

        self.channel_checkboxes = self.show_checkboxes
        left.addWidget(channel_box)

        corr_box = QGroupBox("Spearman (2 channels, online)")
        cl = QVBoxLayout(corr_box)

        cl.addWidget(QLabel("Channel 1"))
        self.ch1 = QComboBox()
        self.ch1.addItems(CHANNEL_NAMES)
        cl.addWidget(self.ch1)

        cl.addWidget(QLabel("Channel 2"))
        self.ch2 = QComboBox()
        self.ch2.addItems(CHANNEL_NAMES)
        self.ch2.setCurrentIndex(1)
        cl.addWidget(self.ch2)

        cl.addWidget(QLabel("Window (samples)"))
        self.win = QSpinBox()
        self.win.setRange(50, 5000)
        self.win.setValue(500)
        cl.addWidget(self.win)

        cl.addWidget(QLabel("Step (samples)"))
        self.step = QSpinBox()
        self.step.setRange(1, 2000)
        self.step.setValue(50)
        cl.addWidget(self.step)

        sg = QGroupBox("Sound condition")
        sl = QVBoxLayout(sg)

        sl.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Threshold  |r| > T", "Range  r in [min, max]"])
        sl.addWidget(self.mode_combo)

        self.invert_check = QCheckBox("Invert condition")
        sl.addWidget(self.invert_check)

        sl.addWidget(QLabel("Threshold T"))
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0, 1)
        self.thr.setValue(0.7)
        self.thr.setSingleStep(0.05)
        sl.addWidget(self.thr)

        sl.addWidget(QLabel("Range min"))
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(-1, 1)
        self.range_min.setValue(-0.3)
        self.range_min.setSingleStep(0.05)
        sl.addWidget(self.range_min)

        sl.addWidget(QLabel("Range max"))
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(-1, 1)
        self.range_max.setValue(0.8)
        self.range_max.setSingleStep(0.05)
        sl.addWidget(self.range_max)

        cl.addWidget(sg)

        self.btn_start_corr = QPushButton("Start correlation")
        self.btn_stop_corr = QPushButton("Stop correlation")
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

        rgb_box = QGroupBox("RGB Spearman (6 channels)")
        rl = QVBoxLayout(rgb_box)
        self.rgb_ch = []
        for i in range(6):
            cb = QComboBox()
            cb.addItems(CHANNEL_NAMES)
            if i < N_CH:
                cb.setCurrentIndex(i)
            self.rgb_ch.append(cb)
            rl.addWidget(QLabel(f"Channel {i+1}"))
            rl.addWidget(cb)
        self.btn_rgb_start = QPushButton("Start RGB")
        self.btn_rgb_stop = QPushButton("Stop RGB")
        rl.addWidget(self.btn_rgb_start)
        rl.addWidget(self.btn_rgb_stop)
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

        self.plot_widget = GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, stretch=1)

    def update_zoom(self):
        gain = self.gain_spin.value()
        sec = self.time_spin.value()
        disp_size = int(self.fs * sec)
        with self.disp_lock:
            for ch in CHANNEL_NAMES:
                if ch in self.disp_buffers:
                    self.disp_buffers[ch] = deque(self.disp_buffers[ch], maxlen=disp_size)
        for _ch, g in self.gains.items():
            g.setValue(gain)

    # Timers

    def _setup_timers(self):
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self._gui_tick)
        self.draw_timer.start(1000 // GUI_FPS)

    # GUI tick

    def _gui_tick(self):
        self._redraw_plots()
        self._process_corr_results()

    def _redraw_plots(self):
        if not self.plots:
            return

        shown_channels = self.get_shown_channels()

        with self.disp_lock:
            snapshots = {ch: list(self.disp_buffers[ch]) for ch in shown_channels if ch in self.disp_buffers}

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
        last_r = None
        while True:
            try:
                r, _ = self.corr_result_q.get_nowait()
                last_r = r
                if self.corr_window:
                    self.corr_window.add_point(r)
            except queue.Empty:
                break

        if last_r is not None:
            cond = self._check_sound_condition(last_r)
            color = "lime" if cond else "gray"
            self.indicator.setStyleSheet(f"font-size:24px; color:{color};")

    # LSL

    def refresh_streams(self):
        self.stream_list.clear()
        self.streams = resolve_streams()
        for s in self.streams:
            self.stream_list.addItem(f"{s.name()} | {s.channel_count()} ch | {s.nominal_srate()} Hz")

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
                self.channel_show[ch] = self.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.record_checkboxes[ch].isChecked()

        self._setup_plots()
        self._start_log()
        self._start_data_thread()

        self.cb_rec_all.setEnabled(False)
        for cb in self.record_checkboxes.values():
            cb.setEnabled(False)

    def disconnect_stream(self):
        self._stop_data_thread()
        self.inlet = None
        self._stop_log()

        for ch in CHANNEL_NAMES:
            if ch in self.show_checkboxes:
                self.channel_show[ch] = self.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.record_checkboxes[ch].isChecked()

        self.cb_rec_all.setEnabled(True)
        for cb in self.record_checkboxes.values():
            cb.setEnabled(True)

    # Plots

    def _clear_plots(self):
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.gains.clear()
        self.y_scale_spinboxes.clear()

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

            ys = QDoubleSpinBox()
            ys.setRange(0, 1_000_000)
            ys.setValue(0)
            ys.setDecimals(1)
            ys.setSingleStep(10)
            ys.setToolTip(f"{ch}: max amplitude (0 = autorange)")
            ys.setFixedWidth(80)
            ys.setFixedHeight(22)
            ys.setStyleSheet(
                "QDoubleSpinBox { font-size: 9px; padding: 0px 2px; "
                "background: rgba(30,30,30,200); color: #ddd; border: 1px solid #555; }"
            )

            proxy = QGraphicsProxyWidget(p)
            proxy.setWidget(ys)
            proxy.setZValue(10)

            def _place(px=proxy, plot=p):
                r = plot.boundingRect()
                px.setPos(r.right() - px.widget().width() - 4, r.top() + 2)

            _place()
            p.geometryChanged.connect(_place)

            def _on_yscale_changed(val, plot=p):
                if val == 0:
                    plot.enableAutoRange(axis="y", enable=True)
                else:
                    plot.enableAutoRange(axis="y", enable=False)
                    plot.setYRange(-val, val, padding=0)

            ys.valueChanged.connect(_on_yscale_changed)
            self.y_scale_spinboxes[ch] = ys
            vb._yscale_spinbox = ys

            vb.sigXRangeChanged.connect(self._sync_x_ranges)

            p.setVisible(self.channel_show.get(ch, True))

        self._layout_guard = pg.LabelItem("")
        self.plot_widget.addItem(self._layout_guard, row=len(self.available_channels), col=0)

    # Data thread

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

    # Logging

    def _start_log(self):
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/log_{ts}.txt"
        self.log_file = open(self.log_filename, "w", newline="", buffering=65536)  # noqa: SIM115
        self.log_writer = csv.writer(self.log_file, delimiter=" ")
        self.log_writer.writerow(["timestamp", *self.get_active_channels()])
        print("Log started:", self.log_filename)

    def _stop_log(self):
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = self.log_writer = None
            print("Log stopped")

    # Correlation

    def start_correlation(self):
        if not self.inlet or not self.data_thread:
            QMessageBox.warning(self, "LSL", "Not connected!")
            return

        win = self.win.value()
        stp = self.step.value()
        ch1 = self.ch1.currentText()
        ch2 = self.ch2.currentText()
        i1 = CHANNEL_NAMES.index(ch1)
        i2 = CHANNEL_NAMES.index(ch2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"corr_{ch1}_{ch2}_win{win}_step{stp}_{ts}.txt"
        self.corr_file = open(fname, "w", newline="", buffering=8192)  # noqa: SIM115
        self.corr_writer = csv.writer(self.corr_file, delimiter=" ")
        self.corr_writer.writerow(["timestamp", f"spearman_{ch1}_{ch2}"])

        for q in (self.corr_job_q, self.corr_result_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self.corr_thread and self.corr_thread.is_alive():
            self.corr_thread.stop()
            self.corr_thread.join(timeout=0.5)

        self._cache_sound_params()

        self.corr_thread = CorrThread(
            i1=i1,
            i2=i2,
            job_q=self.corr_job_q,
            result_q=self.corr_result_q,
            corr_file=self.corr_file,
            corr_writer=self.corr_writer,
            lock=self.corr_lock,
        )
        self.corr_thread.start()

        self.data_thread.set_corr(True, win, stp)
        self.corr_active = True

        if self.corr_window is None:
            self.corr_window = CorrelationWindow()
        self.corr_window.reset()

        mode = self.mode_combo.currentIndex()
        if mode == 0:
            self.corr_window.set_threshold_lines("abs", self.thr.value())
        else:
            self.corr_window.set_threshold_lines("range", self.range_min.value(), self.range_max.value())

        self.corr_window.show()
        self.corr_window.raise_()
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

    # Channel visibility / recording

    def get_shown_channels(self):
        return [ch for ch in self.available_channels if self.channel_show.get(ch, False)]

    def get_recorded_channels(self):
        return [ch for ch in self.available_channels if self.channel_record.get(ch, False)]

    def get_active_channels(self):
        return self.get_recorded_channels()

    def _sync_x_ranges(self, changed_vb, x_range):
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            x0, x1 = x_range
            for _ch, p in self.plots.items():
                if not p.isVisible():
                    continue
                vb = p.getViewBox()
                if vb is changed_vb:
                    continue
                vb.setXRange(x0, x1, padding=0)
        finally:
            self._syncing_x = False

    def _toggle_all_show(self, state):
        checked = state == Qt.Checked
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
        checked = state == Qt.Checked
        for ch, cb in self.record_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.channel_record[ch] = checked
        if self.inlet and self.data_thread:
            self._restart_log_for_active_channels()

    def toggle_show(self, ch, state):
        self.channel_show[ch] = state == Qt.Checked
        if ch in self.plots:
            self.plots[ch].setVisible(self.channel_show[ch])
        with self.disp_lock:
            if ch in self.disp_buffers and not self.channel_show[ch]:
                self.disp_buffers[ch].clear()

    def toggle_record(self, ch, state):
        self.channel_record[ch] = state == Qt.Checked
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
        cond = abs(r) > self._snd_thr if self._snd_mode == 0 else self._snd_rmin <= r <= self._snd_rmax
        return not cond if self._snd_invert else cond

    # RGB

    def start_rgb(self):
        if not self.data_thread:
            QMessageBox.warning(self, "LSL", "Not connected!")
            return

        w = self.win.value()
        idx = [CHANNEL_NAMES.index(cb.currentText()) if cb.currentText() in CHANNEL_NAMES else 0 for cb in self.rgb_ch]

        self.data_thread.set_rgb(True, idx, w)
        self.rgb_active = True

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rgb_win{w}_step{self.step.value()}_{ts}.txt"
        self.rgb_file = open(fname, "w", newline="", buffering=8192)  # noqa: SIM115
        self.rgb_writer = csv.writer(self.rgb_file, delimiter=" ")
        self.rgb_writer.writerow(["timestamp", "R", "G", "B"])
        self._rgb_tick_cnt = 0
        print("RGB started:", fname)

        self._rgb_timer = QTimer()
        self._rgb_timer.timeout.connect(self._compute_rgb_tick)
        self._rgb_timer.start(200)

    def stop_rgb(self):
        self.rgb_active = False
        if hasattr(self, "_rgb_timer"):
            self._rgb_timer.stop()
        if self.data_thread:
            self.data_thread.set_rgb(False, [], 0)
        if self.rgb_file:
            self.rgb_file.flush()
            self.rgb_file.close()
            self.rgb_file = self.rgb_writer = None
        self.rgb_bar.setStyleSheet("background-color:black;")
        print("RGB stopped")

    def _compute_rgb_tick(self):
        if not self.data_thread or not self.rgb_active:
            return

        w = self.win.value()
        rings = self.data_thread.rgb_rings

        if any(len(rb) < w for rb in rings):
            return

        rgb_vals = []
        for i in range(0, 6, 2):
            x = np.array(rings[i], dtype=np.float64)[-w:]
            y = np.array(rings[i + 1], dtype=np.float64)[-w:]
            if np.all(x == x[0]) or np.all(y == y[0]):
                rv = 0.0
            else:
                rv, _ = spearmanr(x, y)
                if not np.isfinite(rv):
                    rv = 0.0
            rgb_vals.append(abs(rv))

        r = int(np.clip(rgb_vals[0], 0, 1) * 255)
        g = int(np.clip(rgb_vals[1], 0, 1) * 255)
        b = int(np.clip(rgb_vals[2], 0, 1) * 255)
        self.rgb_bar.setStyleSheet(f"background-color: rgb({r},{g},{b});")

        if self.rgb_writer:
            self.rgb_writer.writerow([self._rgb_tick_cnt, rgb_vals[0], rgb_vals[1], rgb_vals[2]])
        self._rgb_tick_cnt += 1

    # Offline correlation

    def load_offline_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select log file", "", "Text/CSV (*.txt *.csv)")
        if not fname:
            return

        try:
            with open(fname) as f:
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
            hdr = ["window"] + [f"{names[i]}_{names[j]}" for i in range(n_ch) for j in range(i + 1, n_ch)]
            f.write(" ".join(hdr) + "\n")

            idx = 0
            row = 0
            while idx + w <= sig.shape[0]:
                vals = []
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        x, y = sig[idx : idx + w, i], sig[idx : idx + w, j]
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

    # Close

    @override
    def closeEvent(self, event):
        self.stop_correlation()
        self._stop_data_thread()
        self._stop_log()
        self.stop_rgb()
        try:
            import os

            os.unlink(self._tone_path)
        except Exception:
            pass
        event.accept()
