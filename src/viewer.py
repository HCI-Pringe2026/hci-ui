import csv
import queue
import threading
from datetime import datetime
from typing import override

from pylsl import StreamInlet, resolve_streams
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QMessageBox, QWidget

from .config import CHANNEL_NAMES, GUI_FPS
from .correlation import CorrelationController
from .panels import LeftPanel
from .plot_area import PlotArea
from .rgb import RGBController
from .threads import DataThread


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

        self.data_thread: DataThread | None = None

        self.log_file = None
        self.log_writer = None
        self.log_filename = None

        self.channel_show = dict.fromkeys(CHANNEL_NAMES, True)
        self.channel_record = dict.fromkeys(CHANNEL_NAMES, True)
        self.available_channels = CHANNEL_NAMES[:]

        self.panel = LeftPanel(self)
        self.plot_area = PlotArea(self.disp_lock, self.disp_buffers)
        self.correlation = CorrelationController(self)
        self.rgb = RGBController(self, self.panel.rgb_bar)

        layout = QHBoxLayout(self)
        layout.addWidget(self.panel)
        layout.addWidget(self.plot_area.widget, stretch=1)

        self._wire_signals()
        self._setup_timers()
        self.refresh_streams()

    def _wire_signals(self):
        p = self.panel

        p.btn_refresh.clicked.connect(self.refresh_streams)
        p.btn_connect.clicked.connect(self.connect_stream)
        p.btn_disconnect.clicked.connect(self.disconnect_stream)

        p.cb_show_all.stateChanged.connect(self._toggle_all_show)
        p.cb_rec_all.stateChanged.connect(self._toggle_all_record)

        for ch, cb in p.show_checkboxes.items():
            cb.stateChanged.connect(lambda state, c=ch: self.toggle_show(c, state))
        for ch, cb in p.record_checkboxes.items():
            cb.stateChanged.connect(lambda state, c=ch: self.toggle_record(c, state))

        p.btn_start_corr.clicked.connect(self.start_correlation)
        p.btn_stop_corr.clicked.connect(self.stop_correlation)
        p.btn_offline.clicked.connect(self.load_offline_file)

        p.btn_rgb_start.clicked.connect(self.start_rgb)
        p.btn_rgb_stop.clicked.connect(self.stop_rgb)

    def _setup_timers(self):
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self._gui_tick)
        self.draw_timer.start(1000 // GUI_FPS)

    def _gui_tick(self):
        self.plot_area.redraw(self.get_shown_channels())
        self._process_corr_results()

    def _process_corr_results(self):
        results = self.correlation.drain_results()
        if not results:
            return
        for r, _ in results:
            if self.correlation.corr_window:
                self.correlation.corr_window.add_point(r)
        last_r = results[-1][0]
        cond = self.correlation.check_sound_condition(last_r)
        color = "lime" if cond else "gray"
        self.panel.indicator.setStyleSheet(f"font-size:24px; color:{color};")

    # LSL

    def refresh_streams(self):
        self.panel.stream_list.clear()
        self.streams = resolve_streams()
        for s in self.streams:
            self.panel.stream_list.addItem(f"{s.name()} | {s.channel_count()} ch | {s.nominal_srate()} Hz")

    def connect_stream(self):
        idx = self.panel.stream_list.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "LSL", "Select a stream first")
            return

        info = self.streams[idx]
        self.inlet = StreamInlet(info, max_buflen=2)
        self.fs = int(info.nominal_srate())

        for ch in CHANNEL_NAMES:
            if ch in self.panel.show_checkboxes:
                self.channel_show[ch] = self.panel.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.panel.record_checkboxes[ch].isChecked()

        self.plot_area.setup(self.fs, self.available_channels, self.channel_show)
        self._start_log()
        self._start_data_thread()

        self.panel.cb_rec_all.setEnabled(False)
        for cb in self.panel.record_checkboxes.values():
            cb.setEnabled(False)

    def disconnect_stream(self):
        self._stop_data_thread()
        self.inlet = None
        self._stop_log()

        for ch in CHANNEL_NAMES:
            if ch in self.panel.show_checkboxes:
                self.channel_show[ch] = self.panel.show_checkboxes[ch].isChecked()
                self.channel_record[ch] = self.panel.record_checkboxes[ch].isChecked()

        self.panel.cb_rec_all.setEnabled(True)
        for cb in self.panel.record_checkboxes.values():
            cb.setEnabled(True)

    # Data thread

    def _start_data_thread(self):
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.stop()
            self.data_thread.join(timeout=1.0)

        while not self.correlation.job_q.empty():
            try:
                self.correlation.job_q.get_nowait()
            except queue.Empty:
                break

        self.data_thread = DataThread(
            inlet=self.inlet,
            fs=self.fs,
            log_file=self.log_file,
            log_writer=self.log_writer,
            disp_lock=self.disp_lock,
            disp_buffers=self.disp_buffers,
            corr_job_q=self.correlation.job_q,
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/log_{ts}.txt"
        self.log_file = open(self.log_filename, "w", newline="", buffering=65536)
        self.log_writer = csv.writer(self.log_file, delimiter=" ")
        self.log_writer.writerow(["timestamp", *self.get_active_channels()])
        print("Log started:", self.log_filename)

    def _stop_log(self):
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = self.log_writer = None
            print("Log stopped")

    # Channel visibility / recording

    def get_shown_channels(self):
        return [ch for ch in self.available_channels if self.channel_show.get(ch, False)]

    def get_recorded_channels(self):
        return [ch for ch in self.available_channels if self.channel_record.get(ch, False)]

    def get_active_channels(self):
        return self.get_recorded_channels()

    def _toggle_all_show(self, state):
        checked = state == Qt.Checked
        for ch, cb in self.panel.show_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.channel_show[ch] = checked
            if ch in self.plot_area.plots:
                self.plot_area.plots[ch].setVisible(checked)
            if not checked:
                with self.disp_lock:
                    if ch in self.disp_buffers:
                        self.disp_buffers[ch].clear()

    def _toggle_all_record(self, state):
        checked = state == Qt.Checked
        for ch, cb in self.panel.record_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.channel_record[ch] = checked
        if self.inlet and self.data_thread:
            self._restart_log_for_active_channels()

    def toggle_show(self, ch, state):
        self.channel_show[ch] = state == Qt.Checked
        if ch in self.plot_area.plots:
            self.plot_area.plots[ch].setVisible(self.channel_show[ch])
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

    # Correlation

    def start_correlation(self):
        if not self.inlet or not self.data_thread:
            QMessageBox.warning(self, "LSL", "Not connected!")
            return

        p = self.panel
        self.correlation.cache_sound_params(
            mode=p.mode_combo.currentIndex(),
            invert=p.invert_check.isChecked(),
            thr=p.thr.value(),
            rmin=p.range_min.value(),
            rmax=p.range_max.value(),
        )
        self.correlation.start(
            data_thread=self.data_thread,
            win=p.win.value(),
            stp=p.step.value(),
            ch1=p.ch1.currentText(),
            ch2=p.ch2.currentText(),
            thr_mode=p.mode_combo.currentIndex(),
            thr_val=p.thr.value(),
            rmin=p.range_min.value(),
            rmax=p.range_max.value(),
        )
        p.indicator.setStyleSheet("font-size:24px; color:gray;")

    def stop_correlation(self):
        self.correlation.stop(self.data_thread)
        self.panel.indicator.setStyleSheet("font-size:24px; color:gray;")

    def load_offline_file(self):
        self.correlation.load_offline_file(
            win=self.panel.win.value(),
            stp=self.panel.step.value(),
        )

    # RGB

    def start_rgb(self):
        p = self.panel
        self.rgb.start(
            data_thread=self.data_thread,
            rgb_ch_combos=p.rgb_ch,
            win=p.win.value(),
            step=p.step.value(),
        )

    def stop_rgb(self):
        self.rgb.stop()

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
