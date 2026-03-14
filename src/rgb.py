import csv
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget
from scipy.stats import spearmanr

from .config import CHANNEL_NAMES


class RGBController:
    def __init__(self, parent: QWidget, rgb_bar: QLabel):
        self.parent = parent
        self.rgb_bar = rgb_bar

        self.active = False
        self.rgb_file = None
        self.rgb_writer = None
        self._tick_cnt = 0
        self._timer: QTimer | None = None

    def start(self, data_thread, rgb_ch_combos, win, step):
        if not data_thread:
            QMessageBox.warning(self.parent, "LSL", "Not connected!")
            return

        idx = [
            CHANNEL_NAMES.index(cb.currentText()) if cb.currentText() in CHANNEL_NAMES else 0 for cb in rgb_ch_combos
        ]

        data_thread.set_rgb(True, idx, win)
        self.active = True
        self._data_thread = data_thread
        self._win = win

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"logs/rgb_win{win}_step{step}_{ts}.txt"
        self.rgb_file = open(fname, "w", newline="", buffering=8192)
        self.rgb_writer = csv.writer(self.rgb_file, delimiter=" ")
        self.rgb_writer.writerow(["timestamp", "R", "G", "B"])
        self._tick_cnt = 0
        print("RGB started:", fname)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    def stop(self):
        self.active = False
        if self._timer:
            self._timer.stop()
            self._timer = None
        if hasattr(self, "_data_thread") and self._data_thread:
            self._data_thread.set_rgb(False, [], 0)
        if self.rgb_file:
            self.rgb_file.flush()
            self.rgb_file.close()
            self.rgb_file = self.rgb_writer = None
        self.rgb_bar.setStyleSheet("background-color:black;")
        print("RGB stopped")

    def _tick(self):
        if not self.active:
            return

        data_thread = getattr(self, "_data_thread", None)
        if not data_thread:
            return

        win = self._win
        rings = data_thread.rgb_rings

        if any(len(rb) < win for rb in rings):
            return

        rgb_vals = []
        for i in range(0, 6, 2):
            x = np.array(rings[i], dtype=np.float64)[-win:]
            y = np.array(rings[i + 1], dtype=np.float64)[-win:]
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
            self.rgb_writer.writerow([self._tick_cnt, rgb_vals[0], rgb_vals[1], rgb_vals[2]])
        self._tick_cnt += 1
