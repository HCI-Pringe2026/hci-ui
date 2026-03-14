import csv
import queue
import threading
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from scipy.stats import spearmanr

from .config import CHANNEL_NAMES
from .corr_window import CorrelationWindow
from .threads import CorrThread


class CorrelationController:
    def __init__(self, parent: QWidget):
        self.parent = parent

        self.job_q: queue.Queue = queue.Queue(maxsize=4)
        self.result_q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

        self.corr_file = None
        self.corr_writer = None
        self.corr_thread: CorrThread | None = None
        self.corr_window: CorrelationWindow | None = None
        self.active = False

        self._snd_mode = 0
        self._snd_invert = False
        self._snd_thr = 0.7
        self._snd_rmin = -0.3
        self._snd_rmax = 0.8

    def cache_sound_params(self, mode, invert, thr, rmin, rmax):
        self._snd_mode = mode
        self._snd_invert = invert
        self._snd_thr = thr
        self._snd_rmin = rmin
        self._snd_rmax = rmax

    def check_sound_condition(self, r):
        cond = abs(r) > self._snd_thr if self._snd_mode == 0 else self._snd_rmin <= r <= self._snd_rmax
        return not cond if self._snd_invert else cond

    def start(self, data_thread, win, stp, ch1, ch2, thr_mode, thr_val, rmin, rmax):
        i1 = CHANNEL_NAMES.index(ch1)
        i2 = CHANNEL_NAMES.index(ch2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"logs/corr_{ch1}_{ch2}_win{win}_step{stp}_{ts}.txt"
        self.corr_file = open(fname, "w", newline="", buffering=8192)
        self.corr_writer = csv.writer(self.corr_file, delimiter=" ")
        self.corr_writer.writerow(["timestamp", f"spearman_{ch1}_{ch2}"])

        for q in (self.job_q, self.result_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self.corr_thread and self.corr_thread.is_alive():
            self.corr_thread.stop()
            self.corr_thread.join(timeout=0.5)

        self.corr_thread = CorrThread(
            i1=i1,
            i2=i2,
            job_q=self.job_q,
            result_q=self.result_q,
            corr_file=self.corr_file,
            corr_writer=self.corr_writer,
            lock=self.lock,
        )
        self.corr_thread.start()

        data_thread.set_corr(True, win, stp)
        self.active = True

        if self.corr_window is None:
            self.corr_window = CorrelationWindow()
        self.corr_window.reset()

        if thr_mode == 0:
            self.corr_window.set_threshold_lines("abs", thr_val)
        else:
            self.corr_window.set_threshold_lines("range", rmin, rmax)

        self.corr_window.show()
        self.corr_window.raise_()
        print(f"Correlation started: {ch1} vs {ch2} win={win} step={stp}")

    def stop(self, data_thread=None):
        self.active = False

        if data_thread:
            data_thread.set_corr(False)

        if self.corr_thread:
            self.corr_thread.stop()
            self.corr_thread.join(timeout=0.5)
            self.corr_thread = None

        with self.lock:
            if self.corr_file:
                self.corr_file.flush()
                self.corr_file.close()
                self.corr_file = self.corr_writer = None

        if self.corr_window:
            self.corr_window.close()
            self.corr_window = None

        print("Correlation stopped")

    def drain_results(self):
        results = []
        while True:
            try:
                results.append(self.result_q.get_nowait())
            except queue.Empty:
                break
        return results

    def load_offline_file(self, win, stp):
        fname, _ = QFileDialog.getOpenFileName(self.parent, "Select log file", "", "Text/CSV (*.txt *.csv)")
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
            QMessageBox.critical(self.parent, "Error", str(e))
            return

        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] < 2:
            QMessageBox.critical(self.parent, "Error", "Not enough rows.")
            return

        sig = data[:, 1:]
        n_ch = sig.shape[1]

        if n_ch != len(names):
            QMessageBox.critical(self.parent, "Error", "Header/data column mismatch.")
            return

        out = fname.rsplit(".", 1)[0] + "_offline_corr.txt"

        with open(out, "w") as f:
            hdr = ["window"] + [f"{names[i]}_{names[j]}" for i in range(n_ch) for j in range(i + 1, n_ch)]
            f.write(" ".join(hdr) + "\n")

            idx = 0
            row = 0
            while idx + win <= sig.shape[0]:
                vals = []
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        x, y = sig[idx : idx + win, i], sig[idx : idx + win, j]
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
        QMessageBox.information(self.parent, "Done", f"Saved:\n{out}")
