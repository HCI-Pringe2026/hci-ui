import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class CorrelationWindow(QWidget):
    MAXLEN = 50

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
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.curve = self.plot.plot(pen=pg.mkPen("y", width=2))
        layout.addWidget(self.plot)

        self._thr_pos = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", width=1, style=Qt.DashLine))
        self._thr_neg = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", width=1, style=Qt.DashLine))
        self.plot.addItem(self._thr_pos)
        self.plot.addItem(self._thr_neg)
        self._thr_pos.setVisible(False)
        self._thr_neg.setVisible(False)

        self._ys = np.full(self.MAXLEN, np.nan, dtype=np.float32)
        self._ptr = 0
        self._cnt = 0
        self._idx = 0

    def add_point(self, r):
        self._ys[self._ptr] = r
        self._ptr = (self._ptr + 1) % self.MAXLEN
        self._cnt = min(self._cnt + 1, self.MAXLEN)
        self._idx += 1

        if self._cnt < self.MAXLEN:
            ys = self._ys[: self._cnt]
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
