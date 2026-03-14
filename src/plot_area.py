import threading
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QDoubleSpinBox, QGraphicsProxyWidget
from pyqtgraph import GraphicsLayoutWidget

from .config import CHANNEL_NAMES, DISPLAY_SEC, ch_color
from .viewbox import TimeZoomViewBox


class PlotArea:
    def __init__(self, disp_lock, disp_buffers):
        self.disp_lock: threading.Lock = disp_lock
        self.disp_buffers: dict = disp_buffers

        self.widget = GraphicsLayoutWidget()

        self.plots: dict = {}
        self.curves: dict = {}
        self.gains: dict = {}
        self.y_scale_spinboxes: dict = {}
        self._layout_guard = None
        self._syncing_x = False

    def setup(self, fs, available_channels, channel_show):
        self._clear()

        disp_size = max(int(fs * DISPLAY_SEC), 256)

        with self.disp_lock:
            for ch in available_channels:
                self.disp_buffers[ch] = deque(maxlen=disp_size)

        for i, ch in enumerate(available_channels):
            vb = TimeZoomViewBox()
            p = self.widget.addPlot(row=i, col=0, viewBox=vb)

            p.setLabel("left", ch)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.hideButtons()
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=True)
            p.enableAutoRange(x=False, y=True)
            p.setXRange(-(disp_size - 1), 0, padding=0)

            if i < len(available_channels) - 1:
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

            p.setVisible(channel_show.get(ch, True))

        self._layout_guard = pg.LabelItem("")
        self.widget.addItem(self._layout_guard, row=len(available_channels), col=0)

    def _clear(self):
        self.widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.gains.clear()
        self.y_scale_spinboxes.clear()

        with self.disp_lock:
            self.disp_buffers.clear()

    def redraw(self, shown_channels):
        if not self.plots:
            return

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
