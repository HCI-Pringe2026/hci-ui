from typing import override

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


class TimeZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.setMouseMode(self.PanMode)
        self.setDefaultPadding(0.0)
        self._yscale_spinbox = None

    @override
    def wheelEvent(self, ev, _axis=None):
        mods = QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier:
            self.enableAutoRange(axis="y", enable=False)
            super().wheelEvent(ev, axis=1)
        else:
            delta = ev.delta()
            if delta == 0:
                ev.accept()
                return
            zoom_factor = 1.15 if delta > 0 else 1.0 / 1.15
            x_min, x_max = self.viewRange()[0]
            new_width = (x_max - x_min) / zoom_factor
            self.enableAutoRange(axis="x", enable=False)
            self.setXRange(x_max - new_width, 0, padding=0)
            ev.accept()

    @override
    def mouseDragEvent(self, ev, _axis=None):
        ev.accept()

    @override
    def mouseDoubleClickEvent(self, ev):
        self.enableAutoRange(axis="y", enable=True)
        if self._yscale_spinbox is not None:
            self._yscale_spinbox.blockSignals(True)
            self._yscale_spinbox.setValue(0)
            self._yscale_spinbox.blockSignals(False)
        ev.accept()
