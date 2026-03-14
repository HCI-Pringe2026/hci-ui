from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import CHANNEL_NAMES, N_CH


class LeftPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(340)

        inner = QWidget()
        left = QVBoxLayout(inner)
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

        left.addWidget(self._build_channel_box())
        left.addWidget(self._build_corr_box())
        left.addWidget(self._build_rgb_box())
        left.addStretch()

        self.setWidget(inner)

    def _centered_cb(self, cb):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setAlignment(Qt.AlignHCenter)
        h.addWidget(cb)
        return w

    def _build_channel_box(self):
        box = QGroupBox("Channels")
        grid = QGridLayout(box)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        for col, label in enumerate(("Channel", "Show", "Rec")):
            hdr = QLabel(label)
            hdr.setAlignment(Qt.AlignCenter)
            grid.addWidget(hdr, 0, col)

        all_lbl = QLabel("All")
        all_lbl.setAlignment(Qt.AlignCenter)
        grid.addWidget(all_lbl, 1, 0)

        self.cb_show_all = QCheckBox()
        self.cb_show_all.setChecked(True)
        self.cb_show_all.setTristate(False)
        grid.addWidget(self._centered_cb(self.cb_show_all), 1, 1)

        self.cb_rec_all = QCheckBox()
        self.cb_rec_all.setChecked(True)
        self.cb_rec_all.setTristate(False)
        grid.addWidget(self._centered_cb(self.cb_rec_all), 1, 2)

        self.show_checkboxes = {}
        self.record_checkboxes = {}

        for row_i, ch in enumerate(CHANNEL_NAMES, start=2):
            name_lbl = QLabel(ch)
            name_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            cb_show = QCheckBox()
            cb_show.setChecked(True)

            cb_rec = QCheckBox()
            cb_rec.setChecked(True)

            grid.addWidget(name_lbl, row_i, 0)
            grid.addWidget(self._centered_cb(cb_show), row_i, 1)
            grid.addWidget(self._centered_cb(cb_rec), row_i, 2)

            self.show_checkboxes[ch] = cb_show
            self.record_checkboxes[ch] = cb_rec

        return box

    def _build_corr_box(self):
        box = QGroupBox("Spearman (2 channels, online)")
        cl = QVBoxLayout(box)

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

        return box

    def _build_rgb_box(self):
        box = QGroupBox("RGB Spearman (6 channels)")
        rl = QVBoxLayout(box)

        self.rgb_ch = []
        for i in range(6):
            cb = QComboBox()
            cb.addItems(CHANNEL_NAMES)
            if i < N_CH:
                cb.setCurrentIndex(i)
            self.rgb_ch.append(cb)
            rl.addWidget(QLabel(f"Channel {i + 1}"))
            rl.addWidget(cb)

        self.btn_rgb_start = QPushButton("Start RGB")
        self.btn_rgb_stop = QPushButton("Stop RGB")
        rl.addWidget(self.btn_rgb_start)
        rl.addWidget(self.btn_rgb_stop)

        self.rgb_bar = QLabel()
        self.rgb_bar.setFixedHeight(25)
        self.rgb_bar.setStyleSheet("background-color:black;")
        rl.addWidget(self.rgb_bar)

        return box
