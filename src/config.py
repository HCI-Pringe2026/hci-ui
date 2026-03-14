import pyqtgraph as pg

CHANNEL_NAMES = [
    "FP1-A1",
    "FP2-A2",
    "F3-A1",
    "F4-A2",
    "C3-A1",
    "C4-A2",
    "P3-A1",
    "P4-A2",
    "O1-A1",
    "O2-A2",
    "F7-A1",
    "F8-A2",
    "T3-A1",
    "T4-A2",
    "T5-A1",
    "T6-A2",
    "ECG",
]
N_CH = len(CHANNEL_NAMES)

GUI_FPS = 25
DISPLAY_SEC = 5
LOG_FLUSH_N = 50


def ch_color(i):
    return pg.intColor(i, hues=N_CH)
