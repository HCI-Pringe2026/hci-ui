from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rgb import RGBController


def make_rgb_bar(qtbot):
    from PyQt5.QtWidgets import QLabel, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    bar = QLabel(parent)
    return parent, bar


def make_mock_data_thread(ring_values=None, win=10):
    dt = MagicMock()
    if ring_values is None:
        vals = list(range(win))
        ring_values = [deque(vals, maxlen=win) for _ in range(6)]
    dt.rgb_rings = ring_values
    return dt


@pytest.fixture
def ctrl(qtbot):
    parent, bar = make_rgb_bar(qtbot)
    return RGBController(parent=parent, rgb_bar=bar)


class TestRGBControllerInit:
    def test_initially_inactive(self, ctrl):
        assert ctrl.active is False

    def test_no_file_initially(self, ctrl):
        assert ctrl.rgb_file is None
        assert ctrl.rgb_writer is None

    def test_no_timer_initially(self, ctrl):
        assert ctrl._timer is None


class TestRGBControllerStop:
    def test_stop_sets_inactive(self, ctrl):
        ctrl.active = True
        ctrl.stop()
        assert ctrl.active is False

    def test_stop_resets_bar_color(self, ctrl):
        ctrl.rgb_bar.setStyleSheet("background-color: rgb(255,0,0);")
        ctrl.stop()
        assert "black" in ctrl.rgb_bar.styleSheet()

    def test_stop_closes_file(self, ctrl, tmp_path):
        f = open(tmp_path / "rgb.txt", "w")
        ctrl.rgb_file = f
        ctrl.rgb_writer = MagicMock()
        ctrl.stop()
        assert f.closed
        assert ctrl.rgb_file is None


class TestRGBControllerTick:
    def _setup_active(self, ctrl, rings, win):
        ctrl.active = True
        ctrl._data_thread = make_mock_data_thread(rings, win)
        ctrl._win = win

    def test_tick_updates_bar_color(self, ctrl):
        win = 5
        vals = list(np.linspace(0, 1, win))
        rings = [deque(vals, maxlen=win) for _ in range(6)]
        self._setup_active(ctrl, rings, win)
        ctrl._tick()
        assert "rgb(" in ctrl.rgb_bar.styleSheet()

    def test_tick_skipped_when_inactive(self, ctrl):
        ctrl.active = False
        ctrl.rgb_bar.setStyleSheet("background-color:black;")
        ctrl._tick()
        assert "black" in ctrl.rgb_bar.styleSheet()

    def test_tick_skipped_when_not_enough_data(self, ctrl):
        win = 50
        rings = [deque([1.0], maxlen=win) for _ in range(6)]
        self._setup_active(ctrl, rings, win)
        ctrl.rgb_bar.setStyleSheet("background-color:black;")
        ctrl._tick()
        assert "black" in ctrl.rgb_bar.styleSheet()

    def test_tick_increments_counter(self, ctrl):
        win = 5
        vals = list(range(win))
        rings = [deque(vals, maxlen=win) for _ in range(6)]
        self._setup_active(ctrl, rings, win)
        ctrl._tick_cnt = 0
        ctrl._tick()
        assert ctrl._tick_cnt == 1

    def test_tick_writes_to_file(self, ctrl, tmp_path):
        import csv

        win = 5
        vals = list(np.linspace(0, 1, win))
        rings = [deque(vals, maxlen=win) for _ in range(6)]
        self._setup_active(ctrl, rings, win)

        f = open(tmp_path / "rgb_out.txt", "w", newline="")
        ctrl.rgb_file = f
        ctrl.rgb_writer = csv.writer(f, delimiter=" ")

        ctrl._tick()
        f.flush()
        f.close()

        lines = open(tmp_path / "rgb_out.txt").readlines()
        assert len(lines) == 1

    def test_color_is_valid_rgb(self, ctrl):
        win = 5
        vals = list(np.linspace(0, 1, win))
        rings = [deque(vals, maxlen=win) for _ in range(6)]
        self._setup_active(ctrl, rings, win)
        ctrl._tick()

        style = ctrl.rgb_bar.styleSheet()
        import re

        match = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", style)
        assert match is not None
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255
