import numpy as np
import pytest

from src.corr_window import CorrelationWindow


@pytest.fixture
def win(qtbot):
    w = CorrelationWindow()
    qtbot.addWidget(w)
    return w


def test_initial_state(win):
    assert win._ptr == 0
    assert win._cnt == 0
    assert win._idx == 0
    assert np.all(np.isnan(win._ys))


def test_add_single_point(win):
    win.add_point(0.5)
    assert win._cnt == 1
    assert win._idx == 1
    assert win._ys[0] == pytest.approx(0.5)


def test_add_points_increments_index(win):
    for i in range(5):
        win.add_point(float(i) / 4)
    assert win._idx == 5
    assert win._cnt == 5


def test_ring_wraps_at_maxlen(win):
    maxlen = CorrelationWindow.MAXLEN
    for _ in range(maxlen + 10):
        win.add_point(0.1)
    assert win._cnt == maxlen
    assert win._idx == maxlen + 10


def test_reset_clears_state(win):
    win.add_point(0.9)
    win.add_point(-0.5)
    win.reset()
    assert win._ptr == 0
    assert win._cnt == 0
    assert win._idx == 0
    assert np.all(np.isnan(win._ys))


def test_threshold_lines_abs_mode(win):
    win.set_threshold_lines("abs", 0.7)
    assert win._thr_pos.isVisible()
    assert win._thr_neg.isVisible()
    assert win._thr_pos.value() == pytest.approx(0.7)
    assert win._thr_neg.value() == pytest.approx(-0.7)


def test_threshold_lines_range_mode(win):
    win.set_threshold_lines("range", -0.3, 0.8)
    assert win._thr_pos.isVisible()
    assert win._thr_neg.isVisible()
    assert win._thr_pos.value() == pytest.approx(0.8)
    assert win._thr_neg.value() == pytest.approx(-0.3)


def test_threshold_lines_hidden_for_unknown_mode(win):
    win.set_threshold_lines("abs", 0.5)
    win.set_threshold_lines("none", 0.0)
    assert not win._thr_pos.isVisible()
    assert not win._thr_neg.isVisible()
