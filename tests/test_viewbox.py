from unittest.mock import MagicMock

import pytest

from src.viewbox import TimeZoomViewBox


@pytest.fixture
def vb(qtbot):  # noqa: ARG001
    return TimeZoomViewBox()


class TestTimeZoomViewBoxInit:
    def test_yscale_spinbox_is_none(self, vb):
        assert vb._yscale_spinbox is None

    def test_default_padding_is_zero(self, vb):
        assert vb.suggestPadding(0) == pytest.approx(0.0)


class TestMouseDoubleClick:
    def test_double_click_enables_y_autorange(self, vb):
        vb.enableAutoRange(axis="y", enable=False)
        ev = MagicMock()
        vb.mouseDoubleClickEvent(ev)
        state = vb.autoRangeEnabled()
        assert state[1]

    def test_double_click_resets_spinbox(self, vb, qtbot):
        from PyQt5.QtWidgets import QDoubleSpinBox

        spin = QDoubleSpinBox()
        qtbot.addWidget(spin)
        spin.setValue(100.0)
        vb._yscale_spinbox = spin

        ev = MagicMock()
        vb.mouseDoubleClickEvent(ev)

        assert spin.value() == pytest.approx(0.0)

    def test_double_click_no_spinbox_does_not_crash(self, vb):
        vb._yscale_spinbox = None
        ev = MagicMock()
        vb.mouseDoubleClickEvent(ev)

    def test_double_click_accepts_event(self, vb):
        ev = MagicMock()
        vb.mouseDoubleClickEvent(ev)
        ev.accept.assert_called_once()

    def test_double_click_spinbox_signals_blocked_during_reset(self, vb, qtbot):
        from PyQt5.QtWidgets import QDoubleSpinBox

        spin = QDoubleSpinBox()
        qtbot.addWidget(spin)
        spin.setValue(50.0)
        vb._yscale_spinbox = spin

        signal_fired = []
        spin.valueChanged.connect(lambda v: signal_fired.append(v))

        ev = MagicMock()
        vb.mouseDoubleClickEvent(ev)

        assert signal_fired == []


class TestMouseDrag:
    def test_drag_accepts_event(self, vb):
        ev = MagicMock()
        vb.mouseDragEvent(ev)
        ev.accept.assert_called_once()
