import threading

import pytest

from src.config import CHANNEL_NAMES
from src.plot_area import PlotArea


@pytest.fixture
def area(qtbot):
    disp_lock = threading.Lock()
    disp_buffers = {}
    pa = PlotArea(disp_lock, disp_buffers)
    qtbot.addWidget(pa.widget)
    return pa


@pytest.fixture
def area_setup(area):
    channel_show = dict.fromkeys(CHANNEL_NAMES, True)
    area.setup(fs=250, available_channels=CHANNEL_NAMES, channel_show=channel_show)
    return area


class TestPlotAreaSetup:
    def test_plots_created_for_all_channels(self, area_setup):
        assert set(area_setup.plots.keys()) == set(CHANNEL_NAMES)

    def test_curves_created_for_all_channels(self, area_setup):
        assert set(area_setup.curves.keys()) == set(CHANNEL_NAMES)

    def test_gains_created_for_all_channels(self, area_setup):
        assert set(area_setup.gains.keys()) == set(CHANNEL_NAMES)

    def test_gains_default_to_one(self, area_setup):
        for ch, g in area_setup.gains.items():
            assert g.value() == pytest.approx(1.0), f"gain for {ch} should be 1.0"

    def test_y_scale_spinboxes_default_to_zero(self, area_setup):
        for ch, ys in area_setup.y_scale_spinboxes.items():
            assert ys.value() == pytest.approx(0.0), f"y_scale for {ch} should be 0"

    def test_disp_buffers_populated(self, area_setup):
        assert set(area_setup.disp_buffers.keys()) == set(CHANNEL_NAMES)

    def test_hidden_channels_not_visible(self, area):
        channel_show = dict.fromkeys(CHANNEL_NAMES, False)
        channel_show[CHANNEL_NAMES[0]] = True
        area.setup(fs=250, available_channels=CHANNEL_NAMES, channel_show=channel_show)
        assert area.plots[CHANNEL_NAMES[0]].isVisible()
        assert not area.plots[CHANNEL_NAMES[1]].isVisible()

    def test_setup_clears_previous_state(self, area_setup):
        channel_show = dict.fromkeys(CHANNEL_NAMES, True)
        area_setup.setup(fs=250, available_channels=CHANNEL_NAMES, channel_show=channel_show)
        assert len(area_setup.plots) == len(CHANNEL_NAMES)


class TestPlotAreaRedraw:
    def test_redraw_does_nothing_with_no_plots(self, area):
        area.redraw(CHANNEL_NAMES)

    def test_redraw_sets_data_for_shown_channels(self, area_setup):
        ch = CHANNEL_NAMES[0]
        area_setup.disp_buffers[ch].extend([1.0, 2.0, 3.0])
        area_setup.redraw([ch])
        _xdata, ydata = area_setup.curves[ch].getData()
        assert len(ydata) == 3

    def test_redraw_clears_hidden_channels(self, area_setup):
        ch = CHANNEL_NAMES[0]
        area_setup.disp_buffers[ch].extend([1.0, 2.0, 3.0])
        area_setup.redraw([ch])
        area_setup.redraw([])
        _xdata, ydata = area_setup.curves[ch].getData()
        assert ydata is None or len(ydata) == 0

    def test_redraw_applies_gain(self, area_setup):
        ch = CHANNEL_NAMES[0]
        area_setup.gains[ch].setValue(2.0)
        area_setup.disp_buffers[ch].extend([1.0, 2.0, 3.0])
        area_setup.redraw([ch])
        _, ydata = area_setup.curves[ch].getData()
        assert ydata == pytest.approx([2.0, 4.0, 6.0])

    def test_redraw_x_axis_ends_at_zero(self, area_setup):
        ch = CHANNEL_NAMES[0]
        area_setup.disp_buffers[ch].extend([1.0, 2.0, 3.0, 4.0, 5.0])
        area_setup.redraw([ch])
        xdata, _ = area_setup.curves[ch].getData()
        assert xdata[-1] == pytest.approx(0.0)


class TestPlotAreaClear:
    def test_clear_empties_all_dicts(self, area_setup):
        area_setup._clear()
        assert area_setup.plots == {}
        assert area_setup.curves == {}
        assert area_setup.gains == {}
        assert area_setup.y_scale_spinboxes == {}

    def test_clear_empties_disp_buffers(self, area_setup):
        area_setup._clear()
        assert area_setup.disp_buffers == {}
