import pytest

from src.config import CHANNEL_NAMES, N_CH
from src.panels import LeftPanel


@pytest.fixture
def panel(qtbot):
    p = LeftPanel()
    qtbot.addWidget(p)
    return p


class TestLeftPanelStructure:
    def test_stream_list_exists(self, panel):
        assert panel.stream_list is not None

    def test_connection_buttons_exist(self, panel):
        assert panel.btn_refresh is not None
        assert panel.btn_connect is not None
        assert panel.btn_disconnect is not None

    def test_show_checkboxes_for_all_channels(self, panel):
        assert set(panel.show_checkboxes.keys()) == set(CHANNEL_NAMES)

    def test_record_checkboxes_for_all_channels(self, panel):
        assert set(panel.record_checkboxes.keys()) == set(CHANNEL_NAMES)

    def test_all_show_checkboxes_checked_by_default(self, panel):
        assert all(cb.isChecked() for cb in panel.show_checkboxes.values())

    def test_all_record_checkboxes_checked_by_default(self, panel):
        assert all(cb.isChecked() for cb in panel.record_checkboxes.values())

    def test_cb_show_all_checked_by_default(self, panel):
        assert panel.cb_show_all.isChecked()

    def test_cb_rec_all_checked_by_default(self, panel):
        assert panel.cb_rec_all.isChecked()

    def test_corr_channel_combos_exist(self, panel):
        assert panel.ch1 is not None
        assert panel.ch2 is not None

    def test_ch2_default_index_is_1(self, panel):
        assert panel.ch2.currentIndex() == 1

    def test_win_spinbox_default(self, panel):
        assert panel.win.value() == 500

    def test_step_spinbox_default(self, panel):
        assert panel.step.value() == 50

    def test_thr_spinbox_default(self, panel):
        assert panel.thr.value() == pytest.approx(0.7)

    def test_range_min_default(self, panel):
        assert panel.range_min.value() == pytest.approx(-0.3)

    def test_range_max_default(self, panel):
        assert panel.range_max.value() == pytest.approx(0.8)

    def test_rgb_ch_combos_count(self, panel):
        assert len(panel.rgb_ch) == 6

    def test_rgb_ch_combos_have_channel_names(self, panel):
        for cb in panel.rgb_ch:
            assert cb.count() == N_CH

    def test_rgb_bar_exists(self, panel):
        assert panel.rgb_bar is not None

    def test_indicator_label_exists(self, panel):
        assert panel.indicator is not None

    def test_corr_buttons_exist(self, panel):
        assert panel.btn_start_corr is not None
        assert panel.btn_stop_corr is not None
        assert panel.btn_offline is not None

    def test_rgb_buttons_exist(self, panel):
        assert panel.btn_rgb_start is not None
        assert panel.btn_rgb_stop is not None
