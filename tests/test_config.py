from src.config import CHANNEL_NAMES, DISPLAY_SEC, GUI_FPS, LOG_FLUSH_N, N_CH


def test_channel_names_not_empty():
    assert len(CHANNEL_NAMES) > 0


def test_n_ch_matches_channel_names():
    assert len(CHANNEL_NAMES) == N_CH


def test_channel_names_are_unique():
    assert len(CHANNEL_NAMES) == len(set(CHANNEL_NAMES))


def test_channel_names_are_strings():
    assert all(isinstance(ch, str) for ch in CHANNEL_NAMES)


def test_ecg_present():
    assert "ECG" in CHANNEL_NAMES


def test_constants_positive():
    assert GUI_FPS > 0
    assert DISPLAY_SEC > 0
    assert LOG_FLUSH_N > 0
