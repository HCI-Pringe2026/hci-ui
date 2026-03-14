import queue
import threading
import time
from collections import deque
from unittest.mock import MagicMock

import pytest

from src.config import CHANNEL_NAMES
from src.threads import DataThread


def make_thread(chunks, tmp_path):
    """Build a DataThread with a mock inlet that yields given chunks then blocks."""
    call_count = 0

    def pull_chunk(timeout, max_samples):  # noqa: ARG001
        nonlocal call_count
        if call_count < len(chunks):
            result = chunks[call_count]
            call_count += 1
            return result
        time.sleep(timeout)
        return [], []

    inlet = MagicMock()
    inlet.pull_chunk.side_effect = pull_chunk

    log_file = open(tmp_path / "log.txt", "w", newline="")
    import csv

    log_writer = csv.writer(log_file, delimiter=" ")
    log_writer.writerow(["timestamp", *CHANNEL_NAMES])

    disp_lock = threading.Lock()
    disp_buffers = {ch: deque(maxlen=500) for ch in CHANNEL_NAMES}
    corr_job_q = queue.Queue(maxsize=4)

    t = DataThread(
        inlet=inlet,
        fs=250,
        log_file=log_file,
        log_writer=log_writer,
        disp_lock=disp_lock,
        disp_buffers=disp_buffers,
        corr_job_q=corr_job_q,
        active_channels_getter=lambda: CHANNEL_NAMES,
        shown_channels_getter=lambda: CHANNEL_NAMES,
    )
    return t, disp_lock, disp_buffers, corr_job_q, log_file


def make_sample(val=1.0):
    return [val] * len(CHANNEL_NAMES)


class TestDataThreadBuffering:
    def test_samples_reach_disp_buffers(self, tmp_path):
        sample = make_sample(42.0)
        chunks = [([sample, sample], [1.0, 2.0])]

        t, disp_lock, disp_buffers, _, log_file = make_thread(chunks, tmp_path)
        t.start()
        time.sleep(0.15)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        with disp_lock:
            buf = list(disp_buffers[CHANNEL_NAMES[0]])
        assert len(buf) == 2
        assert all(v == pytest.approx(42.0) for v in buf)

    def test_multiple_chunks_accumulate(self, tmp_path):
        sample = make_sample(1.0)
        chunks = [
            ([sample] * 5, [float(i) for i in range(5)]),
            ([sample] * 3, [float(i) for i in range(5, 8)]),
        ]

        t, disp_lock, disp_buffers, _, log_file = make_thread(chunks, tmp_path)
        t.start()
        time.sleep(0.2)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        with disp_lock:
            buf = list(disp_buffers[CHANNEL_NAMES[0]])
        assert len(buf) == 8


class TestDataThreadCorrelation:
    def test_corr_job_posted_after_step(self, tmp_path):
        n_samples = 60
        sample = make_sample(1.0)
        chunks = [([sample] * n_samples, list(range(n_samples)))]

        t, _, _, corr_job_q, log_file = make_thread(chunks, tmp_path)
        t.set_corr(active=True, window_size=50, step_size=50)
        t.start()
        time.sleep(0.2)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        assert not corr_job_q.empty()
        window, _ts = corr_job_q.get_nowait()
        assert len(window) >= 50

    def test_no_corr_job_when_inactive(self, tmp_path):
        sample = make_sample(1.0)
        chunks = [([sample] * 100, list(range(100)))]

        t, _, _, corr_job_q, log_file = make_thread(chunks, tmp_path)
        t.start()
        time.sleep(0.2)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        assert corr_job_q.empty()


class TestDataThreadRGB:
    def test_rgb_rings_fill_when_active(self, tmp_path):
        sample = [float(i) for i in range(len(CHANNEL_NAMES))]
        chunks = [([sample] * 20, list(range(20)))]

        t, _, _, _, log_file = make_thread(chunks, tmp_path)
        t.set_rgb(active=True, ch_indices=list(range(6)), window_size=100)
        t.start()
        time.sleep(0.2)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        assert len(t.rgb_rings[0]) == 20

    def test_rgb_rings_empty_when_inactive(self, tmp_path):
        sample = make_sample(1.0)
        chunks = [([sample] * 20, list(range(20)))]

        t, _, _, _, log_file = make_thread(chunks, tmp_path)
        t.start()
        time.sleep(0.2)
        t.stop()
        t.join(timeout=1.0)
        log_file.close()

        assert all(len(ring) == 0 for ring in t.rgb_rings)


class TestDataThreadSetCorr:
    def test_set_corr_updates_params(self, tmp_path):
        t, _, _, _, log_file = make_thread([], tmp_path)
        t.set_corr(active=True, window_size=200, step_size=25)
        assert t.corr_active is True
        assert t.window_size == 200
        assert t.step_size == 25
        log_file.close()

    def test_set_corr_resets_ring(self, tmp_path):
        t, _, _, _, log_file = make_thread([], tmp_path)
        t.corr_ring.append([1, 2, 3])
        t.set_corr(active=False)
        assert len(t.corr_ring) == 0
        log_file.close()

    def test_set_rgb_updates_params(self, tmp_path):
        t, _, _, _, log_file = make_thread([], tmp_path)
        t.set_rgb(active=True, ch_indices=[0, 1, 2, 3, 4, 5], window_size=300)
        assert t.rgb_active is True
        assert t.rgb_ch_idx == [0, 1, 2, 3, 4, 5]
        assert all(ring.maxlen == 300 for ring in t.rgb_rings)
        log_file.close()
