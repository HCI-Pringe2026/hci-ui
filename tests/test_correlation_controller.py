import queue
import threading

import pytest

from src.correlation import CorrelationController


@pytest.fixture
def ctrl(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from PyQt5.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    return CorrelationController(parent)


class TestSoundCondition:
    def test_threshold_mode_above(self, ctrl):
        ctrl.cache_sound_params(mode=0, invert=False, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.8) is True

    def test_threshold_mode_below(self, ctrl):
        ctrl.cache_sound_params(mode=0, invert=False, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.5) is False

    def test_threshold_mode_inverted(self, ctrl):
        ctrl.cache_sound_params(mode=0, invert=True, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.8) is False
        assert ctrl.check_sound_condition(0.5) is True

    def test_range_mode_inside(self, ctrl):
        ctrl.cache_sound_params(mode=1, invert=False, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.5) is True

    def test_range_mode_outside(self, ctrl):
        ctrl.cache_sound_params(mode=1, invert=False, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.9) is False

    def test_range_mode_inverted(self, ctrl):
        ctrl.cache_sound_params(mode=1, invert=True, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.5) is False
        assert ctrl.check_sound_condition(0.9) is True

    def test_threshold_boundary_exact(self, ctrl):
        ctrl.cache_sound_params(mode=0, invert=False, thr=0.7, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(0.7) is False

    def test_negative_r_threshold(self, ctrl):
        ctrl.cache_sound_params(mode=0, invert=False, thr=0.5, rmin=-0.3, rmax=0.8)
        assert ctrl.check_sound_condition(-0.8) is True


class TestDrainResults:
    def test_empty_queue_returns_empty_list(self, ctrl):
        assert ctrl.drain_results() == []

    def test_drains_all_items(self, ctrl):
        ctrl.result_q.put((0.5, 1.0))
        ctrl.result_q.put((0.8, 2.0))
        results = ctrl.drain_results()
        assert len(results) == 2
        assert results[0] == (0.5, 1.0)
        assert results[1] == (0.8, 2.0)

    def test_queue_empty_after_drain(self, ctrl):
        ctrl.result_q.put((0.1, 0.0))
        ctrl.drain_results()
        assert ctrl.drain_results() == []


class TestCorrThread:
    def test_computes_and_posts_result(self, ctrl, tmp_path):  # noqa: ARG002
        import csv

        from src.threads import CorrThread

        out = tmp_path / "corr.txt"
        f = open(out, "w", newline="")
        writer = csv.writer(f, delimiter=" ")

        n = 50
        data = [list(range(17)) for _ in range(n)]

        job_q = queue.Queue()
        result_q = queue.Queue()
        lock = threading.Lock()

        t = CorrThread(
            i1=0,
            i2=1,
            job_q=job_q,
            result_q=result_q,
            corr_file=f,
            corr_writer=writer,
            lock=lock,
        )
        t.start()
        job_q.put((data, 123.456))
        r, ts = result_q.get(timeout=2.0)
        t.stop()
        t.join(timeout=1.0)
        f.close()

        assert -1.0 <= r <= 1.0
        assert ts == pytest.approx(123.456)

    def test_constant_signal_gives_zero(self, ctrl, tmp_path):  # noqa: ARG002
        import csv

        from src.threads import CorrThread

        out = tmp_path / "corr_const.txt"
        f = open(out, "w", newline="")
        writer = csv.writer(f, delimiter=" ")

        n = 50
        data = [[1.0] * 17 for _ in range(n)]

        job_q = queue.Queue()
        result_q = queue.Queue()
        lock = threading.Lock()

        t = CorrThread(
            i1=0,
            i2=1,
            job_q=job_q,
            result_q=result_q,
            corr_file=f,
            corr_writer=writer,
            lock=lock,
        )
        t.start()
        job_q.put((data, 0.0))
        r, _ = result_q.get(timeout=2.0)
        t.stop()
        t.join(timeout=1.0)
        f.close()

        assert r == pytest.approx(0.0)
