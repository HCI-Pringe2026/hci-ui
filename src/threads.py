import contextlib
import queue
import threading
import time
from collections import deque

import numpy as np
from scipy.stats import spearmanr

from .config import CHANNEL_NAMES, LOG_FLUSH_N


class DataThread(threading.Thread):
    def __init__(
        self,
        inlet,
        fs,
        log_file,
        log_writer,
        disp_lock,
        disp_buffers,
        corr_job_q,
        active_channels_getter,
        shown_channels_getter,
    ):
        super().__init__(daemon=True)
        self.inlet = inlet
        self.fs = fs
        self.log_file = log_file
        self.log_writer = log_writer
        self.disp_lock = disp_lock
        self.disp_buffers = disp_buffers
        self.corr_job_q = corr_job_q
        self.active_channels_getter = active_channels_getter
        self.shown_channels_getter = shown_channels_getter

        self.running = True

        self.corr_active = False
        self.window_size = 500
        self.step_size = 50
        self.corr_ring = deque()
        self.samples_step = 0

        self.rgb_active = False
        self.rgb_rings = [deque() for _ in range(6)]
        self.rgb_ch_idx = list(range(6))

        self._flush_cnt = 0

    def run(self):
        while self.running:
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.02, max_samples=256)
            if not chunk:
                continue

            active_channels = self.active_channels_getter()
            shown_channels = self.shown_channels_getter()
            active_indices = [CHANNEL_NAMES.index(ch) for ch in active_channels]

            rows = []
            disp_batch = {ch: [] for ch in shown_channels}

            for sample, ts in zip(chunk, timestamps):
                t = ts if ts else time.time()

                row = [f"{t:.6f}"]
                for i in active_indices:
                    row.append(f"{sample[i]:.4f}" if i < len(sample) else "0.0000")
                rows.append(row)

                for i, ch in enumerate(CHANNEL_NAMES):
                    if ch in disp_batch and i < len(sample):
                        disp_batch[ch].append(sample[i])

                if self.corr_active:
                    self.corr_ring.append(sample)
                    self.samples_step += 1
                    if self.samples_step >= self.step_size:
                        self.samples_step = 0
                        if len(self.corr_ring) >= self.window_size:
                            with contextlib.suppress(queue.Full):
                                self.corr_job_q.put_nowait((list(self.corr_ring), t))

                if self.rgb_active:
                    for k, ci in enumerate(self.rgb_ch_idx):
                        if ci < len(sample):
                            self.rgb_rings[k].append(sample[ci])

            for row in rows:
                self.log_writer.writerow(row)
            self._flush_cnt += len(rows)
            if self._flush_cnt >= LOG_FLUSH_N:
                self.log_file.flush()
                self._flush_cnt = 0

            with self.disp_lock:
                for ch in shown_channels:
                    if ch in self.disp_buffers:
                        self.disp_buffers[ch].extend(disp_batch[ch])

    def stop(self):
        self.running = False

    def set_corr(self, active, window_size=500, step_size=50):
        self.window_size = window_size
        self.step_size = step_size
        self.corr_ring = deque(maxlen=window_size)
        self.samples_step = 0
        self.corr_active = active

    def set_rgb(self, active, ch_indices, window_size=500):
        self.rgb_ch_idx = ch_indices
        self.rgb_rings = [deque(maxlen=window_size) for _ in range(6)]
        self.rgb_active = active


class CorrThread(threading.Thread):
    def __init__(self, i1, i2, job_q, result_q, corr_file, corr_writer, lock, sound_callback=None):
        super().__init__(daemon=True)
        self.i1 = i1
        self.i2 = i2
        self.job_q = job_q
        self.result_q = result_q
        self.corr_file = corr_file
        self.corr_writer = corr_writer
        self.lock = lock
        self.running = True
        self.sound_callback = sound_callback

    def run(self):
        while self.running:
            try:
                window, t = self.job_q.get(timeout=0.1)
            except queue.Empty:
                continue

            i1, i2 = self.i1, self.i2
            try:
                x = np.array([s[i1] for s in window if i1 < len(s)], dtype=np.float64)
                y = np.array([s[i2] for s in window if i2 < len(s)], dtype=np.float64)
            except Exception:
                continue

            if len(x) < 2 or len(y) < 2:
                continue

            if np.all(x == x[0]) or np.all(y == y[0]):
                r = 0.0
            else:
                r, _ = spearmanr(x, y)
                if not np.isfinite(r):
                    r = 0.0

            with self.lock:
                if self.corr_writer:
                    self.corr_writer.writerow([f"{t:.6f}", f"{r:.6f}"])
                    self.corr_file.flush()

            with contextlib.suppress(queue.Full):
                self.result_q.put_nowait((r, t))

    def stop(self):
        self.running = False
