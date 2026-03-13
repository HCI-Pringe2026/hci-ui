import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet


def main():
    info = StreamInfo('OscillatorStream', 'EEG', 20, 500, 'float32', 'oscillator123')
    outlet = StreamOutlet(info)

    print("Streaming started. Channel 0 oscillates, channels 1-19 are zeros.")
    print("Press Ctrl+C to stop...")

    frequency = 10  # Hz
    amplitude = 100  # microvolts
    sampling_rate = 500  # Hz
    phase = 0

    try:
        while True:
            phase += 2 * np.pi * frequency / sampling_rate
            if phase > 2 * np.pi:
                phase -= 2 * np.pi

            sample = np.zeros(20, dtype=np.float32)
            sample[0] = amplitude * np.sin(phase)
            outlet.push_sample(sample)
            time.sleep(1.0 / sampling_rate)

    except KeyboardInterrupt:
        print("\nStreaming stopped.")


if __name__ == "__main__":
    main()
