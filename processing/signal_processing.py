import numpy as np
from scipy.signal import butter, filtfilt


class SignalProcessor:
    """
    rPPG signal processing:
    - Normalization
    - Bandpass filtering
    - FFT-based BPM estimation
    """

    def __init__(self, fps=30, min_bpm=40, max_bpm=180):

        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def normalize(self, signal):

        signal = np.array(signal, dtype=np.float32)

        if signal.size == 0:
            return signal

        signal = signal - np.mean(signal)

        std = np.std(signal)

        if std > 0:
            signal = signal / std

        return signal

    def bandpass_filter(self, signal):

        signal = np.array(signal, dtype=np.float32)

        if signal.size < 30:
            return signal

        nyq = 0.5 * self.fps

        low = self.min_bpm / 60.0 / nyq
        high = self.max_bpm / 60.0 / nyq

        if low <= 0 or high >= 1 or low >= high:
            return signal

        b, a = butter(3, [low, high], btype="band")

        return filtfilt(b, a, signal)

    def fft_bpm(self, signal):

        signal = self.normalize(signal)

        if signal.size == 0:
            return 0.0, np.array([]), np.array([]), 0.0

        signal = self.bandpass_filter(signal)

        fft = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal), d=1 / self.fps)

        power = np.abs(fft)

        bpm_freq = freq * 60
        mask = (bpm_freq >= self.min_bpm) & (bpm_freq <= self.max_bpm)

        if not np.any(mask):
            return 0.0, np.array([]), np.array([]), 0.0

        bpm_freq = bpm_freq[mask]
        freq = freq[mask]
        power = power[mask]

        peak = np.argmax(power)

        bpm = bpm_freq[peak]
        dominant_freq = bpm / 60.0

        return float(bpm), freq, power, float(dominant_freq)
