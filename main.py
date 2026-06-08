import numpy as np
from collections import deque

from camera import Camera
from tracking.eye_tracker import EyeTracker
from tracking.iris_tracker import IrisTracker
from tracking.blink_detector import BlinkDetector

from processing.signal_processing import SignalProcessor
from processing.rf_model import RFHeartRateModel
from processing.fuzzy_logic import FuzzyQualitySystem

from gui import MedicalGUI
from processing.heart_rate import smooth_bpm


class HeartApp:

    def __init__(self):

        self.camera = Camera()
        self.eye = EyeTracker()
        self.iris = IrisTracker()
        self.blink = BlinkDetector()

        self.signal_processor = SignalProcessor()
        self.rf = RFHeartRateModel()
        self.fuzzy = FuzzyQualitySystem()

        self.buffer = deque(maxlen=300)
        self.blink_buffer = deque(maxlen=300)

        self.bpm = 0.0
        self.freq = 0.0
        self.confidence = 0.0
        self.last_power = []

    def process_frame(self, frame):

        roi, landmarks, frame = self.eye.process_frame(frame)

        h, w, _ = frame.shape

        blinked, ear = self.blink.detect(landmarks, w, h)

        _, frame = self.iris.get_iris_roi(frame)

        signal_value = self.eye.extract_green_signal(roi)

        if signal_value is not None:
            self.buffer.append(signal_value)
        else:
            self.buffer.append(0)

        self.blink_buffer.append(1 if blinked else 0)

        return frame, roi, signal_value

    def compute_metrics(self):

        if len(self.buffer) < 60:
            return

        signal = np.array(self.buffer)

        # FFT BPM
        self.bpm_fft, freqs, power, self.freq = self.signal_processor.fft_bpm(signal)
        self.last_power = power

        # RF features
        features = np.array([
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            np.max(signal),
            np.min(signal)
        ])

        rf_bpm = self.rf.predict(features)

        # Fuzzy confidence
        self.confidence = self.fuzzy.evaluate(signal)

        # Fusion model
        fused_bpm = (0.7 * self.bpm_fft) + (0.3 * rf_bpm)

        # Confidence weighting keeps low-quality signals closer to FFT BPM.
        confidence_factor = self.confidence / 100.0
        raw_bpm = (
            fused_bpm * confidence_factor +
            self.bpm_fft * (1 - confidence_factor)
        )

        if not hasattr(self, "bpm_history"):
            self.bpm_history = deque(maxlen=10)

        self.bpm_history.append(raw_bpm)

        stable_bpm = np.mean(self.bpm_history)

        self.bpm = smooth_bpm(stable_bpm)

    def get_frame(self):

        frame = self.camera.read()

        if frame is None:
            return None, None, None

        processed_frame, roi, _ = self.process_frame(frame)

        self.compute_metrics()

        return processed_frame, roi, {
            "bpm": self.bpm,
            "freq": self.freq,
            "confidence": self.confidence
        }


if __name__ == "__main__":

    app = HeartApp()

    gui = MedicalGUI(app)

    gui.run()
