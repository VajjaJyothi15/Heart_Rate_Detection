import cv2
import numpy as np


class SignalPlot:

    def draw(self, canvas, signal):

        if len(signal) < 5:
            return canvas

        signal = np.array(signal, dtype=np.float32)

        signal = signal - np.min(signal)
        signal = signal / (np.max(signal) + 1e-6)

        h, w, _ = canvas.shape

        pts = []

        for i, v in enumerate(signal):

            x = int(i * w / len(signal))
            y = int(h - v * h)

            pts.append((x, y))

        for i in range(1, len(pts)):

            cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 0), 2)

        cv2.putText(
            canvas,
            "rPPG Signal",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return canvas