import cv2
import numpy as np


class FFTPlot:

    def draw(self, canvas, freqs, power):

        if len(freqs) == 0:
            return canvas

        bpm = freqs * 60

        mask = (bpm >= 40) & (bpm <= 180)

        bpm = bpm[mask]
        power = power[mask]

        if len(power) < 5:
            return canvas

        power = power / (np.max(power) + 1e-6)

        h, w, _ = canvas.shape

        for i in range(1, len(power)):

            x1 = int((i - 1) * w / len(power))
            x2 = int(i * w / len(power))

            y1 = int(h - power[i - 1] * h)
            y2 = int(h - power[i] * h)

            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.putText(
            canvas,
            "FFT Spectrum",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return canvas