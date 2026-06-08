import cv2
import numpy as np


class BlinkPlot:

    def draw(self, canvas, blink_signal):

        if len(blink_signal) < 5:
            return canvas

        signal = np.array(blink_signal)

        h, w, _ = canvas.shape

        for i in range(1, len(signal)):

            x1 = int((i - 1) * w / len(signal))
            x2 = int(i * w / len(signal))

            y1 = h - int(signal[i - 1] * h * 0.8)
            y2 = h - int(signal[i] * h * 0.8)

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            canvas,
            "Blink Signal",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return canvas