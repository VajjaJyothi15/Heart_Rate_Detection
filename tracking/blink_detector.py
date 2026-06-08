import numpy as np
import mediapipe as mp


class BlinkDetector:
    """
    Eye blink detection using EAR (Eye Aspect Ratio)
    """

    def __init__(self):

        self.blink_count = 0
        self.closed_frames = 0

        self.EAR_THRESHOLD = 0.22
        self.MIN_FRAMES = 2

    def _distance(self, p1, p2):

        return np.linalg.norm(
            np.array(p1) - np.array(p2)
        )

    def _eye_aspect_ratio(self, eye):

        # eye = 6 points

        v1 = self._distance(eye[1], eye[5])
        v2 = self._distance(eye[2], eye[4])
        h = self._distance(eye[0], eye[3])

        return (v1 + v2) / (2.0 * h + 1e-6)

    def detect(self, landmarks, w, h):

        if landmarks is None:
            return False, 0.0

        LEFT = [33, 160, 158, 133, 153, 144]
        RIGHT = [362, 385, 387, 263, 373, 380]

        left_eye = []
        right_eye = []

        for idx in LEFT:
            lm = landmarks.landmark[idx]
            left_eye.append((lm.x * w, lm.y * h))

        for idx in RIGHT:
            lm = landmarks.landmark[idx]
            right_eye.append((lm.x * w, lm.y * h))

        ear_left = self._eye_aspect_ratio(left_eye)
        ear_right = self._eye_aspect_ratio(right_eye)

        ear = (ear_left + ear_right) / 2.0

        blink = False

        if ear < self.EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.MIN_FRAMES:
                self.blink_count += 1
                blink = True

            self.closed_frames = 0

        return blink, ear