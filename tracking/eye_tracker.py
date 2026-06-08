import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Extracts:
    - Eye ROI
    - Face landmarks
    - Iris (retina) region
    """

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, frame

        landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        points = []

        for idx in self.LEFT_EYE + self.RIGHT_EYE:

            lm = landmarks.landmark[idx]

            points.append([
                int(lm.x * w),
                int(lm.y * h)
            ])

        points = np.array(points)

        x, y, w_box, h_box = cv2.boundingRect(points)

        roi = frame[y:y + h_box, x:x + w_box]

        if roi.size == 0:
            return None, landmarks, frame

        cv2.rectangle(
            frame,
            (x, y),
            (x + w_box, y + h_box),
            (0, 255, 0),
            2
        )

        return roi, landmarks, frame

    def extract_green_signal(self, roi):
        if roi is None:
            return None

        # reduce noise first
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # split channels
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)

        # CHROM-style enhancement (important for rPPG stability)
        signal = g - (0.5 * r)

        # final ROI signal value
        return float(np.mean(signal))