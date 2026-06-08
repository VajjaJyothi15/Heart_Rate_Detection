import cv2
import mediapipe as mp
import numpy as np


class IrisTracker:
    """
    Tracks iris / retina region using MediaPipe FaceMesh
    Landmarks 468–477
    """

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]

    def get_iris_roi(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, frame

        landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        points = []

        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:

            lm = landmarks.landmark[idx]

            points.append([
                int(lm.x * w),
                int(lm.y * h)
            ])

        points = np.array(points)

        x, y, w_box, h_box = cv2.boundingRect(points)

        iris_roi = frame[y:y + h_box, x:x + w_box]

        if iris_roi.size == 0:
            return None, frame

        # draw iris box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w_box, y + h_box),
            (255, 0, 0),
            2
        )

        return iris_roi, frame