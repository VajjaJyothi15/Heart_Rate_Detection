import cv2


class Camera:
    """
    Webcam manager.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ) -> None:

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            fps
        )

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def is_opened(self) -> bool:

        return self.cap.isOpened()

    def release(self) -> None:

        if self.cap is not None:

            self.cap.release()