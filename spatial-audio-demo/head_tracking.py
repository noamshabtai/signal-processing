import pathlib
import threading
import time
import urllib.request

import cv2
import mediapipe
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
STOP_TIMEOUT = 2.0


def cameras():
    return sorted(int(path.name[len("video") :]) for path in pathlib.Path("/dev").glob("video[0-9]*"))


def fetch_model(path):
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310
    return path


def yaw_pitch_roll(transformation):
    rotation = np.asarray(transformation)[:3, :3]
    yaw = np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
    pitch = np.rad2deg(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    roll = np.rad2deg(np.arctan2(rotation[2, 1], rotation[2, 2]))
    return np.array([yaw, pitch, roll])


class HeadTracking:
    def __init__(self, **kwargs):
        self.camera = kwargs.get("camera", 0)
        self.model_path = kwargs.get("model_path", "face_landmarker.task")
        self.smoothing = kwargs.get("smoothing", 0.6)
        self.orientation = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.model_path = fetch_model(self.model_path)
        self.running = True
        self.thread = threading.Thread(target=self.track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=STOP_TIMEOUT)
            self.thread = None
        with self.lock:
            self.orientation = None

    def read(self):
        with self.lock:
            return self.orientation

    def options(self):
        return mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )

    def smooth(self, angles):
        with self.lock:
            previous = self.orientation
            self.orientation = angles if previous is None else self.smoothing * previous + (1 - self.smoothing) * angles

    def track(self):
        capture = cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        try:
            with mediapipe.tasks.vision.FaceLandmarker.create_from_options(self.options()) as landmarker:
                start = time.monotonic()
                while self.running:
                    grabbed, frame = capture.read()
                    if not grabbed:
                        continue
                    image = mediapipe.Image(
                        image_format=mediapipe.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    result = landmarker.detect_for_video(image, int((time.monotonic() - start) * 1000))
                    if result.facial_transformation_matrixes:
                        self.smooth(yaw_pitch_roll(result.facial_transformation_matrixes[0]))
        finally:
            capture.release()
