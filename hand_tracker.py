import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:

    def __init__(self, model_path = "hand_landmarker.task", num_hands = 1):
        base_options = python.BaseOptions(model_asset_path= model_path)
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = num_hands,
            min_hand_detection_confidence = 0.3,
            min_hand_presence_confidence = 0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

        if not self.cap.isOpened():
            raise RuntimeError("Error can't open Webcam")

    def get_frame_and_landmarks(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
    
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

        results = self.detector.detect(rgb_image)
        return frame, results

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
    

