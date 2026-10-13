import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(
    model_asset_path = "hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2,
    min_hand_detection_confidence = 0.3,
    min_hand_presence_confidence = 0.3,
)
detector = vision.HandLandmarker.create_from_options(options)

if not cap.isOpened():
    print("Error can't open camera")
    exit()

print("Camera started. Press q to quit")

window_name = "Camera Feed"

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error can't recieve frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)


    results = detector.detect(rgb_image)

    if results.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                cx = int(landmark.x*w)
                cy = int(landmark.y*h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)



    cv2.imshow(window_name, frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()