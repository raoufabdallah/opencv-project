import cv2
import time
import numpy as np
import pyautogui
from hand_tracker import HandTracker

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True
screen_w, screen_h = pyautogui.size()
padding = 150

tracker = HandTracker(num_hands = 1)
print("Airmouse started pressq to quit")

window_name = "Airmouse tester"
click_cooldown = 0

while True:
    frame, results = tracker.get_frame_and_landmarks()

    if frame is None:
        print("Error can't get frame")
        break

    if results.hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = results.hand_landmarks[0]

        for landmark in hand_landmarks:
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        screen_x = np.interp(ix, (padding, w - padding), (0, screen_w))
        screen_y = np.interp(iy, (padding, h - padding), (0, screen_h))
        pyautogui.moveTo(screen_x, screen_y)

        distance = np.hypot(ix - tx, iy - ty)
        if distance < 35 and (time.time() - click_cooldown) > 0.3:
            pyautogui.click()
            click_cooldown = time.time()


    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


tracker.release()