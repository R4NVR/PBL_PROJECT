import cv2
import mediapipe as mp
import pyautogui
import math

pyautogui.PAUSE = 0

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

frame_reduction = 80
smooth = 4

prev_x, prev_y = 0,0
clicking = False
frame_counter = 0

while True:

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame,1)

    frame_counter += 1
    if frame_counter % 2 != 0:
        cv2.imshow("Gesture Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            landmarks = []

            for id, lm in enumerate(hand):

                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                landmarks.append((x,y))

            index = landmarks[8]
            thumb = landmarks[4]

            screen_x = screen_w * index[0] / frame.shape[1]
            screen_y = screen_h * index[1] / frame.shape[0]

            curr_x = prev_x + (screen_x-prev_x)/smooth
            curr_y = prev_y + (screen_y-prev_y)/smooth

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            distance = math.hypot(
                index[0]-thumb[0],
                index[1]-thumb[1]
            )

            if distance < 25 and not clicking:
                pyautogui.click()
                clicking = True

            if distance > 40:
                clicking = False

    cv2.imshow("Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()