import cv2
import mediapipe as mp
import os
import time

# Mediapipe setup
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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_action = 0
cooldown = 2   # seconds

while True:

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            landmarks = []

            for lm in hand:

                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                landmarks.append((x,y))

            # -------- Finger Detection --------

            fingers = 0

            if landmarks[8][1] < landmarks[6][1]:
                fingers += 1

            if landmarks[12][1] < landmarks[10][1]:
                fingers += 1

            if landmarks[16][1] < landmarks[14][1]:
                fingers += 1

            if landmarks[20][1] < landmarks[18][1]:
                fingers += 1

            # -------- Gesture Actions --------

            current_time = time.time()

            if current_time - last_action > cooldown:

                if fingers == 1:
                    os.startfile("spotify")  # opens Spotify
                    last_action = current_time

                elif fingers == 2:
                    os.startfile("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
                    last_action = current_time

                elif fingers == 3:
                    print("Stopping program")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()