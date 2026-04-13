import cv2
import mediapipe as mp

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model file
model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:

        for hand in detection_result.hand_landmarks:

            landmarks = []

            for id, landmark in enumerate(hand):

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                landmarks.append((x, y))

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            # Finger counting
            fingers = 0

            # Thumb
            if landmarks[4][0] > landmarks[3][0]:
                fingers += 1

            # Other fingers
            finger_tips = [8, 12, 16, 20]

            for tip in finger_tips:
                if landmarks[tip][1] < landmarks[tip-2][1]:
                    fingers += 1

            cv2.putText(frame,
                        f'Fingers: {fingers}',
                        (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()