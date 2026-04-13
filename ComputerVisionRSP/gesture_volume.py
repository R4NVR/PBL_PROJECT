import cv2
import mediapipe as mp
import math

from pycaw.pycaw import AudioUtilities

# ----------------------------
# Volume setup (fixed section)
# ----------------------------

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# ----------------------------
# Mediapipe setup
# ----------------------------

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

# ----------------------------
# Webcam
# ----------------------------

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

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            landmarks = []

            for id, lm in enumerate(hand):

                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                landmarks.append((x, y))

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            # Thumb and index finger
            thumb = landmarks[4]
            index = landmarks[8]

            cv2.circle(frame, thumb, 10, (255,0,0), -1)
            cv2.circle(frame, index, 10, (255,0,0), -1)

            cv2.line(frame, thumb, index, (255,0,0), 3)

            # Distance calculation
            distance = math.hypot(
                index[0] - thumb[0],
                index[1] - thumb[1]
            )

            # Map distance to volume
            vol = min_vol + (distance / 200) * (max_vol - min_vol)
            volume.SetMasterVolumeLevel(vol, None)

            cv2.putText(frame,
                        f'Distance: {int(distance)}',
                        (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()