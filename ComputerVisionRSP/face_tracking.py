import cv2
import mediapipe as mp

# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, frame = cap.read()
    if not success:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    h, w, _ = frame.shape
    center_screen = (w//2, h//2)

    cv2.circle(frame, center_screen, 10, (0,255,0), -1)

    if result.detections:

        for detection in result.detections:

            bbox = detection.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            w_box = bbox.width
            h_box = bbox.height

            face_center = (x + w_box//2, y + h_box//2)

            cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),(255,0,0),2)
            cv2.circle(frame, face_center, 8, (0,0,255), -1)

            dx = face_center[0] - center_screen[0]
            dy = face_center[1] - center_screen[1]

            cv2.putText(frame,
                        f"Offset X:{dx} Y:{dy}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

    cv2.imshow("AI Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()