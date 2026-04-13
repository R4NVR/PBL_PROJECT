import cv2
import mediapipe as mp

# Use MediaPipe Tasks Vision
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model automatically if not present
model_path = "blaze_face_short_range.tflite"

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()