import cv2
from object_apriltag import ObjectDetector
from object_apriltag.calibration import load_intrinsics

camera_matrix, dist_coeffs, _, _, _ = load_intrinsics("config/Camera/nexplaygroundcam/intrinsics.json")
detector = ObjectDetector(
    camera_matrix,
    dist_coeffs,
    marker_model="config/Model/object_01/marker_model.json",
)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    pose = detector.detect(frame)  # ObjectPose(origin, rotation) or None
    if pose is not None:
        print(pose)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()