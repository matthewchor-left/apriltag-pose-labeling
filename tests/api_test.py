import cv2
from paddle_apriltag import PaddleDetector
from paddle_apriltag.calibration import load_intrinsics

camera_matrix, dist_coeffs, _, _, _ = load_intrinsics("calibration/camera_calibration.json")
detector = PaddleDetector(
    camera_matrix,
    dist_coeffs,
    marker_layout="calibration/marker_layout.json",
)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    pose = detector.detect(frame)  # PaddlePose(origin, rotation) or None
    if pose is not None:
        print(pose)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()