# Paddle AprilTag

AprilTag-based pose estimation and live annotation for a table tennis paddle.

## Language

**Reference Marker Center**:
The geometric center of the reference marker's four footprint corners (marker 0 by default). Eraser plane corners are offsets from this point.
_Avoid_: paddle reference origin, marker origin (when meaning bottom-edge midpoint)

**Eraser Plane**:
A quadrilateral region on the paddle, defined by four named corners (top_left, top_right, bottom_right, bottom_left), used to mask AprilTag stickers from the camera image.
_Avoid_: mask polygon, erase region

**Layout Frame**:
The 3D coordinate system used to author marker footprints and eraser planes: +X right, +Y down, +Z into the scene, units in meters.
_Avoid_: OpenCV frame, camera frame, paddle pose frame

**Eraser Model**:
A calibration file listing eraser planes for one paddle, stored separately from marker layout and paddle skeleton.
_Avoid_: mask config, annotation layout

**Eraser Plane Corner**:
A `[dx, dy, dz]` offset from the Reference Marker Center, expressed in the Layout Frame.
_Avoid_: layout coordinate, absolute corner

**Fused Paddle Pose**:
The paddle origin and rotation in the camera frame, estimated by combining all visible marker detections.
_Avoid_: fused pose, paddle pose (when meaning a single marker solve)

**Background Plate**:
A full-frame camera image captured without the paddle, pasted wherever eraser planes project onto the current frame.
_Avoid_: background image, clean plate

**Eraser Mask**:
The union of all projected eraser planes on the current frame, clipped to the image bounds.
_Avoid_: hull, bounding box

**Layout Bounds Hull**:
(Deprecated) Convex hull of the marker layout axis-aligned bounding box; replaced by the Eraser Model.
_Avoid_: bounds mask, convex hull eraser
