# Object AprilTag

AprilTag-based pose estimation and live annotation for a tracked object.

## Language

**Marker Model**:
JSON file describing AprilTag sticker footprint positions on the object (`marker_model.json`).
_Avoid_: marker layout, layout file

**Object Model**:
JSON file describing object skeleton keypoints and bone edges (`object_model.json`). Keypoint positions use the Model Frame (`coordinate_frame: "marker_model"`), the same frame as Marker/Eraser geometry.
_Avoid_: skeleton file, object skeleton JSON (when meaning the file)

**Object Model Keypoint**:
A named 3D point in an Object Model, expressed in the Model Frame relative to the Reference Marker Center.
_Avoid_: skeleton node, landmark entry

**Interactive Object Model Capture**:
Live `object-detect` session that records Object Model keypoints from Board Coordinate terminal input while `--preview`, `--board-frame`, and `--overlay-object-model` are all enabled. Requires a current Fused Object Pose and applicable Board Pose Estimate; edits update the in-memory overlay immediately and persist only on explicit save.
_Avoid_: live annotation, skeleton editor

**Eraser Model**:
JSON file listing eraser planes for masking tags from the camera image (`eraser_model.json`).
_Avoid_: mask config, annotation layout

**Reference Marker Center**:
The geometric center of the reference marker's four footprint corners (marker 0 by default).
_Avoid_: object reference origin, marker origin (when meaning bottom-edge midpoint)

**Eraser Plane**:
A quadrilateral region on the object, defined by four named corners, used to mask AprilTag stickers from the camera image.
_Avoid_: mask polygon, erase region

**Model Frame**:
The 3D coordinate system used in Marker Model, Eraser Model, and Object Model: +X right, +Y down, +Z into the scene, units in meters.
_Avoid_: layout frame, OpenCV frame, camera frame

**Eraser Plane Corner**:
A `[dx, dy, dz]` offset from the Reference Marker Center, expressed in the Model Frame.
_Avoid_: layout coordinate, absolute corner

**Fused Object Pose**:
The object origin and rotation in the camera frame, estimated by combining all visible marker detections.
_Avoid_: fused pose, object pose (when meaning a single marker solve)

**Background Plate**:
A full-frame camera image captured without the object, pasted wherever eraser planes project onto the current frame.
_Avoid_: background image, clean plate

**Detection Outline**:
AprilTag corner box and ID drawn on the live camera frame.
_Avoid_: marker annotation, overlay, visualization

**Pose Projection**:
3D object geometry drawn on the live camera frame from the fused pose.
_Avoid_: overlay, visualize, draw pose

**Skeleton Chart**:
Separate matplotlib 3D plot of object_model keypoints, not drawn on the camera frame.
_Avoid_: plot graph, 3D plot

**Board Reference Frame**:
The canonical 3D coordinate system attached to a printed ChArUco board. Origin at the outer top-left board corner; +X right when viewing the board upright; +Y out of the printed face toward a front-facing camera; +Z down on the printed board; right-handed; units in meters.
_Avoid_: board frame, ChArUco frame, OpenCV board frame

**Board Pose Estimate**:
A camera-relative estimate of the Board Reference Frame origin and orientation from observed ChArUco intersections. This is an estimate, not the convention itself.
_Avoid_: board pose, board transform (when meaning the estimate)

**Board Coordinate**:
A 3D point expressed in the Board Reference Frame. Live overlays annotate active model points with Board Coordinates when board tracking and a pose projection overlay are both enabled.
_Avoid_: board position, ChArUco coordinate, board pixel location

**Board Model**:
JSON file describing a ChArUco board geometry and its Board Reference Frame convention (`board_model.json`).
_Avoid_: board config, charuco layout file

**Marker Model Diagram**:
Static plot of marker footprints in model coordinates (`object-inspect-marker-model`).
_Avoid_: layout diagram, layout visualization

**Marker Model Calibration**:
Live estimation of marker sticker layout on an object from co-visible AprilTag detections, producing `marker_model.json`.
_Avoid_: marker layout calibration session, live layout capture

**Co-visibility Observation**:
One timed camera sample where at least two expected markers are simultaneously visible, recording their image corners for layout calibration.
_Avoid_: calibration frame, capture sample

**Relative Marker Transform**:
The rigid transform from one marker frame to another, estimated from paired detections across co-visible observations.
_Avoid_: marker-to-marker pose, inter-marker offset

**Marker Pose Graph**:
The graph of expected markers connected by relative transforms with sufficient co-visibility support, anchored at the reference marker.
_Avoid_: marker graph, connectivity graph
