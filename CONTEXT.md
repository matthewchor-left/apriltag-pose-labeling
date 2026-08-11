# Object AprilTag

AprilTag-based pose estimation and live annotation for a tracked object.

## Language

**Marker Model**:
JSON file describing AprilTag sticker footprint positions on the object (`marker_model.json`).
_Avoid_: marker layout, layout file

**Object Model**:
JSON file describing object skeleton keypoints and bone edges (`object_model.json`).
_Avoid_: skeleton file, object skeleton JSON (when meaning the file)

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
The 3D coordinate system used in marker_model and eraser_model: +X right, +Y down, +Z into the scene, units in meters.
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

**Marker Model Diagram**:
Static plot of marker footprints in model coordinates (`object-calibrate-marker-model`).
_Avoid_: layout diagram, layout visualization
