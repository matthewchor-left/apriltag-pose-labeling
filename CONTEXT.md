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
A named 3D point in an Object Model, expressed as absolute coordinates in the Model Frame (same convention as `marker_model.json` corner positions).
_Avoid_: skeleton node, landmark entry

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
A named 3D point defining an eraser quadrilateral, expressed as absolute coordinates in the Model Frame.
_Avoid_: layout coordinate, absolute corner

**Fused Object Pose**:
The reference-marker-centered object origin and rotation in the camera frame, estimated from all visible marker detections via global layout PnP.
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
The canonical 3D coordinate convention declared in a Board Model JSON (`reference_frame` block). Origin at the outer top-left board corner; +X right when viewing the board upright; +Y out of the printed face toward a front-facing camera; +Z down on the printed board; right-handed; units in meters.
_Avoid_: board frame, ChArUco frame, OpenCV board frame

**Board Model**:
JSON file describing a ChArUco board geometry and its Board Reference Frame convention (`board_model.json`). Used by `object-charuco` camera calibration.
_Avoid_: board config, charuco layout file

**Marker Model Diagram**:
Static plot of marker footprints in model coordinates (`object-inspect-marker-model`).
_Avoid_: layout diagram, layout visualization

**Setup**:
One physical marker and object sticker arrangement. If markers are moved or replaced on the object, that starts a new Setup.
_Avoid_: experiment folder, capture session (when meaning physical layout)

**Calibration Workspace**:
One calibration identity pairing a source capture with a material Calibration Recipe and its resulting models and diagnostics. A different source capture or materially different recipe belongs to a different Calibration Workspace.
_Avoid_: calibration run folder, output directory (when meaning the recipe-owned workspace)

**Calibration Recipe**:
The declared inputs, marker inventory, execution mode, solver policy, and object-keypoint derivation for one Calibration Workspace. It configures Marker Model Calibration and Object Model derivation; it is not itself an Object Model.
_Avoid_: marker config, calibration settings file

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

**Anchor Core**:
A small explicit subset of markers (including the reference marker) used to bootstrap IPPE assignment with `2^k` exhaustive search, then expand the remaining markers hierarchically from conditioned pose hypotheses before final corner bundle adjustment.
_Avoid_: anchor set, bootstrap markers

**CAD Landmark**:
A named meshless node in a CAD GLB file whose world-space position is extracted for geometry comparison.
_Avoid_: CAD node, mesh landmark

**Marker-Derived Landmark**:
A 3D point computed from `keypoint_sources` and a candidate marker layout at evaluation time, never taken from persisted `object_model.keypoints`.
_Avoid_: persisted keypoint, stored landmark

**CAD Disagreement**:
Rigid-fit and leave-one-marker-out disagreement between CAD landmarks and marker-derived landmarks in millimeters. Combines nominal CAD geometry, physical installation, export/padding choices, and vision-calibration effects; it is not calibration error because physical installation is unsurveyed.
_Avoid_: calibration error, CAD accuracy

**Detection Consistency**:
Held-out moving-video metric that solves pose from visible markers excluding the held-out marker, projects its footprint, and scores corner error in pixels. Uses frozen detections shared across candidates.
_Avoid_: reprojection error, in-sample residual

**Evaluation Candidate**:
One marker-model layout under test, declared in an evaluation manifest with `name`, `marker_model`, `capture_session`, `solver_variant`, and calibration-source provenance.
_Avoid_: calibration run, layout variant
