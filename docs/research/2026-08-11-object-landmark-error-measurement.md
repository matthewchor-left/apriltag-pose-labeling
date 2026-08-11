# Measuring final object-landmark error and separating its causes

Date: 2026-08-11

## Decision summary

The primary result should be **held-out, metric landmark error against an independent, synchronized reference**, not in-sample tag reprojection error. For each declared object landmark, report 3D Euclidean error and camera-axis components; also report object-pose translation and geodesic rotation error, detection coverage, and latency. Stratify all results by distance, view angle, image position, marker ID/count, motion, and hardware state.

Run the evaluation in stages and freeze raw frames/corners so one factor can be substituted at a time:

1. quantify repeatability and tag-corner observation error;
2. validate camera intrinsics/distortion on held-out observations;
3. validate the marker-to-object geometry with independent metrology and cross-marker consistency;
4. test PnP and frame-transform choices on frozen correspondences;
5. run end-to-end static and dynamic landmark tests against independent ground truth;
6. propagate reference and fitted-parameter uncertainty into every acceptance result.

Camera calibration, marker geometry, corner localization, and pose are generally **not identifiable from the same tag images and their reprojection residuals alone**. A physical reference, held-out constraints, or controlled substitutions are required to assign error to a component. This follows from the fact that calibration and pose are jointly optimized to explain image observations, while uncertainty propagation must retain parameter covariance rather than considering each input independently ([OpenCV calibration model](https://docs.opencv.org/4.9.0/d9/d0c/group__calib3d.html), [JCGM GUM §§5.1–5.2](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/05_e.html)).

## Scope and assumptions

The measurand is the camera-frame position of each named point in `object_model.json`, for example `center`, `home`, and `back` in the current [`remote1` object model](../../config/Model/remote1/object_model.json). If another downstream measurand matters—image-space overlay position, world-frame impact point, object orientation, or velocity—it needs a separate acceptance line because low pose error in one coordinate or operating region does not imply low error in another.

Assumptions to confirm before claiming accuracy:

- the active camera profile, sensor mode, resolution, crop, focus, zoom, exposure, gain, and image-processing state are fixed and logged;
- tag size means the physical edge represented by the detector's returned corner locations, not an arbitrary outside printed edge; the upstream AprilTag project explicitly defines tag size at the detection corners/border transition ([AprilRobotics pose documentation](https://github.com/AprilRobotics/apriltag#pose-estimation));
- `marker_model.json` and `object_model.json` describe the same rigid object frame and the object does not flex appreciably;
- the reference system's camera-to-reference and object-to-reference transforms, timestamp convention, and uncertainty are known;
- test data are disjoint from all camera-, marker-, hand-eye-, and timing-calibration data.

These assumptions are testable requirements, not conveniences. NIST defines metrological traceability as a documented unbroken calibration chain in which every link contributes uncertainty, and cautions that traceability alone does not establish fitness for purpose ([NIST traceability policy](https://www.nist.gov/calibrations/traceability)).

## What the repository currently computes

The current path is:

1. [`ObjectDetector.find_markers`](../../src/object_apriltag/detector.py) calls OpenCV `ArucoDetector.detectMarkers` on grayscale frames and retains IDs in the marker model.
2. [`build_detector_parameters`](../../src/object_apriltag/apriltag.py) uses OpenCV AprilTag dictionaries; the relaxed/aggressive presets enable `CORNER_REFINE_APRILTAG`. OpenCV describes this mode as tag and corner detection based on the AprilTag 2 approach ([OpenCV ArUco API](https://docs.opencv.org/4.11.0/de/d67/group__objdetect__aruco.html)); the primary AprilTag 2 paper evaluates detection rate, false positives, and speed, but does not certify this repository's corner uncertainty ([Wang and Olson 2016](https://april.eecs.umich.edu/pdfs/wang2016iros.pdf)).
3. [`estimate_marker_pose`](../../src/object_apriltag/pose.py) calls `cv2.solvePnP(..., SOLVEPNP_IPPE)` independently for each four-corner planar tag. OpenCV specifies that IPPE requires coplanar points and that `solvePnPGeneric`—not the single-result `solvePnP` wrapper—exposes all possible IPPE solutions ([OpenCV solvePnP guide](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)).
4. [`object_pose_from_marker_pose`](../../src/object_apriltag/pose.py) applies per-marker transforms derived from [`marker_model.json`](../../config/Model/remote1/marker_model.json) and `OBJECT_AXIS_FLIP`.
5. [`estimate_fused_pose`](../../src/object_apriltag/pose.py) takes an unweighted arithmetic mean of per-marker origins and a normalized arithmetic mean of sign-aligned quaternions; it is not a joint all-corner PnP or covariance-weighted estimator.
6. [`object_world_points_from_pose`](../../src/object_apriltag/viz/skeleton.py) turns object-model landmarks into camera-frame points for display.

There is a high-priority frame-contract issue to resolve before measuring the current landmarks: [`derive_marker_to_object_transforms`](../../src/object_apriltag/layout.py) defines the estimated object origin at the reference marker's rectangle center, and [`object_world_points_from_pose`](../../src/object_apriltag/viz/skeleton.py) applies object-model keypoints directly from that origin, but the current [`remote1/object_model.json`](../../config/Model/remote1/object_model.json) declares its origin as the blade/handle junction. Unless those two physical points coincide or an omitted transform is applied elsewhere, the configured landmark offset is part of the end-to-end error. Treat the intended origin as an explicit datum to survey and test, not as settled terminology.

Camera calibration currently collects manually selected ChArUco or checkerboard views, accepts as few as three, calls `cv2.calibrateCamera` with default flags, and writes a five-coefficient default calibration plus a mean pointwise Euclidean reprojection error ([`cli/charuco.py`](../../src/object_apriltag/cli/charuco.py)). OpenCV's extended calibration API can additionally return intrinsic/extrinsic standard deviations and per-view RMS errors ([OpenCV `calibrateCameraExtended`](https://docs.opencv.org/4.9.0/d9/d0c/group__calib3d.html)).

The live HUD's [`marker_reprojection_error`](../../src/object_apriltag/pose.py) estimates a fresh pose from a tag's four corners and projects the same four points back. That is an **in-sample fit residual**. It can expose gross failures, but it is not an independent estimate of object-landmark error, camera-calibration bias, marker-model error, or even held-out corner error. Planar IPPE can also have two plausible poses when the projection is close to affine, and reprojection error may not distinguish them ([Collins and Bartoli 2014](https://doi.org/10.1007/s11263-014-0725-5), [authors' IPPE implementation notes](https://github.com/tobycollins/IPPE#ippe)).

## Measurement model and metrics

### Frames

Use the transform notation \(^{A}T_B\): coordinates in frame \(B\) mapped into frame \(A\). For landmark \(k\), the detector produces

\[
\hat{\mathbf p}^{C}_{k,i}
=
{}^{C}\hat R_{O,i}\,\mathbf p^{O}_{k,\text{config}}
+
{}^{C}\hat{\mathbf t}_{O,i}.
\]

An external system with world/reference frame \(W\) produces

\[
\mathbf p^{C,*}_{k,i}
=
{}^{C}T_W(t_i)\,
{}^{W}T_O(t_i)\,
\mathbf p^{O,*}_{k}.
\]

The camera extrinsic \(^{C}T_W\), object rigid-body attachment \(^{W}T_O\), physical landmark \(\mathbf p^{O,*}_k\), and time \(t_i\) must be independently calibrated or measured. Otherwise their errors are silently included in the apparent detector error. Time-synchronized external pose has precedent as a vision ground-truth method, but the benchmark's own reference calibration and synchronization still bound the conclusion ([Sturm et al. 2012](https://doi.org/10.1109/IROS.2012.6385773)).

### Primary end-to-end metrics

For every valid frame \(i\) and landmark \(k\):

\[
\mathbf e_{k,i}=\hat{\mathbf p}^{C}_{k,i}-\mathbf p^{C,*}_{k,i},
\qquad
e_{k,i}=\|\mathbf e_{k,i}\|_2.
\]

Report, without dropping failures:

- landmark \(e\): median, RMSE, 95th and 99th percentiles, maximum, and bootstrap confidence intervals;
- signed \(e_x,e_y,e_z\): mean bias and standard deviation, because depth and lateral failure modes differ;
- pose translation \( \|\hat{\mathbf t}-\mathbf t^*\|_2 \);
- pose rotation \( \theta=\cos^{-1}(\operatorname{clip}((\operatorname{tr}(R^{*T}\hat R)-1)/2,-1,1)) \);
- 2D landmark error against independently annotated/measured image points when image overlay is a deliverable;
- detection probability, false-ID rate, pose-valid probability, and conditional error **plus** unconditional failure rate;
- dynamic latency and timestamp error, reported separately from geometric error.

RMSE makes large misses visible while percentiles state service levels; signed components reveal systematic bias that a scalar norm hides. Confidence intervals should resample at the independent unit—usually capture session or trajectory, not individual adjacent video frames—because adjacent frames are temporally correlated. The distinction between systematic bias and variance, and the failure of idealized covariance estimates under non-ideal calibration data, is demonstrated by Hagemann et al. ([peer-reviewed primary study](https://doi.org/10.1007/s11263-021-01528-x)).

Always stratify by:

- distance and projected tag side length in pixels;
- incidence/view angle and whether a pose is near the planar ambiguity regime;
- image radius/quadrant;
- marker ID, visible marker count, and marker combination;
- static versus translational/rotational speed;
- exposure, gain, focus, blur score, frame rate, temperature/session, and lighting;
- detector sensitivity preset and corner-refinement mode.

### Observation/corner metrics

For independently referenced image corner \(\mathbf u^*_{m,j,i}\) and detector corner \(\hat{\mathbf u}_{m,j,i}\):

\[
\mathbf r^{\text{corner}}_{m,j,i}
=
\hat{\mathbf u}_{m,j,i}-\mathbf u^*_{m,j,i}.
\]

Report signed 2D bias, radial/tangential components relative to the principal point, covariance ellipses, RMSE/percentiles, and miss/false-ID rates by tag size, angle, blur, contrast, and image location. Suitable references are (a) a much higher-resolution, independently calibrated still camera transformed into the test image, (b) manually or line-fit corner labels with inter-annotator uncertainty, or (c) projected corners from surveyed geometry plus independently measured camera/object pose. Do not define \(\mathbf u^*\) with the same detected corners or the same fitted PnP pose.

The AprilTag family coding primarily protects identity detection; corner position still depends on quad/edge localization. The original AprilTag paper presents the detector and its benchmark methodology ([Olson 2011](https://doi.org/10.1109/ICRA.2011.5979561)), while the AprilTag source describes edge refinement as snapping quad edges to nearby gradients ([official source header](https://github.com/AprilRobotics/apriltag/blob/master/apriltag.h)). Neither source supplies a universal corner-error number for this camera, print, optics, OpenCV port, or preset, so it must be measured here.

### Uncertainty on a landmark result

Collect uncertain inputs into

\[
\mathbf x =
[\mathbf u,\ \boldsymbol\kappa,\ \boldsymbol\theta_{\text{marker}},\
\boldsymbol\xi_{\text{pose}},\ \mathbf p_k^O,\ \boldsymbol\xi_{\text{reference}},\
\Delta t,\ldots].
\]

For \(\mathbf y=f(\mathbf x)\), first-order propagation is

\[
\Sigma_{\mathbf y}\approx J\Sigma_{\mathbf x}J^\mathsf T,
\qquad
J=\left.\frac{\partial f}{\partial\mathbf x}\right|_{\hat{\mathbf x}}.
\]

This is the correlated-input form of the law of propagation of uncertainty; covariance terms must be retained ([JCGM 100:2008 §§5.1–5.2](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/05_e.html)). OpenCV `projectPoints` can return derivatives with respect to rotation, translation, focal lengths, principal point, and distortion coefficients, which are useful for image-plane propagation ([OpenCV source/API contract](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp#L855-L881)).

Use session/trajectory bootstrap or Monte Carlo propagation when residuals are heteroscedastic, the PnP solution is multimodal, the planar ambiguity is active, or linearization is poor. Report the reference uncertainty separately and, when independence is justified, the detector contribution as \(u_\text{detector}\approx\sqrt{\max(0,u_\text{observed}^2-u_\text{reference}^2)}\); a negative radicand means the experiment cannot resolve detector error at that scale, not that detector uncertainty is zero. The GUM also requires stating any coverage factor used for expanded uncertainty ([JCGM 100:2008](https://doi.org/10.59161/JCGM100-2008E)).

If a complete landmark-error covariance \(\Sigma_{e,k}\) is available in the same frame, also check the normalized squared residual

\[
d^2_{M,k}=\mathbf e_k^\mathsf T\Sigma_{e,k}^{-1}\mathbf e_k .
\]

This Mahalanobis check is valid only when the covariance includes detector, calibration, marker/object-model, transform, timing, and reference terms (including correlations) and is nonsingular or handled with a justified subspace/pseudoinverse. Otherwise it mainly diagnoses an incomplete uncertainty model, not a specific physical error source.

## Error-source isolation

### 1. Tag observation and corner localization

Freeze raw frames and detector outputs. Compare:

- OpenCV `default`, `relaxed`, and `aggressive`;
- corner refinement none versus `CORNER_REFINE_APRILTAG`;
- optional independent line-intersection corner estimates;
- original corners versus synthetically perturbed corners with known covariance.

Then run the *same* intrinsics, marker geometry, PnP, and fusion on each corner set. A change in final landmark error is attributable to the changed observation path only within that frozen dataset. The AprilTag 2 paper supports testing detection over image scale and degradation, but its reported benchmark does not replace camera-specific validation ([Wang and Olson 2016](https://april.eecs.umich.edu/pdfs/wang2016iros.pdf)).

Valid conclusion: “On this held-out image distribution, replacing corner set A with B changed 95th-percentile landmark error by X while all downstream inputs were fixed.”

Invalid conclusion: “A low same-corner PnP reprojection error proves the detector corners are accurate.”

### 2. Intrinsics and distortion

Acquire a richer set of calibration views spanning the full sensor, distance, tilt, and operating focus/zoom. Calibrate on a training subset and evaluate on held-out sessions/views. Record the returned RMS, per-view residual fields, parameter covariance or resampled distributions, and stability across independently collected calibration sets. Zhang's primary method requires multiple orientations of a planar target and performs nonlinear refinement under a camera/distortion model ([Zhang 2000](https://doi.org/10.1109/34.888718)); Hagemann et al. show that low residuals can coexist with systematic model bias or underestimated uncertainty ([Hagemann et al. 2022](https://doi.org/10.1007/s11263-021-01528-x)).

Measure the printed board with traceable tools and check flatness. The current PNG generation correctly warns to print at 100%, but print scale, anisotropy, mounting, and deformation remain physical inputs. OpenCV explicitly warns that off-the-shelf printed boards can be inaccurate and documents `calibrateCameraRO` for refining roughly planar object points when fixed distances are accurately measured ([OpenCV calibration tutorial](https://docs.opencv.org/4.12.0/d4/d94/tutorial_camera_calibration.html)); dynamic target deformation can also bias calibration ([Hagemann et al., WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Hagemann_Modeling_Dynamic_Target_Deformation_in_Camera_Calibration_WACV_2022_paper.html)).

Do not rank calibrations by training reprojection error alone. Torsello et al. experimentally show that an inaccurate target can yield low reprojection error while producing a poor calibration because fitted camera/pose parameters absorb target error ([Torsello et al. 2010](https://www.dsi.unive.it/~atorsell/papers/Conferences/bmvc2010.pdf)).

Isolation experiment:

1. save calibration A and independently acquired calibration B;
2. save one frozen end-to-end test set with external ground truth;
3. replay identical detector corners/model/PnP with A and B;
4. compare landmark error by image radius, range, and angle;
5. optionally compare a higher-grade reference calibration C.

This estimates sensitivity to the candidate calibration. It identifies “A performs better than B on this held-out set,” but without reference C or physical metrology it cannot prove the remaining error belongs only to intrinsics.

### 3. Marker-to-object and landmark geometry

Survey all tag detection corners and final landmarks in one physical object frame, preferably with a coordinate measuring machine, articulated arm, photogrammetric system, or jig whose uncertainty is small relative to the acceptance target. Record repeated setup measurements and the uncertainty/covariance of shared datums. The current loader enforces tag edge length consistency but cannot establish that configured coordinates match the physical object ([`layout.py`](../../src/object_apriltag/layout.py), [`remote1/marker_model.json`](../../config/Model/remote1/marker_model.json)).

Useful tests:

- **single-marker consistency:** for a static object, estimate object pose separately from each ID; transform the same landmark and compare ID-specific biases;
- **leave-one-marker-out:** fuse all but marker \(m\), project \(m\)'s surveyed corners, and measure held-out residuals;
- **model substitution:** replay frozen corners/intrinsics/PnP with hand-measured model A and independently surveyed model B;
- **physical landmark check:** compare transformed configured keypoints to independently observed physical keypoints, not just tag corners;
- **rigidity check:** repeat surveys and static captures after handling, temperature change, and impacts.

ID-specific pose offsets that remain stable across camera positions are evidence consistent with marker-model error; image-radius-dependent residuals across all IDs are evidence consistent with intrinsics/distortion; projected-size/blur dependence is evidence consistent with observation noise. These are diagnostic signatures, not proofs, because correlated errors can mimic one another. Only controlled substitution or independent constraints support attribution.

### 4. PnP, ambiguity, transforms, and fusion

Replay identical corners, intrinsics, and model through:

- the current one-solution `SOLVEPNP_IPPE`;
- `solvePnPGeneric(..., SOLVEPNP_IPPE)` retaining both planar candidates;
- `SOLVEPNP_IPPE_SQUARE` after expressing points in its required centered ordering;
- optional all-visible-corner joint PnP/refinement;
- current unweighted pose fusion versus uncertainty/residual-weighted candidates.

OpenCV documents both IPPE variants, their coplanar constraints, multi-solution behavior, and nonlinear refinement methods ([OpenCV solvePnP guide](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)). The IPPE primary work explains that two poses can be observationally indistinguishable under near-affine projection ([Collins and Bartoli 2014](https://doi.org/10.1007/s11263-014-0725-5)). Therefore report candidate separation, reprojection ratio, positive-depth checks, and temporal/multi-marker disambiguation failures rather than silently accepting the first solution.

Verify frame algebra with independently surveyed poses:

\[
{}^C T_O = {}^C T_M\,{}^M T_O.
\]

For every marker, compare the implementation's transformed basis and origin against this equation, including the bottom-center marker origin and `OBJECT_AXIS_FLIP` in [`pose.py`](../../src/object_apriltag/pose.py) and [`layout.py`](../../src/object_apriltag/layout.py). A convention/sign/order bug can look like a geometry error and will not be repaired by lower corner noise.

### 5. Temporal, rolling-shutter, and hardware effects

Static tests separate geometry from motion-related effects. Dynamic tests must timestamp image exposure and reference pose in one clock domain or estimate offset and drift. A rolling shutter exposes rows at different times, so a single rigid pose cannot exactly explain a moving object across the whole image; the foundational rolling-shutter model assigns different times to image rows and predicts systematic projection bias under motion ([Meingast, Geyer, and Sastry 2005](https://arxiv.org/abs/cs/0503076)). A modern primary evaluation uses hardware synchronization, pre-calibrated sensor transforms, motion-capture ground truth, and explicit time alignment ([Schubert et al. 2019](https://arxiv.org/abs/1911.01015)).

Run paired conditions:

- camera and object static;
- static camera, object motion at controlled linear/angular speeds and directions;
- object static, camera motion;
- short versus long exposure at matched illumination where possible;
- autofocus/autoexposure disabled and fixed versus enabled;
- central versus vertically separated tags, which changes row-time separation under rolling shutter;
- cold start versus warmed camera and repeated sessions.

The current live CLI uses `VideoCapture.read()` without recording exposure timestamps and only requests width/height ([`cli/detect.py`](../../src/object_apriltag/cli/detect.py)); the current webcam profile records camera-control settings including autofocus ([`config/Camera/webcam/uvcc.json`](../../config/Camera/webcam/uvcc.json)). Requested settings are not evidence that the device accepted them, so read back and log actual controls, negotiated resolution/fps/fourcc, and frame timestamps.

Signatures:

- error increasing approximately with speed and reversing with motion direction suggests latency/time-offset bias;
- corner/pose shear varying with image row and angular velocity suggests rolling shutter;
- blur-linked corner variance suggests exposure/motion blur;
- session drift with focus or temperature suggests changing intrinsics;
- dropped/duplicated frames invalidate naive frame-index synchronization.

These signatures are hypotheses until isolated by synchronized controls or a validated rolling-shutter model.

## Non-identifiability and practical identifiability

The image residual has the form

\[
\mathbf r =
\mathbf u_{\text{observed}}
-
\pi(\mathbf P_{\text{object}};\,
\boldsymbol\kappa_{\text{camera}},
\boldsymbol\xi_{\text{pose}},
\boldsymbol\theta_{\text{marker}}).
\]

Changing focal length can be partly compensated by depth; changing distortion can be absorbed by per-view pose; changing tag scale changes metric translation scale; and changing a marker-to-object transform can be absorbed by that marker's pose. Since OpenCV calibration minimizes total squared reprojection distance over camera parameters and per-view poses, low training residual alone does not resolve these compensations ([OpenCV calibration algorithm](https://docs.opencv.org/4.9.0/d9/d0c/group__calib3d.html)).

Practical identifiability checks:

1. build the residual Jacobian with respect to intrinsics, each marker transform, object landmarks, camera/object extrinsics, and timing;
2. inspect singular values, condition number, and high parameter correlations;
3. profile one parameter/group while re-optimizing nuisance parameters;
4. bootstrap whole calibration views and whole test sessions;
5. add experiments until weak directions are constrained: varied range/tilt/image position, multiple non-coplanar marker faces, independently measured lengths, static/dynamic reversals, and external pose;
6. keep one component fixed to an independent measurement before estimating another.

A rank-deficient or ill-conditioned Jacobian means the available experiment cannot distinguish some parameter combinations. First-order covariance is informative near one well-behaved mode; resampling and explicit alternate PnP modes are required when the estimator is biased, multimodal, or nonlinear ([JCGM GUM](https://doi.org/10.59161/JCGM100-2008E), [Hagemann et al.](https://doi.org/10.1007/s11263-021-01528-x)).

For a joint multi-view analysis, use nested parameter blocks: (A) fixed camera and surveyed object model; (B) free camera, fixed object; (C) fixed camera, free marker transforms; (D) both free; and optionally per-session camera or per-marker bias effects. Compare held-out likelihood/error, residual fields, and parameter stability—not training fit alone. With enough crossed replication, camera session, marker ID, pose, and repeat can be treated as variance components; without that replication their effects remain confounded. Bundle adjustment remains self-consistency unless surveyed scale/geometry or external poses anchor it.

## Staged protocol

### Stage 0 — freeze the specification

Write down:

- landmark names and physical definitions;
- object/camera/reference frame diagrams and transform direction;
- operating envelope: range, angle, image region, speed, lighting, occlusion, marker count;
- failure policy and maximum allowed latency;
- acceptance values and reference uncertainty budget;
- exact software revision, OpenCV build/version, configs, board/tag print artifacts, and camera controls.

### Stage 1 — qualify physical references

1. Measure ChArUco/checkerboard scale in both axes and flatness.
2. Measure AprilTag detection-edge size, every tag corner, and every final object landmark.
3. Calibrate camera-to-reference and object-target transforms.
4. Verify reference repeatability and synchronization.
5. Produce an uncertainty budget with certificates, repeated measurements, fit residuals, and covariance.

Do not call the reference “ground truth” without its uncertainty and calibration chain ([NIST traceability policy](https://www.nist.gov/calibrations/traceability), [JCGM 100:2008](https://doi.org/10.59161/JCGM100-2008E)).

### Stage 2 — camera calibration qualification

1. Capture at least two independent sessions with broad image and pose coverage.
2. Fit the selected lens model on training views.
3. inspect per-view and spatial residual maps, parameter uncertainty/correlation, and session-to-session stability;
4. evaluate held-out views and downstream end-to-end landmark error;
5. repeat at each sensor mode/focus/zoom that will be supported.

The current three-frame minimum is a software guard, not an accuracy prescription ([`cli/charuco.py`](../../src/object_apriltag/cli/charuco.py)); calibration uncertainty depends on observation amount, coverage, pose diversity, model adequacy, and target quality ([Hagemann et al.](https://doi.org/10.1007/s11263-021-01528-x)).

### Stage 3 — observation qualification

Capture a designed grid over tag pixel size, incidence angle, image radius, blur, lighting, and exposure. Independently label/reference corners. Report localization error and detection/false-ID rates. Freeze this dataset for all downstream solver comparisons.

### Stage 4 — geometry and transform qualification

Use surveyed marker/object geometry. Run per-ID, leave-one-marker-out, and frame-algebra checks. Reject or correct tags whose stable ID-specific bias exceeds the geometry allocation. Keep these data separate from the final acceptance set.

### Stage 5 — pose/fusion qualification

Replay frozen corners through all candidate PnP/ambiguity/fusion methods. Compare to independent pose/landmark reference, not just reprojection. Record failure and candidate-switch rates.

### Stage 6 — end-to-end acceptance

Use unseen static poses distributed across the operating envelope, then unseen dynamic trajectories with synchronized reference data. Report aggregate and stratified metrics, confidence intervals, missing detections, and the complete uncertainty budget. Keep an immutable manifest mapping each result to raw images, timestamps, configs, calibration/model hashes, and reference records.

### Stage 7 — regression

Retain a small raw-frame replay set for deterministic software regressions and a smaller physical fixture test for camera/config regressions. Replay alone cannot detect a camera whose focus, mount, or controls changed.

## Acceptance placeholders

Fill these before looking at final acceptance data:

- operating range: `[____, ____] m`
- incidence angle: `≤ ____°`
- image region: `[full frame / specified ROI: ____]`
- object speed: `≤ ____ m/s`; angular speed: `≤ ____ °/s`
- required landmark detection probability: `≥ ____%`
- false-ID probability: `≤ ____`
- landmark median / RMSE / P95 / P99 / maximum: `____ / ____ / ____ / ____ / ____ mm`
- signed mean bias \(|x|/|y|/|z|\): `≤ ____ / ____ / ____ mm`
- translation P95: `≤ ____ mm`
- rotation P95: `≤ ____°`
- end-to-end latency P95 and timestamp uncertainty: `≤ ____ / ____ ms`
- reference expanded uncertainty and coverage factor: `≤ ____ mm, k=____`
- corner localization P95 under qualified conditions: `≤ ____ px`
- calibration session-to-session focal/principal/distortion stability: `[define: ____]`
- per-marker leave-one-out projected-corner P95: `≤ ____ px`
- maximum allowed invalid/missing-pose run: `____ frames or ____ ms`
- sample size: `≥ ____ independent sessions, ____ trajectories/poses per stratum`

An acceptance threshold below or close to the reference expanded uncertainty is not resolvable by that setup. Report “inconclusive at this reference capability” rather than pass/fail.

## Concise first experiment

**Purpose:** establish a defensible static end-to-end baseline and determine whether marker geometry is already a dominant, identifiable problem before adding dynamic complexity.

1. Rigidly mount the current camera and disable/fix autofocus, exposure, gain, zoom, resolution, fps, and processing where supported; record read-back values.
2. Attach the `remote1` object to a rigid fixture. Independently measure the three `object_model.json` landmarks and all AprilTag detection corners in one object frame; record measurement uncertainty.
3. Place at least 20 unseen static poses spanning near/mid/far distance, central/corner image positions, and frontal/oblique views. Use an independently surveyed fixture or synchronized metrology system to provide \(^{C}T_O\).
4. At each pose, capture at least 100 frames without moving anything. Save raw frames, detected corners/IDs, per-marker poses, fused pose, camera controls, and timestamps.
5. Compute end-to-end landmark errors, static repeatability, detection rate, and per-ID object-pose/landmark bias.
6. Replay the same detected corners with (a) the current model and (b) the independently surveyed marker model. Do not recalibrate intrinsics or tune on these poses.
7. Bootstrap by physical pose, and report results by marker ID/count, range, angle, and image radius with the reference uncertainty.

If independent metrology is not yet available, run the same gridded 100-frame static captures and the fresh-calibration replay as a **preflight**. It can rank repeatability, per-marker disagreement, image-region/range/angle sensitivity, and between-calibration shifts, but it cannot establish absolute accuracy or uniquely attribute a stable common bias. A repeatable wrong answer remains possible.

Valid conclusions:

- the total static end-to-end error distribution for this camera/config/object and tested envelope;
- an upper bound on random frame-to-frame variation at fixed poses;
- whether substituting surveyed marker geometry materially changes held-out landmark error;
- whether stable marker-ID-specific offsets are consistent with marker-model errors;
- whether error correlates with image radius/range/angle strongly enough to prioritize an intrinsics experiment.

Invalid conclusions:

- dynamic accuracy, latency, or rolling-shutter performance;
- universal AprilTag corner accuracy;
- that unchanged error after model substitution proves camera calibration is the cause;
- that low live-HUD reprojection error means landmarks are accurate;
- that one camera calibration or one object generalizes to other focus, sensor mode, print, mount, camera, or object instances;
- that an observed component smaller than the reference uncertainty has been resolved.

## Source register: strength and limitations

**Strength A — normative/official implementation evidence**

- [OpenCV calibration and pose API](https://docs.opencv.org/4.9.0/d9/d0c/group__calib3d.html): official mathematical/API behavior for calibration, projection, and PnP. **Limitation:** documents the library, not this camera's uncertainty or adequacy of a chosen model.
- [OpenCV solvePnP guide](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html): official solver constraints and multi-solution behavior. **Limitation:** does not validate this repository's point ordering, transforms, or fusion.
- [OpenCV ArUco API](https://docs.opencv.org/4.11.0/de/d67/group__objdetect__aruco.html): official meaning of AprilTag dictionaries and corner-refinement mode. **Limitation:** no camera-specific localization guarantee.
- [AprilRobotics source and pose notes](https://github.com/AprilRobotics/apriltag): first-party tag-size and detector implementation evidence. **Limitation:** this repository uses OpenCV's implementation/presets, not necessarily identical behavior.
- [JCGM 100:2008 GUM](https://doi.org/10.59161/JCGM100-2008E): international metrology guidance for uncertainty and covariance propagation. **Limitation:** supplies a framework, not this experiment's input distributions.
- [NIST traceability policy](https://www.nist.gov/calibrations/traceability): authoritative interpretation of traceability and fitness-for-purpose limits. **Limitation:** traceability alone does not provide a low-enough uncertainty.

**Strength B — peer-reviewed/foundational primary research**

- [Zhang 2000](https://doi.org/10.1109/34.888718): foundational planar camera-calibration method and assumptions. **Limitation:** its idealized model and reported experiments do not cover every modern lens, target defect, or rolling-shutter condition.
- [Olson 2011](https://doi.org/10.1109/ICRA.2011.5979561): primary AprilTag system and benchmark paper. **Limitation:** detection benchmarks are not a metrological certification of corners or downstream object landmarks.
- [Wang and Olson 2016](https://april.eecs.umich.edu/pdfs/wang2016iros.pdf): primary AprilTag 2 detector paper. **Limitation:** focuses strongly on detector robustness/efficiency; OpenCV integration and this hardware still require validation.
- [Collins and Bartoli 2014](https://doi.org/10.1007/s11263-014-0725-5): primary IPPE derivation and planar ambiguity analysis. **Limitation:** does not validate the repository's choice to consume one solution or its multi-marker fusion.
- [Hagemann et al. 2022](https://doi.org/10.1007/s11263-021-01528-x): primary work separating calibration bias and uncertainty and testing resampling under non-ideal conditions. **Limitation:** methods must be adapted and validated for this ChArUco/OpenCV workflow.
- [Torsello et al. 2010](https://www.dsi.unive.it/~atorsell/papers/Conferences/bmvc2010.pdf): primary experiment showing that inaccurate calibration-target geometry can coexist with low reprojection error. **Limitation:** its targets, cameras, and calibration implementation differ from this project.
- [Meingast, Geyer, and Sastry 2005](https://arxiv.org/abs/cs/0503076): foundational rolling-shutter geometry. **Limitation:** actual sensor readout/exposure behavior must be measured; the paper's approximations are not a device specification.
- [Sturm et al. 2012](https://doi.org/10.1109/IROS.2012.6385773): primary example of synchronized motion-capture ground truth and pose-error evaluation. **Limitation:** SLAM trajectory metrics and its hardware are precedents, not direct acceptance values for object landmarks.

## Likely future implementation touchpoints

No source changes are made by this note. If measurement support is implemented later, keep it separate from live visualization and prefer raw-data logging plus offline replay:

- [`cli/charuco.py`](../../src/object_apriltag/cli/charuco.py): save observations, flags/model, OpenCV RMS, per-view errors, uncertainty, and calibration-session metadata;
- [`detector.py`](../../src/object_apriltag/detector.py): expose raw rejected/accepted detections and immutable corner records;
- [`pose.py`](../../src/object_apriltag/pose.py): expose all IPPE candidates, per-marker results, fusion diagnostics, and held-out—not same-fit—residual hooks;
- [`layout.py`](../../src/object_apriltag/layout.py): retain surveyed geometry uncertainty and datum provenance;
- [`cli/detect.py`](../../src/object_apriltag/cli/detect.py): log raw frames, monotonic/device timestamps, negotiated camera properties, read-back controls, and exact config hashes;
- a new offline evaluation command/module only after the dataset schema and acceptance metrics above are frozen.
