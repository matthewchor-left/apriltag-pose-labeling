# Prior art for Marker Model Calibration

Date: 2026-08-14

## Decision summary

**This pipeline is not novel as a problem.** Recovering unknown rigid poses of square planar fiducials from co-visible observations, then refining them with corner-level bundle adjustment, is the UCO “planar marker mapping” problem ([Muñoz-Salinas et al. 2018, MarkerMapper](https://arxiv.org/abs/1606.00151), [Sarmadi et al. 2019](https://arxiv.org/abs/2103.09141)). Robotics practice independently rediscovered a weaker form as AprilTag **bundle calibration** ([apriltag_ros](https://wiki.ros.org/apriltag_ros/Tutorials/Bundle%20calibration)).

**This pipeline is distinctive as a product.** Relative to MarkerMapper / `apriltag_ros`, the current solver already has:

- combinatorial IPPE-branch assignment rather than “pick lowest reprojection error”
- optional anchor-core hierarchical expansion
- transform chaining (tags need not co-appear with the reference marker)
- mixed marker sizes
- hard quality gates and structured refuse/partial outcomes
- live pair-readiness HUD

**Do not rebuild MarkerMapper, TagSLAM, or GTSAM.** Those stacks solve a broader SLAM/map problem. Highest-payoff imports for *this* codebase are discrete-front-end robustness and capture guidance, not a new optimizer.

**Build next (ranked):**

1. Keep both IPPE hypotheses when their reprojection-error ratio is ambiguous; drop the worse one only when the ratio is clearly separable ([Collins and Bartoli 2014](https://doi.org/10.1007/s11263-014-0725-5), [Ch’ng et al. 2020](https://arxiv.org/abs/1909.11888)).
2. Turn pair-readiness into “capture this pair / viewpoint next” guidance, AprilCal-style ([Richardson, Strom, Olson 2013](https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf)).
3. Add per-frame / holdout reprojection diagnostics so in-sample 2 px RMS is not the only acceptance number.
4. Only if assignment still fails on 4-face objects: Ch’ng-style clique-constrained rotation consistency as a fallback, not a replacement of the current search.

Already decided elsewhere: keep SciPy sparse corner BA as the authoritative write path ([ADR 0002](../adr/0002-corner-bundle-adjustment-for-marker-calibration.md), [online vs offline note](./2026-08-12-online-vs-offline-marker-layout-optimization.md)).

## Scope and assumptions

Question: *Has similar or identical work been done, and what would change in this repo?*

This note concerns **Marker Model Calibration** as defined in [`CONTEXT.md`](../../CONTEXT.md): estimating sticker footprints on a rigid object from co-visible AprilTag corners, writing `marker_model.json`. Camera intrinsics stay fixed. Scale comes from known tag edge length(s).

Out of scope: replacing SciPy with GTSAM/iSAM; camera intrinsic calibration; deformable objects.

Assumptions aligned with the current implementation in [`marker_layout_calibration.py`](../../src/object_apriltag/marker_layout_calibration.py):

1. Accumulate co-visible frames (≥2 expected markers).
2. Per marker: `solvePnPGeneric(..., SOLVEPNP_IPPE)` → up to two facing-camera candidates.
3. Pairwise relative-transform seed-and-expand consensus; default 20 inliers, 10% translation / 5° rotation gates.
4. Combinatorial assignment of one IPPE branch per marker per frame.
5. Optional anchor-core `2^k` bootstrap + hierarchical expansion.
6. Pose-graph init at the reference marker; sparse Huber corner BA; prune >3 px and refit.
7. Hard quality gates before write.

Research was run with three Composer 2.5 subagents (systems / IPPE front-end / BA+gates) plus independent primary-source checks. Claims below are tied to fetched papers or official docs. Unverified items are listed at the end.

## Thematic map

### Theme 1: Unknown multi-marker layout from co-observations (near-exact match)

**MarkerMapper** ([Muñoz-Salinas et al., *Pattern Recognition* 2018](https://doi.org/10.1016/j.patcog.2017.08.010); [arXiv:1606.00151](https://arxiv.org/abs/1606.00151)) is the closest published system. From images of square markers at *unknown* relative poses it:

1. builds a **quiver** of pairwise relative marker poses from co-visible frames
2. collapses that to a pose graph (best edge per pair)
3. **distributes cycle error** on the graph (rotations then translations)
4. runs global optimization that treats each marker as a **rigid 6-DoF body** (four corners moved together), minimizing reprojection error

Their abstract states the problem this repo solves: mapping from a large set of planar markers whose relative pose is *not* known beforehand. Their §2.4 is explicit that planar pose has a two-fold ambiguity and that “robust methods for mapping and localization using squared planar markers must take this problem into consideration.”

Toolkit: [UCO MarkerMapper](https://www.uco.es/investiga/grupos/ava/portfolio/marker-mapper/). Operational constraint identical to this repo: a marker joins the map only when seen with a marker already in the map; disconnected clusters stay disconnected.

**Sarmadi et al.** ([IEEE Access 2019](https://doi.org/10.1109/access.2019.2896648); [arXiv:2103.09141](https://arxiv.org/abs/2103.09141)) is the closest *object-attached* paper: markers glued on a rigid object, unknown configuration, recovered jointly with (in their case) a static multi-camera rig. They compare against MarkerMapper + ArUco and AprilTag-2 single-marker tracking. Difference vs this repo: multi-camera extrinsics are part of the estimand; capture is offline video, not a gated live CLI.

**apriltag_ros bundle calibration** ([ROS Wiki tutorial](https://wiki.ros.org/apriltag_ros/Tutorials/Bundle%20calibration), [`calibrate_bundle.m`](https://github.com/AprilRobotics/apriltag_ros/blob/master/apriltag_ros/scripts/calibrate_bundle.m)) is the robotics-practice twin. For each frame where a **master** tag is visible, it stores tag-to-master relative transforms and outputs **median translation / mean quaternion**. No assignment, no corner BA, no chaining. The README lists transform chaining (cube opposite-face) as future work — this repo already does that via the pair graph. The wiki’s main warning is the same failure mode this repo’s gates exist to catch: a bad bundle calibration produces **mirrored** poses at unfortunate viewpoints.

**TagSLAM** ([Pfrommer et al. 2019](https://arxiv.org/abs/1910.00679), [docs](https://berndpfrommer.github.io/tagslam_web/)) models **bodies** with tags attached and can discover unknown tag poses when co-visible with known tags. It is a GTSAM SLAM front-end, not a marker-model calibrator. Useful as existence proof that “tags on a rigid object” is a first-class abstraction; not a backend to copy ([already decided](./2026-08-12-online-vs-offline-marker-layout-optimization.md)).

**Kalibr AprilGrid / AprilCal** use AprilTags whose **layout is known by construction**. They are *not* unknown-layout solvers. AprilCal is still relevant for capture UX (Theme 4).

### Theme 2: Planar two-fold ambiguity and discrete assignment

Established:

- Planar PnP has **at most two** rotation solutions (reflection about a plane through the camera–object line of sight) ([Collins and Bartoli, IJCV 2014](https://doi.org/10.1007/s11263-014-0725-5); author notes: [tobycollins/IPPE](https://github.com/tobycollins/IPPE)).
- The ambiguity is not only “far / small”: Schweighofer & Pinz showed two local minima even at close range with wide-angle lenses ([TPAMI 2006](https://doi.org/10.1109/tpami.2006.252)).
- When the projection is near-affine, both solutions can fit the four corners up to noise; **reprojection error cannot pick the true pose** (Collins §3.4: return both; likelihood-ratio test is application-specific).
- OpenCV `SOLVEPNP_IPPE` via `solvePnPGeneric` exposes both solutions and sorts by reprojection error; it does **not** resolve the ambiguity ([OpenCV solvePnP guide](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)).

This repo already keeps facing-camera IPPE candidates and disambiguates with **pair consensus + combinatorial assignment** — closer to the literature’s recommended shape than `apriltag_ros` (which trusts a single PnP per tag) or SPM-SLAM-style **per-marker error-ratio discard**.

**Ch’ng et al., ICRA 2020** ([arXiv:1909.11888](https://arxiv.org/abs/1909.11888)) is the paper that most directly attacks *this* front-end. They:

- keep all PPE/IPPE twins
- build a **multigraph** of marker-to-marker relative rotations (all `00/01/10/11` branch combinations)
- solve rotation averaging with **clique constraints** so each image uses a consistent hypothesis set
- finish with small maximum-weighted-clique problems

They report that per-marker ratio tests fail often (order of ~25% of detections “ambiguous” at ratio ≥ 0.6 in their Hotel2 example; the lower-error pose can be wrong). They then run MarkerMapper’s SfM pipeline. This is a drop-in *front-end* for a MarkerMapper-like BA, which is what this repo already has.

Pairwise consensus literature that generalizes the current seed-and-expand:

- **PCM** (pairwise consistency maximization / max clique on loop closures) ([Mangelson et al., ICRA 2018](https://doi.org/10.1109/ICRA.2018.8460217))
- Rotation averaging with L1 + IRLS ([Chatterjee and Govindu, TPAMI 2017](https://doi.org/10.1109/tpami.2017.2693984))
- Chordal / spectral pose-graph initialization ([Carlone et al. survey](https://www.cis.upenn.edu/~kostas/mypub.dir/tron15icra.pdf); MarkerMapper’s own cycle-error distribution)

The current `_best_pair_consensus` is greedy RANSAC-without-random: each seed expands inliers inside hard 10%/5° gates, largest support wins. That works when one cluster dominates. It can accept a **wrong-branch cluster** that is internally tight and has ≥20 frames. PCM/Ch’ng add *mutual* consistency across hypotheses, not just distance-to-seed.

### Theme 3: Bundle adjustment, robust loss, and quality gates

MarkerMapper, Sarmadi, Ceres examples, and this repo agree on the continuous problem: **reprojection of known-size planar squares**, Huber-like robustness, one marker = one rigid body, gauge fixed by a reference.

Practice numbers (in-sample RMS is a **fit residual**, not held-out truth — see also [object-landmark error note](./2026-08-11-object-landmark-error-measurement.md)):

| System | Typical gate / target | Notes |
|--------|----------------------|--------|
| Kalibr aprilgrid (community notes) | <0.3 px excellent; >1 px poor for VI-SLAM | Joint *intrinsic* calibration, printed grid, many corners |
| OpenCV / ChArUco community | ~0.1–1 px “good”; prune around 1–2 px | Again, camera calibration |
| Ceres `bundle_adjuster --robustify` | HuberLoss(1.0) | Same 1 px scale as this repo’s `huber_delta_px=1.0` |
| COLMAP | Cauchy on *local* BA, then plain LS globally | Robust first, clean second — analogue of prune-then-refit |
| **This repo** | 2 px global + per-marker; 3 px prune; 20 pair inliers; 10% / 5° pair RMS | Appropriate for live AprilTag layout with **fixed** intrinsics |

The 2 px write gate is **loose versus Kalibr camera-cal**, and that is correct: layout BA with motion blur, no extra corner-refine pass, and frozen ChArUco intrinsics should not pretend to be 0.3 px lab photogrammetry.

Few published toolkits gate **3D pair translation/rotation** the way this repo does. That is a strength. The corresponding gap is the missing **holdout / per-view** residual that OpenCV `perViewErrors`, MATLAB `showReprojectionErrors`, and Kalibr plots all expose.

Scale: every serious source sets metric scale from **printed black-border tag size**, not outer paper ([AprilTag pose docs](https://github.com/AprilRobotics/apriltag#pose-estimation)). This repo already documents that. Kalibr folklore: ~1% size error → ~10% metric error. An optional user-supplied **baseline between two marker IDs** would catch a wrong `--marker-size` that reprojection RMS will not.

MarkerMapper’s final step can **re-optimize camera intrinsics**. This repo should **not** copy that: few live-object views, already-calibrated camera, identifiability issues documented in the landmark-error note.

### Theme 4: Capture protocol and live guidance

AprilCal ([Richardson, Strom, Olson, IROS 2013](https://doi.org/10.1109/iros.2013.6696595); [PDF](https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf)) is camera-intrinsic calibration, but the *interaction model* transfers:

- score candidate next poses given current information
- tell the user where to put the target
- stop when a user-specified accuracy is met

This repo’s HUD already shows raw vs robust pair support and reference connectivity — ahead of `apriltag_ros` and MarkerMapper’s offline video ingest. It does **not** yet say “you need a frame with markers {2, 5} from a more oblique angle.” MarkerMapper’s own FAQ is exactly that sentence for disconnected clusters.

Zhang / OpenCV capture lore (15–20 diverse views, 45–60° tilt, full FOV, avoid head-on) still applies to the *object* being waved in front of a fixed camera: pair counts alone do not measure **view diversity**.

## How close is this repo?

```
apriltag_ros median bundle     MarkerMapper / Sarmadi     TagSLAM / UcoSLAM
   (too simple)                (same problem class)        (too much SLAM)
         |                              |                         |
         +----------- this repo --------+                         |
                    |                                             |
         gated live object calibrator                             |
         (assignment + BA + HUD)                                  |
```

| Stage | This repo | MarkerMapper | apriltag_ros bundle | Ch’ng 2020 |
|-------|-----------|--------------|---------------------|------------|
| IPPE twins | Keep facing-camera | Ambiguity-aware graph | Single pose | Keep all, lift to multigraph |
| Pair graph | Seed-and-expand + RMS gates | Best edge + cycle correction | Master-relative only | Rotation averaging + cliques |
| Chaining to reference | Yes | Yes | No (master must be visible) | Via MarkerMapper |
| Mixed sizes | Yes (`--marker-size-for`) | Typically uniform | Per-tag `size` in YAML | N/A |
| Continuous opt | SciPy Huber corner BA | LM corner BA; may refine *K* | None | Then MarkerMapper BA |
| Write policy | Hard gates / refuse / partial | Write the map | Print YAML | Research output |
| Live capture | C / S / pair HUD | Offline video | Bag then MATLAB | Offline |

## Ranked improvements for this codebase

Payoff vs cost for ~4–20 tags, SciPy BA, live CLI. Do not implement all of these.

### High payoff, low–medium cost

1. **Ambiguity-ratio pruning in `_ippe_candidates`**
   If `rms_best / rms_second < τ` (literature default ~0.6), emit only the better candidate; otherwise emit both. Collins explicitly recommends this. Cuts pair-hypothesis explosion without the failure mode of *always* taking min-error.
   *Code:* [`_ippe_candidates`](../../src/object_apriltag/marker_layout_calibration.py).

2. **Next-pair / next-view HUD hint**
   From dropped edges (`insufficient_observed_frames`, `insufficient_inlier_frames`) and raw connectivity, print one sentence: “Need more co-visible frames of 2–5” or “Need a more oblique view of 0–3.” AprilCal scoring is optional later; the discrete hint is already almost in `LivePairReadinessDiagnostics`.
   *Code:* [`calibrate_marker_model.py`](../../src/object_apriltag/cli/calibrate_marker_model.py), [`live_pair_readiness_worker.py`](../../src/object_apriltag/cli/live_pair_readiness_worker.py).

3. **Per-frame reprojection RMS in diagnostics JSON**
   “Worst 5 frames” to delete or recapture. Standard in OpenCV/Kalibr. Does not change the solver.

4. **Document why 2 px / 20 inliers / 5° are the way they are**
   Relative to Kalibr’s 0.3 px, these are *layout* gates. Operators currently have no way to know that passing 2 px is not photogrammetry-grade.

### Medium payoff, medium cost

5. **Holdout-frame RMS gate** (e.g. 10% of frames excluded from BA). Addresses in-sample overfitting ([calib.io on reprojection error](https://calib.io/blogs/knowledge-base/understanding-reprojection-errors)). Complements the existing landmark-error methodology.

6. **Optional metric spot-check:** user supplies distance between two marker centers; compare to solved layout. Catches wrong `--marker-size` that pair-ratio gates cannot.

7. **PCM-style mutual consistency on pair hypotheses** if greedy seed-and-expand is observed to lock onto a wrong IPPE branch (look for high inlier count + later assignment mass-rejection on that pair). Do not replace seed-and-expand speculatively.

8. **MarkerMapper cycle-error distribution** before BA on 15–20 tag graphs, as an alternative/complement to spanning-tree init from the reference. Anchor-core already covers small *k*; cycle correction helps long chains.

### High payoff, high cost — only if measured

9. **Ch’ng clique-constrained rotation averaging** as fallback when `_assign_ippe_candidates` rejects too many frames. Do not replace the current per-frame search first; Ch’ng is the accurate-but-heavy option even in later SLAM papers.

### Do not build

- GTSAM / TagSLAM / UcoSLAM as the write-path backend.
- Joint intrinsic refinement during object capture (MarkerMapper optional path).
- Switching Huber → Cauchy/Tukey.
- Full posterior covariance on `marker_model.json` unless a downstream consumer exists.
- Replacing combinatorial assignment with per-marker min-reprojection (that is a regression).

## Consensus, debate, and gaps

**Consensus**

- Unknown rigid multi-tag geometry from co-visibility is a solved *problem class*: pairwise graph → init → corner BA, scale from tag size, gauge at one marker.
- Planar pose twins must be handled with **multi-view consistency**, not single-view reprojection ranking, when the ratio is high.
- In-sample RMS is necessary but not sufficient.

**Debate / uncertainty**

- Best discrete front-end: greedy inlier expansion (this repo), cycle-corrected best-edge (MarkerMapper), or clique-lifted rotation averaging (Ch’ng). No head-to-head on *object-sticker* datasets with 4–20 tags.
- Whether `min_inliers_per_edge=20` is conservative. Literature rarely uses a fixed quorum; this repo’s number is an engineering reliability choice for a gated write.
- Facing-camera (`normal_z < 0`) as a hard filter: standard, but can drop the true branch at grazing angles if the flip also looks “facing.”

**Gaps**

- Almost no paper ships a **live, gated, refuse-to-write** object calibrator. That is this repo’s actual contribution.
- Mixed-size tags on a handheld object are under-studied; this repo is ahead of most 2018-era MarkerMapper usage.
- Capture-suggestion literature is almost all **camera** calibration (AprilCal), not “wave the remote until pair 2–5 is ready.”

## Suggested reading order

1. [Collins and Bartoli 2014 (IPPE)](https://doi.org/10.1007/s11263-014-0725-5) — why two poses exist.
2. [Muñoz-Salinas et al. 2018 (MarkerMapper)](https://arxiv.org/abs/1606.00151) — the system this solver is a cousin of.
3. [Sarmadi et al. 2019](https://arxiv.org/abs/2103.09141) — tags glued on an object.
4. [Ch’ng et al. 2020](https://arxiv.org/abs/1909.11888) — how to pick IPPE branches globally.
5. [apriltag_ros bundle calibration](https://wiki.ros.org/apriltag_ros/Tutorials/Bundle%20calibration) — what most roboticists actually run, and why it is weaker.
6. [AprilCal 2013](https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf) — capture guidance, not layout math.
7. Existing repo notes: [ADR 0002](../adr/0002-corner-bundle-adjustment-for-marker-calibration.md), [online vs offline BA](./2026-08-12-online-vs-offline-marker-layout-optimization.md), [landmark error](./2026-08-11-object-landmark-error-measurement.md).

## Unverified / not used as evidence

- MICCAI 2025 fiducial-object BA PDF cited by one subagent: content not independently re-checked here; omitted.
- MCGMapper (IROS 2024) mixed-size mapping: not fetched; mixed-size recommendation already supported by this repo’s CLI.
- Kalibr’s exact camera-BA robust kernel (Huber vs plain LS): only detection-stage 4σ filtering is clearly documented in public scripts.
- `apriltag_ros` “Section 3.1.4 of this report”: wiki cites it with no PDF/DOI.

## Search register

Composer 2.5 subagents plus parent searches included: MarkerMapper / SPM-SLAM / UcoSLAM / TagSLAM / AprilCal / apriltag_ros bundle / IPPE Collins / Ch’ng clique constraints / PCM / Kalibr aprilgrid / OpenCV ChArUco RMS / Ceres Huber / Zhang capture protocol / mixed-size AprilTags. Subagent IDs: systems [Rigid multi-tag layout](36ce8858-8a8d-45a3-ab49-9463e7edc540), front-end [IPPE assignment consensus](ba5765b8-6059-4d3f-9a14-c761a6f55ef8), back-end [Calibration quality and BA](dc4b6c28-9153-436a-9f4e-0255eaba947c).
