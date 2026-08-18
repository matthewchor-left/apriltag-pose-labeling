"""Tests for layout-wide runtime object pose estimation."""

from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from object_apriltag.layout import (
    build_marker_layout,
    footprint_from_dict,
    layout_point_to_object_frame,
)
from object_apriltag.pose import (
    GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR,
    GLOBAL_MARKER_MIN_MEAN_EDGE_PX,
    GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG,
    _detected_corners_valid,
    _ippe_marker_candidates,
    _mean_marker_edge_length_px,
    _observed_marker_plane_gate_passes,
    _pair_minimum_normal_angle_deg,
    estimate_global_layout_pose,
    estimate_marker_pose,
)

def _camera() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.zeros((5, 1), dtype=np.float64),
    )


def _project_layout_detections(
    layout,
    marker_ids: list[int],
    object_rvec: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    detections = []
    for marker_id in marker_ids:
        object_points = np.stack(
            [
                layout_point_to_object_frame(point, layout)
                for point in layout.footprints[marker_id].corners()
            ]
        )
        projected, _ = cv2.projectPoints(
            object_points,
            object_rvec,
            object_origin,
            camera_matrix,
            dist_coeffs,
        )
        detections.append(
            (projected.reshape(1, 4, 2).astype(np.float32), marker_id)
        )
    return detections


def _square_payload(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


def _two_marker_layout(z_offset: float) -> object:
    half = 0.035
    return build_marker_layout(
        0,
        0.07,
        {
            0: footprint_from_dict(0, _square_payload(half)),
            1: footprint_from_dict(1, _square_payload(half, z=z_offset)),
        },
    )


def _tilted_square_payload(
    half: float, tilt_x_deg: float, z: float = 0.0
) -> dict[str, list[float]]:
    angle = np.deg2rad(tilt_x_deg)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return {
        corner: (rotation @ np.array(coords, dtype=np.float64)).tolist()
        for corner, coords in _square_payload(half, z).items()
    }


def _shift_payload(
    payload: dict[str, list[float]],
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> dict[str, list[float]]:
    return {
        corner: [coords[0] + x, coords[1] + y, coords[2] + z]
        for corner, coords in payload.items()
    }


def _observable_two_marker_layout(tilt_x_deg: float = 35.0) -> object:
    half = 0.035
    return build_marker_layout(
        0,
        0.07,
        {
            0: footprint_from_dict(0, _square_payload(half)),
            1: footprint_from_dict(
                1, _shift_payload(_tilted_square_payload(half, tilt_x_deg), x=0.12)
            ),
        },
    )


def _observable_four_marker_layout() -> object:
    half = 0.035
    return build_marker_layout(
        0,
        0.07,
        {
            0: footprint_from_dict(0, _square_payload(half)),
            1: footprint_from_dict(
                1, _shift_payload(_tilted_square_payload(half, 35.0), x=0.12)
            ),
            2: footprint_from_dict(
                2, _shift_payload(_tilted_square_payload(half, -25.0), x=-0.12)
            ),
            3: footprint_from_dict(
                3, _shift_payload(_tilted_square_payload(half, 20.0), y=0.12)
            ),
        },
    )


def _rotation_from_plane_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_x = np.cross(helper, normal)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    return np.column_stack((axis_x, axis_y, normal))


def _fake_ippe_candidates(
    normals: list[np.ndarray],
    relative_errors: list[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if relative_errors is None:
        relative_errors = [0.01] * len(normals)
    candidates = []
    for normal, relative_error in zip(normals, relative_errors, strict=True):
        rotation = _rotation_from_plane_normal(normal)
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        candidates.append(
            (
                rvec.reshape(3),
                rotation,
                normal / np.linalg.norm(normal),
                relative_error,
            )
        )
    return candidates


class GlobalObjectPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = _observable_four_marker_layout()
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_rotation, _ = cv2.Rodrigues(self.object_rvec)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)
        self.marker_ids = [0, 1, 2, 3]

    def assert_pose_close(
        self, origin: np.ndarray | None, rotation: np.ndarray | None
    ) -> None:
        self.assertIsNotNone(origin)
        self.assertIsNotNone(rotation)
        assert origin is not None and rotation is not None
        np.testing.assert_allclose(origin, self.object_origin, atol=1e-4)
        np.testing.assert_allclose(rotation, self.object_rotation, atol=1e-4)

    def test_recovers_one_pose_from_all_layout_corners(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            self.marker_ids,
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assert_pose_close(
            *(estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            ) or (None, None))
        )

    def test_ransac_rejects_one_corrupted_marker(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            self.marker_ids,
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        corrupted_corners, corrupted_id = detections[-1]
        detections[-1] = (
            corrupted_corners
            + np.array([[[70.0, -50.0]]], dtype=np.float32),
            corrupted_id,
        )
        self.assert_pose_close(
            *(estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            ) or (None, None))
        )


class ObservedMarkerNormalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)
        self.half = 0.035

    def test_rejects_coplanar_observed_marker_planes(self) -> None:
        layout = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(self.half)),
                1: footprint_from_dict(1, _square_payload(self.half)),
            },
        )
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with mock.patch.object(cv2, "solvePnPRansac", autospec=True) as ransac_mock:
            self.assertIsNone(
                estimate_global_layout_pose(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        ransac_mock.assert_not_called()

    def test_accepts_nonparallel_observed_marker_planes(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNotNone(
            estimate_global_layout_pose(
                detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )

    def test_ippe_called_once_per_marker_not_per_pair(self) -> None:
        layout_three = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(self.half)),
                1: footprint_from_dict(
                    1, _shift_payload(_tilted_square_payload(self.half, 35.0), x=0.12)
                ),
                2: footprint_from_dict(
                    2, _shift_payload(_tilted_square_payload(self.half, -25.0), x=-0.12)
                ),
            },
        )
        detections_three = _project_layout_detections(
            layout_three,
            [0, 1, 2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with mock.patch.object(cv2, "solvePnPGeneric", wraps=cv2.solvePnPGeneric) as generic_mock:
            estimate_global_layout_pose(
                detections_three,
                layout_three,
                self.camera_matrix,
                self.dist_coeffs,
            )
        self.assertEqual(generic_mock.call_count, 3)

    def test_three_markers_compare_three_pairs(self) -> None:
        layout = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(self.half)),
                1: footprint_from_dict(
                    1, _shift_payload(_tilted_square_payload(self.half, 35.0), x=0.12)
                ),
                2: footprint_from_dict(
                    2, _shift_payload(_tilted_square_payload(self.half, -25.0), x=-0.12)
                ),
            },
        )
        detections = _project_layout_detections(
            layout,
            [0, 1, 2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with mock.patch("object_apriltag.pose.print") as print_mock:
            estimate_global_layout_pose(
                detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
            printed = [
                " ".join(str(arg) for arg in call.args)
                for call in print_mock.call_args_list
            ]
        pair_angles = [line for line in printed if "pair(" in line]
        self.assertEqual(len(pair_angles), 3)

    def test_accepted_geometry_preserves_existing_solve_path(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with (
            mock.patch.object(cv2, "solvePnPRansac", wraps=cv2.solvePnPRansac) as ransac_mock,
            mock.patch.object(cv2, "solvePnPRefineLM", wraps=cv2.solvePnPRefineLM) as refine_mock,
        ):
            pose = estimate_global_layout_pose(
                detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        self.assertIsNotNone(pose)
        self.assertEqual(ransac_mock.call_count, 1)
        self.assertEqual(refine_mock.call_count, 1)
        ransac_kwargs = ransac_mock.call_args.kwargs
        self.assertEqual(ransac_kwargs["flags"], cv2.SOLVEPNP_SQPNP)

    def test_diagnostics_omit_svd_sigma_logging(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with mock.patch("object_apriltag.pose.print") as print_mock:
            estimate_global_layout_pose(
                detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
            output = "\n".join(
                " ".join(str(arg) for arg in call.args)
                for call in print_mock.call_args_list
            )
        self.assertNotIn("sigma_min", output)
        self.assertNotIn("planarity_ratio", output)
        self.assertIn("markers=", output)
        self.assertIn("min_angle=", output)

    def test_pair_minimum_angle_uses_abs_dot_for_normal_sign(self) -> None:
        parallel = np.array([0.0, 0.0, 1.0])
        flipped = np.array([0.0, 0.0, -1.0])
        angle = _pair_minimum_normal_angle_deg([parallel], [flipped])
        assert angle is not None
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_pair_angle_boundary_exact_twenty_accepts(self) -> None:
        angle_rad = np.deg2rad(GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG)
        normal_a = np.array([0.0, 0.0, 1.0])
        normal_b = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
        min_angle = _pair_minimum_normal_angle_deg([normal_a], [normal_b])
        assert min_angle is not None
        self.assertGreaterEqual(
            min_angle + 1e-9, GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG
        )

    def test_pair_angle_boundary_just_below_twenty_rejects(self) -> None:
        angle_rad = np.deg2rad(GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG - 1e-4)
        normal_a = np.array([0.0, 0.0, 1.0])
        normal_b = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
        min_angle = _pair_minimum_normal_angle_deg([normal_a], [normal_b])
        assert min_angle is not None
        self.assertLess(min_angle, GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG)

    def test_one_aligned_branch_makes_pair_non_confident(self) -> None:
        parallel = np.array([0.0, 0.0, 1.0])
        tilted = np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))])
        min_angle = _pair_minimum_normal_angle_deg([parallel, tilted], [parallel])
        assert min_angle is not None
        self.assertLess(min_angle, GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG)

    def test_mean_edge_twenty_five_px_accepted_twenty_four_nine_rejected(self) -> None:
        square = np.array(
            [
                [100.0, 100.0],
                [125.0, 100.0],
                [125.0, 125.0],
                [100.0, 125.0],
            ],
            dtype=np.float32,
        )
        self.assertAlmostEqual(_mean_marker_edge_length_px(square), 25.0, places=6)

        layout = _observable_two_marker_layout()
        detections_ok = [(square.reshape(1, 4, 2), 0), (square.reshape(1, 4, 2), 1)]
        small = square.copy()
        scale = 24.9 / 25.0
        center = small.mean(axis=0)
        small = center + (small - center) * scale
        detections_small = [(small.reshape(1, 4, 2), 0), (square.reshape(1, 4, 2), 1)]

        parallel = np.array([0.0, 0.0, 1.0])
        tilted = np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))])

        def ippe_side_effect(
            corners: np.ndarray, *_args, **_kwargs
        ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
            if _mean_marker_edge_length_px(corners) < GLOBAL_MARKER_MIN_MEAN_EDGE_PX:
                return []
            normal = tilted if ippe_side_effect.calls % 2 else parallel
            ippe_side_effect.calls += 1
            return _fake_ippe_candidates([normal])

        ippe_side_effect.calls = 0

        with mock.patch(
            "object_apriltag.pose._ippe_marker_candidates",
            side_effect=ippe_side_effect,
        ):
            self.assertTrue(
                _observed_marker_plane_gate_passes(
                    detections_ok,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
            self.assertFalse(
                _observed_marker_plane_gate_passes(
                    detections_small,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )

    def test_relative_reprojection_threshold_and_normalization(self) -> None:
        layout = _observable_two_marker_layout()
        square = np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ],
            dtype=np.float32,
        )
        detections = [(square.reshape(1, 4, 2), 0), (square.reshape(1, 4, 2), 1)]
        parallel = np.array([0.0, 0.0, 1.0])
        tilted = np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))])

        at_threshold = GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR
        above_threshold = GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR + 1e-6

        def ippe_side_effect(
            corners: np.ndarray, *_args, **_kwargs
        ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
            marker_index = ippe_side_effect.calls
            ippe_side_effect.calls += 1
            normal = parallel if marker_index % 2 == 0 else tilted
            rel_error = above_threshold if marker_index == 2 else at_threshold
            if rel_error > GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR:
                return []
            return _fake_ippe_candidates([normal], [rel_error])

        ippe_side_effect.calls = 0

        with mock.patch(
            "object_apriltag.pose._ippe_marker_candidates",
            side_effect=ippe_side_effect,
        ):
            self.assertTrue(
                _observed_marker_plane_gate_passes(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
            self.assertFalse(
                _observed_marker_plane_gate_passes(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )

    def test_invalid_and_behind_camera_candidates_excluded(self) -> None:
        layout = _observable_two_marker_layout()
        corners = np.array(
            [[[100.0, 100.0], [150.0, 100.0], [150.0, 150.0], [100.0, 150.0]]],
            dtype=np.float32,
        )
        detections = [(corners, 0), (corners, 1)]
        valid = _fake_ippe_candidates([np.array([0.0, 0.0, 1.0])])
        invalid = [
            (
                np.array([np.nan, 0.0, 0.0]),
                _rotation_from_plane_normal(np.array([0.0, 0.0, 1.0])),
                np.array([0.0, 0.0, 1.0]),
                0.01,
            )
        ]

        with mock.patch(
            "object_apriltag.pose._ippe_marker_candidates",
            side_effect=[valid, invalid, valid, []],
        ):
            self.assertFalse(
                _observed_marker_plane_gate_passes(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )

    def test_single_ippe_branch_is_candidate_not_automatic_certainty(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        parallel = np.array([0.0, 0.0, 1.0])

        def single_branch(*_args, **_kwargs):
            return _fake_ippe_candidates([parallel])

        with (
            mock.patch(
                "object_apriltag.pose._ippe_marker_candidates",
                side_effect=single_branch,
            ),
            mock.patch.object(cv2, "solvePnPRansac", autospec=True) as ransac_mock,
        ):
            self.assertIsNone(
                estimate_global_layout_pose(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        ransac_mock.assert_not_called()

    def test_all_near_parallel_branch_combinations_reject_before_ransac(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        parallel = np.array([0.0, 0.0, 1.0])
        nearly_parallel = np.array(
            [0.0, np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0))]
        )

        with (
            mock.patch(
                "object_apriltag.pose._ippe_marker_candidates",
                side_effect=[
                    _fake_ippe_candidates([parallel, nearly_parallel]),
                    _fake_ippe_candidates([parallel, nearly_parallel]),
                ],
            ),
            mock.patch.object(cv2, "solvePnPRansac", autospec=True) as ransac_mock,
        ):
            self.assertIsNone(
                estimate_global_layout_pose(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        ransac_mock.assert_not_called()

    def test_parallel_planes_at_different_depths_rejected(self) -> None:
        layout = _two_marker_layout(z_offset=0.12)
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        with mock.patch.object(cv2, "solvePnPRansac", autospec=True) as ransac_mock:
            self.assertIsNone(
                estimate_global_layout_pose(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        ransac_mock.assert_not_called()

    def test_three_markers_up_to_twelve_branch_combinations(self) -> None:
        layout = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(self.half)),
                1: footprint_from_dict(
                    1, _shift_payload(_tilted_square_payload(self.half, 35.0), x=0.12)
                ),
                2: footprint_from_dict(
                    2, _shift_payload(_tilted_square_payload(self.half, -25.0), x=-0.12)
                ),
            },
        )
        detections = _project_layout_detections(
            layout,
            [0, 1, 2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        branch_checks: list[int] = []

        def counting_pair_minimum(
            normals_a: list[np.ndarray], normals_b: list[np.ndarray]
        ) -> float | None:
            branch_checks.append(len(normals_a) * len(normals_b))
            return _pair_minimum_normal_angle_deg(normals_a, normals_b)

        with mock.patch(
            "object_apriltag.pose._pair_minimum_normal_angle_deg",
            side_effect=counting_pair_minimum,
        ):
            estimate_global_layout_pose(
                detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        self.assertEqual(branch_checks, [4, 4, 4])
        self.assertEqual(sum(branch_checks), 12)

    def test_ippe_marker_candidates_reject_small_marker(self) -> None:
        small_square = np.array(
            [
                [100.0, 100.0],
                [124.9, 100.0],
                [124.9, 124.9],
                [100.0, 124.9],
            ],
            dtype=np.float32,
        )
        self.assertLess(
            _mean_marker_edge_length_px(small_square),
            GLOBAL_MARKER_MIN_MEAN_EDGE_PX,
        )
        self.assertEqual(
            _ippe_marker_candidates(
                small_square,
                0.07,
                self.camera_matrix,
                self.dist_coeffs,
            ),
            [],
        )

    def test_detected_corners_must_be_finite_and_non_degenerate(self) -> None:
        square = np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(_detected_corners_valid(square))
        nan_square = square.copy()
        nan_square[0, 0] = np.nan
        self.assertFalse(_detected_corners_valid(nan_square))
        collapsed = np.array(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=np.float32,
        )
        self.assertFalse(_detected_corners_valid(collapsed))
        with mock.patch.object(cv2, "solvePnPGeneric", autospec=True) as generic_mock:
            self.assertEqual(
                _ippe_marker_candidates(
                    nan_square,
                    0.07,
                    self.camera_matrix,
                    self.dist_coeffs,
                ),
                [],
            )
        generic_mock.assert_not_called()

    def test_duplicate_marker_id_calls_ippe_once_and_uses_first_detection(self) -> None:
        layout = _observable_two_marker_layout()
        first_detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        good_corners, _ = first_detections[0]
        corrupted_corners = good_corners + np.array([[[80.0, -60.0]]], dtype=np.float32)
        duplicate_detections = [
            first_detections[0],
            (corrupted_corners, 0),
            first_detections[1],
        ]
        with mock.patch(
            "object_apriltag.pose._ippe_marker_candidates",
            wraps=_ippe_marker_candidates,
        ) as ippe_mock:
            self.assertTrue(
                _observed_marker_plane_gate_passes(
                    duplicate_detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        self.assertEqual(ippe_mock.call_count, 2)
        np.testing.assert_allclose(
            ippe_mock.call_args_list[0].args[0],
            good_corners,
        )

    def test_duplicate_marker_id_first_invalid_skips_later_valid(self) -> None:
        layout = _observable_two_marker_layout()
        valid_detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        tiny_corners = valid_detections[0][0].copy()
        scale = 24.0 / _mean_marker_edge_length_px(tiny_corners)
        center = tiny_corners.reshape(4, 2).mean(axis=0)
        tiny_corners = (
            center
            + (tiny_corners.reshape(4, 2) - center) * scale
        ).reshape(1, 4, 2).astype(np.float32)
        duplicate_detections = [
            (tiny_corners, 0),
            valid_detections[0],
            valid_detections[1],
        ]
        with mock.patch(
            "object_apriltag.pose._ippe_marker_candidates",
            wraps=_ippe_marker_candidates,
        ) as ippe_mock:
            self.assertFalse(
                _observed_marker_plane_gate_passes(
                    duplicate_detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )
        self.assertEqual(ippe_mock.call_count, 2)

    def test_duplicate_marker_diagnostics_list_unique_ids(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        duplicate_detections = [detections[0], detections[0], detections[1]]
        with mock.patch("object_apriltag.pose.print") as print_mock:
            _observed_marker_plane_gate_passes(
                duplicate_detections,
                layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        marker_lines = [
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
            if str(call.args[0]).startswith("[pose observability] markers=")
        ]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("markers=[0, 1]", marker_lines[0])

    def test_ippe_rejects_behind_camera_branch(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        corners = detections[0][0]
        good_rvec, good_tvec = estimate_marker_pose(
            corners,
            0.07,
            self.camera_matrix,
            self.dist_coeffs,
        )
        good_rvec = np.asarray(good_rvec, dtype=np.float64).reshape(3, 1)
        good_tvec = np.asarray(good_tvec, dtype=np.float64).reshape(3, 1)
        behind_rvec = np.zeros((3, 1), dtype=np.float64)
        behind_tvec = np.array([[0.0], [0.0], [-0.5]], dtype=np.float64)

        def generic_side_effect(*_args, **_kwargs):
            return (
                True,
                [good_rvec, behind_rvec],
                [good_tvec, behind_tvec],
                None,
            )

        with mock.patch.object(
            cv2, "solvePnPGeneric", side_effect=generic_side_effect
        ):
            candidates = _ippe_marker_candidates(
                corners,
                0.07,
                self.camera_matrix,
                self.dist_coeffs,
            )
        self.assertEqual(len(candidates), 1)

    def test_ippe_drops_high_reprojection_branch(self) -> None:
        square = np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ],
            dtype=np.float32,
        )
        mean_edge = _mean_marker_edge_length_px(square)
        normal = np.array([0.0, 0.0, 1.0])
        pose = _fake_ippe_candidates([normal], [0.01])[0]
        rvec = pose[0].reshape(3, 1)
        tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

        def generic_ok(*_args, **_kwargs):
            return (True, [rvec, rvec], [tvec, tvec], None)

        with (
            mock.patch.object(cv2, "solvePnPGeneric", side_effect=generic_ok),
            mock.patch(
                "object_apriltag.pose._mean_reprojection_error_px",
                side_effect=[
                    mean_edge * 0.02,
                    mean_edge * 0.12,
                ],
            ),
        ):
            candidates = _ippe_marker_candidates(
                square, 0.07, self.camera_matrix, self.dist_coeffs
            )
        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0][3], GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR)

    def test_high_reprojection_branch_does_not_force_pair_rejection(self) -> None:
        layout = _observable_two_marker_layout()
        detections = _project_layout_detections(
            layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        parallel = np.array([0.0, 0.0, 1.0])
        tilted = np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))])
        parallel_pose = _fake_ippe_candidates([parallel])[0]
        tilted_pose = _fake_ippe_candidates([tilted])[0]
        parallel_rvec = parallel_pose[0].reshape(3, 1)
        parallel_tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
        tilted_rvec = tilted_pose[0].reshape(3, 1)
        tilted_tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
        mean_edge = 100.0

        def generic_side_effect(*_args, **_kwargs):
            index = generic_side_effect.calls
            generic_side_effect.calls += 1
            if index == 0:
                return (
                    True,
                    [parallel_rvec, parallel_rvec],
                    [parallel_tvec, parallel_tvec],
                    None,
                )
            return (True, [tilted_rvec], [tilted_tvec], None)

        generic_side_effect.calls = 0

        with (
            mock.patch.object(
                cv2, "solvePnPGeneric", side_effect=generic_side_effect
            ),
            mock.patch(
                "object_apriltag.pose._mean_reprojection_error_px",
                side_effect=[
                    mean_edge * 0.02,
                    mean_edge * 0.12,
                    mean_edge * 0.02,
                ],
            ),
        ):
            self.assertTrue(
                _observed_marker_plane_gate_passes(
                    detections,
                    layout,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            )

    def test_ippe_relative_reprojection_exact_five_percent_branch_boundary(self) -> None:
        square = np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ],
            dtype=np.float32,
        )
        mean_edge = _mean_marker_edge_length_px(square)
        at_threshold_px = mean_edge * GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR
        above_threshold_px = mean_edge * (
            GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR + 1e-6
        )
        normal = np.array([0.0, 0.0, 1.0])
        pose = _fake_ippe_candidates([normal], [0.01])[0]
        rvec = pose[0].reshape(3, 1)
        tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

        def generic_ok(*_args, **_kwargs):
            return (True, [rvec], [tvec], None)

        with (
            mock.patch.object(cv2, "solvePnPGeneric", side_effect=generic_ok),
            mock.patch(
                "object_apriltag.pose._mean_reprojection_error_px",
                return_value=at_threshold_px,
            ),
        ):
            accepted = _ippe_marker_candidates(
                square, 0.07, self.camera_matrix, self.dist_coeffs
            )
        self.assertEqual(len(accepted), 1)
        self.assertAlmostEqual(accepted[0][3], GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR)

        with (
            mock.patch.object(cv2, "solvePnPGeneric", side_effect=generic_ok),
            mock.patch(
                "object_apriltag.pose._mean_reprojection_error_px",
                return_value=above_threshold_px,
            ),
        ):
            rejected = _ippe_marker_candidates(
                square, 0.07, self.camera_matrix, self.dist_coeffs
            )
        self.assertEqual(len(rejected), 0)


if __name__ == "__main__":
    unittest.main()
