import numpy as np
import pytest

from paddle_apriltag.cli.projection import calibration_graphics_data, camera_projection


def test_projection_maps_opencv_principal_ray_to_ndc_origin() -> None:
    camera_matrix = np.array(
        [[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]]
    )
    projection = camera_projection(camera_matrix, 100, 80, 0.1, 10.0)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])

    clip = projection @ opencv_to_opengl @ np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(clip[:2] / clip[3], [0.0, 0.0])


def test_graphics_data_calculates_fov_and_physical_values() -> None:
    camera_matrix = np.array(
        [[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]]
    )
    data = calibration_graphics_data(
        camera_matrix, 100, 80, 0.1, 10.0, sensor_width_mm=10.0, sensor_height_mm=8.0
    )

    assert data["field_of_view_degrees"] == pytest.approx(
        {"horizontal": 53.13010235415598, "vertical": 53.13010235415598}
    )
    assert data["focal_length_mm"] == {"x": 10.0, "y": 8.0}
