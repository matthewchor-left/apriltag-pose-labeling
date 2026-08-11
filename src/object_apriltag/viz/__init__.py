"""Optional visualization for object detection."""

from object_apriltag.viz.overlay import (
    draw_eraser_planes,
    draw_live_hud,
    draw_marker_annotations,
    draw_marker_model_footprints,
    draw_object_orientation,
    draw_object_pose,
)
from object_apriltag.viz.plots import LiveHud, make_side_by_side, render_marker_model_plot, render_pose_plots
from object_apriltag.viz.skeleton import DEFAULT_AXIS_LIMITS, DEFAULT_OBJECT_MODEL_PATH, ObjectModel, load_object_model, object_world_points_from_pose

__all__ = [
    "DEFAULT_AXIS_LIMITS",
    "DEFAULT_OBJECT_MODEL_PATH",
    "LiveHud",
    "ObjectModel",
    "draw_eraser_planes",
    "draw_marker_annotations",
    "draw_marker_model_footprints",
    "draw_object_orientation",
    "draw_object_pose",
    "load_object_model",
    "make_side_by_side",
    "object_world_points_from_pose",
    "render_marker_model_plot",
    "render_pose_plots",
]
