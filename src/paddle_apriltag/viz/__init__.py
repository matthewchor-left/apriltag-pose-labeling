"""Optional visualization for paddle detection."""

from paddle_apriltag.viz.overlay import (
    draw_live_hud,
    draw_marker_annotations,
    draw_marker_layout_footprints,
    draw_paddle_orientation,
    draw_paddle_pose,
)
from paddle_apriltag.viz.plots import LiveHud, make_side_by_side, render_marker_layout_plot, render_pose_plots
from paddle_apriltag.viz.skeleton import DEFAULT_AXIS_LIMITS, DEFAULT_PADDLE_MODEL_PATH, PaddleModel, load_paddle_model, paddle_world_points_from_pose

__all__ = [
    "DEFAULT_AXIS_LIMITS",
    "DEFAULT_PADDLE_MODEL_PATH",
    "LiveHud",
    "PaddleModel",
    "draw_live_hud",
    "draw_marker_annotations",
    "draw_marker_layout_footprints",
    "draw_paddle_orientation",
    "draw_paddle_pose",
    "load_paddle_model",
    "make_side_by_side",
    "paddle_world_points_from_pose",
    "render_marker_layout_plot",
    "render_pose_plots",
]
