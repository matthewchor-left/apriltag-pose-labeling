"""Matplotlib plots for object pose and marker layout."""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from object_apriltag.layout import CORNER_LABELS, MarkerLayout, layout_axis_limits, marker_color
from object_apriltag.viz.skeleton import ObjectModel


KEYPOINT_COLORS = {
    "top": "#800080",
    "bottom": "#00aa00",
    "handle": "#ffc0cb",
    "left": "#00cccc",
    "right": "#2a2aa5",
}


def is_sane_world_point(point: list[float], max_abs_m: float = 10.0) -> bool:
    values = np.asarray(point, dtype=np.float64)
    return values.shape == (3,) and np.all(np.isfinite(values)) and np.max(np.abs(values)) <= max_abs_m


def filter_sane_world_points(
    world_points: dict[str, list[float]],
    model: ObjectModel,
    max_abs_m: float = 10.0,
) -> dict[str, list[float]]:
    return {
        name: world_points[name]
        for name in model.keypoint_names
        if name in world_points and is_sane_world_point(world_points[name], max_abs_m)
    }


def set_xy_axis_limits(ax, axis_limits: tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, _, _ = axis_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def set_yz_axis_limits(ax, axis_limits: tuple[float, float, float, float, float, float]) -> None:
    _, _, ymin, ymax, zmin, zmax = axis_limits
    ax.set_xlim(ymax, ymin)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")


def _draw_skeleton_2d(ax, sane_points: dict[str, list[float]], model: ObjectModel, horiz_index: int, vert_index: int) -> None:
    for start_name, end_name in model.skeleton_edges:
        if start_name not in sane_points or end_name not in sane_points:
            continue
        start = np.asarray(sane_points[start_name], dtype=np.float64)
        end = np.asarray(sane_points[end_name], dtype=np.float64)
        ax.plot([start[horiz_index], end[horiz_index]], [start[vert_index], end[vert_index]], color="#444444", linewidth=2)

    for name, point in sane_points.items():
        values = np.asarray(point, dtype=np.float64)
        ax.scatter(values[horiz_index], values[vert_index], s=40, c=KEYPOINT_COLORS.get(name, "#888888"))
        ax.text(values[horiz_index], values[vert_index], f" {name}", fontsize=8)


def _draw_marker_layout_footprints_2d(ax, layout: MarkerLayout, horiz_index: int, vert_index: int) -> None:
    for marker_id in sorted(layout.footprints):
        footprint = layout.footprints[marker_id]
        color = marker_color(marker_id, layout.reference_marker_id)
        corners = footprint.corners()
        xs = [point[horiz_index] for point in corners] + [corners[0][horiz_index]]
        ys = [point[vert_index] for point in corners] + [corners[0][vert_index]]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.85)

        for corner_name, point in footprint.corners_by_name().items():
            label = CORNER_LABELS[corner_name]
            marker_style = {"tl": "o", "tr": "^", "br": "s", "bl": "D"}[label]
            ax.scatter(
                point[horiz_index], point[vert_index], s=70, c=color, marker=marker_style,
                edgecolors="black", linewidths=0.6, zorder=5,
            )
            ax.text(point[horiz_index], point[vert_index], f" {marker_id}:{label}", fontsize=7, color=color)


def render_marker_model_plot(
    marker_model: MarkerLayout,
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: int = 100,
) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    axis_limits = layout_axis_limits(marker_model)
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_yz = fig.add_subplot(1, 2, 2)

    _draw_marker_layout_footprints_2d(ax_xy, marker_model, horiz_index=0, vert_index=1)
    set_xy_axis_limits(ax_xy, axis_limits)
    ax_xy.set_title("X-Y (object frame)")
    ax_xy.set_aspect("equal", adjustable="box")

    _draw_marker_layout_footprints_2d(ax_yz, marker_model, horiz_index=1, vert_index=2)
    set_yz_axis_limits(ax_yz, axis_limits)
    ax_yz.set_title("Y-Z (object frame)")
    ax_yz.set_aspect("equal", adjustable="box")

    fig.suptitle("Marker model: tl/tr/br/bl corners", fontsize=10)
    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot = np.asarray(canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)


def render_pose_plots(
    world_points: dict[str, list[float]],
    model: ObjectModel,
    axis_limits: tuple[float, float, float, float, float, float],
    max_abs_m: float = 10.0,
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: int = 100,
) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sane_points = filter_sane_world_points(world_points, model, max_abs_m)
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_yz = fig.add_subplot(1, 2, 2)

    _draw_skeleton_2d(ax_xy, sane_points, model, horiz_index=0, vert_index=1)
    set_xy_axis_limits(ax_xy, axis_limits)
    ax_xy.set_title("X-Y")
    ax_xy.set_aspect("equal", adjustable="box")

    _draw_skeleton_2d(ax_yz, sane_points, model, horiz_index=1, vert_index=2)
    set_yz_axis_limits(ax_yz, axis_limits)
    ax_yz.set_title("Y-Z")
    ax_yz.set_aspect("equal", adjustable="box")

    fig.suptitle("Object landmarks (camera frame)", fontsize=10)
    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot = np.asarray(canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)


class LiveHud:
    def __init__(self, reproj_window: int = 30) -> None:
        self._prev_time = time.perf_counter()
        self._fps = 0.0
        self._reproj_means: deque[float] = deque(maxlen=reproj_window)
        self._reproj_maxes: deque[float] = deque(maxlen=reproj_window)

    def tick(
        self,
        reproj_mean: float | None = None,
        reproj_max: float | None = None,
    ) -> tuple[float, float | None, float | None]:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0.0:
            instant_fps = 1.0 / dt
            self._fps = instant_fps if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant_fps

        if reproj_mean is not None and reproj_max is not None:
            self._reproj_means.append(reproj_mean)
            self._reproj_maxes.append(reproj_max)

        avg_reproj = sum(self._reproj_means) / len(self._reproj_means) if self._reproj_means else None
        max_reproj = max(self._reproj_maxes) if self._reproj_maxes else None
        return self._fps, avg_reproj, max_reproj


def make_side_by_side(frame_bgr: np.ndarray, plot_bgr: np.ndarray, target_height: int) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_scale = target_height / frame_h
    frame_resized = cv2.resize(frame_bgr, (int(frame_w * frame_scale), target_height))

    plot_h, plot_w = plot_bgr.shape[:2]
    plot_scale = target_height / plot_h
    plot_resized = cv2.resize(plot_bgr, (int(plot_w * plot_scale), target_height))

    return np.hstack([frame_resized, plot_resized])
