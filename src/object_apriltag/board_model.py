"""ChArUco board model loader and OpenCV board construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

ARUCO_DICTIONARIES: dict[str, int] = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}

REFERENCE_FRAME_ORIGINS = frozenset({"outer_top_left"})
REFERENCE_FRAME_AXES = frozenset({"right", "out_of_printed_face", "down"})
BOARD_GEOMETRY_CLI_FLAGS = (
    "--board-type",
    "--layout",
    "--marker-size",
    "--square-size",
    "--dictionary",
)

@dataclass(frozen=True)
class BoardReferenceFrame:
    """Board coordinate axes expressed in layout metadata.

    Attributes:
        origin: Named origin of the board frame (must be ``outer_top_left``).
        x_axis: Positive X direction name (must be ``right``).
        y_axis: Positive Y direction name (must be ``out_of_printed_face``).
        z_axis: Positive Z direction name (must be ``down``).
    """

    origin: str
    x_axis: str
    y_axis: str
    z_axis: str


@dataclass(frozen=True)
class BoardModel:
    """Validated ChArUco board geometry loaded from JSON.

    Attributes:
        board_type: Board kind identifier (must be ``charuco_board``).
        units: Length unit string (must be ``meters``).
        layout_height: Number of chessboard rows.
        layout_width: Number of chessboard columns.
        square_size: Square edge length in ``units``.
        marker_size: ArUco marker edge length in ``units``.
        dictionary: OpenCV ArUco dictionary name.
        reference_frame: Board coordinate frame metadata.
    """

    board_type: str
    units: str
    layout_height: int
    layout_width: int
    square_size: float
    marker_size: float
    dictionary: str
    reference_frame: BoardReferenceFrame

    @property
    def total_charuco_intersections(self) -> int:
        """Count interior chessboard corners used by ChArUco."""
        return (self.layout_height - 1) * (self.layout_width - 1)


def _validate_reference_frame(payload: dict[str, Any]) -> BoardReferenceFrame:
    """Parse and validate the board ``reference_frame`` object.

    Args:
        payload: Parsed board-model JSON root object.

    Returns:
        Validated reference-frame metadata.

    Raises:
        ValueError: If ``reference_frame`` is missing or uses unsupported axis names.
    """
    reference = payload.get("reference_frame")
    if not isinstance(reference, dict):
        raise ValueError("board_model must include a 'reference_frame' object.")

    origin = str(reference.get("origin", ""))
    x_axis = str(reference.get("x_axis", ""))
    y_axis = str(reference.get("y_axis", ""))
    z_axis = str(reference.get("z_axis", ""))
    if origin != "outer_top_left":
        raise ValueError("reference_frame.origin must be 'outer_top_left'.")
    if x_axis != "right":
        raise ValueError("reference_frame.x_axis must be 'right'.")
    if y_axis != "out_of_printed_face":
        raise ValueError("reference_frame.y_axis must be 'out_of_printed_face'.")
    if z_axis != "down":
        raise ValueError("reference_frame.z_axis must be 'down'.")
    return BoardReferenceFrame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


def load_board_model(path: str | Path) -> BoardModel:
    """Load and validate a ChArUco board model JSON file.

    Args:
        path: Path to a ``board_model.json`` file.

    Returns:
        Parsed and validated board geometry.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required fields are missing or violate board constraints.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Board model file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    board_type = str(data.get("board_type", ""))
    if board_type != "charuco_board":
        raise ValueError("board_type must be 'charuco_board'.")

    units = str(data.get("units", ""))
    if units != "meters":
        raise ValueError("units must be 'meters'.")

    layout = data.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("layout must be an object with 'height' and 'width'.")
    layout_height = int(layout["height"])
    layout_width = int(layout["width"])
    if layout_height < 2 or layout_width < 2:
        raise ValueError("layout height and width must each be at least 2.")

    square_size = float(data["square_size"])
    marker_size = float(data["marker_size"])
    if square_size <= 0.0:
        raise ValueError("square_size must be positive.")
    if marker_size <= 0.0:
        raise ValueError("marker_size must be positive.")
    if marker_size >= square_size:
        raise ValueError("marker_size must be smaller than square_size.")

    dictionary = str(data.get("dictionary", ""))
    if dictionary not in ARUCO_DICTIONARIES:
        raise ValueError(f"dictionary must be one of {sorted(ARUCO_DICTIONARIES)}.")

    reference_frame = _validate_reference_frame(data)
    return BoardModel(
        board_type=board_type,
        units=units,
        layout_height=layout_height,
        layout_width=layout_width,
        square_size=square_size,
        marker_size=marker_size,
        dictionary=dictionary,
        reference_frame=reference_frame,
    )


def build_charuco_board_from_geometry(
    layout_width: int,
    layout_height: int,
    square_size: float,
    marker_size: float,
    dictionary_name: str,
) -> cv2.aruco.CharucoBoard:
    """Construct an OpenCV ``CharucoBoard`` from scalar geometry fields.

    Args:
        layout_width: Number of chessboard columns.
        layout_height: Number of chessboard rows.
        square_size: Square edge length in meters.
        marker_size: ArUco marker edge length in meters.
        dictionary_name: Key in ``ARUCO_DICTIONARIES``.

    Returns:
        OpenCV ChArUco board instance.

    Raises:
        ValueError: If ``marker_size`` is not strictly smaller than ``square_size``.
        KeyError: If ``dictionary_name`` is not a supported dictionary key.
    """
    if marker_size >= square_size:
        raise ValueError("marker_size must be smaller than square_size for charuco_board.")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    return cv2.aruco.CharucoBoard(
        (layout_width, layout_height),
        float(square_size),
        float(marker_size),
        dictionary,
    )


def build_charuco_board(model: BoardModel) -> cv2.aruco.CharucoBoard:
    """Construct an OpenCV ``CharucoBoard`` from a validated ``BoardModel``.

    Args:
        model: Loaded board geometry.

    Returns:
        OpenCV ChArUco board instance.

    Raises:
        ValueError: If ``model.board_type`` is not ``charuco_board``.
    """
    if model.board_type != "charuco_board":
        raise ValueError("Only charuco_board models are supported.")
    return build_charuco_board_from_geometry(
        model.layout_width,
        model.layout_height,
        model.square_size,
        model.marker_size,
        model.dictionary,
    )


def board_model_geometry_flags_provided(args: Any) -> list[str]:
    """List CLI board-geometry flags explicitly set on ``args``.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Names of geometry flags that differ from their implicit defaults.
    """
    provided: list[str] = []
    if getattr(args, "board_type", None) not in (None, "charuco_board"):
        provided.append("--board-type")
    if getattr(args, "layout", None) is not None:
        provided.append("--layout")
    if getattr(args, "marker_size", None) is not None:
        provided.append("--marker-size")
    if getattr(args, "square_size", None) not in (None, 0.025):
        provided.append("--square-size")
    if getattr(args, "dictionary", None) not in (None, "4x4_50"):
        provided.append("--dictionary")
    return provided


def reject_mixed_board_model_args(args: Any) -> None:
    """Reject CLI usage that mixes ``--board-model`` with per-field geometry flags.

    Args:
        args: Parsed argparse namespace.

    Raises:
        RuntimeError: If ``--board-model`` is set together with conflicting geometry flags.
    """
    if getattr(args, "board_model", None) is None:
        return
    conflicts = board_model_geometry_flags_provided(args)
    if conflicts:
        joined = ", ".join(conflicts)
        raise RuntimeError(
            f"--board-model cannot be combined with individual board geometry flags: {joined}."
        )
