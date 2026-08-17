"""Entry point for ``python -m object_apriltag.marker_layout_calibration``.

Runs package self-checks and exits successfully when all invariants pass.
"""

from object_apriltag.marker_layout_calibration import _input_boundary_self_check, _self_check

if __name__ == "__main__":
    _self_check()
    _input_boundary_self_check()
    print("marker_layout_calibration self-check passed")
