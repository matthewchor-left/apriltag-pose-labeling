"""AprilTag dictionary and detector parameter presets."""

from __future__ import annotations

import cv2

APRILTAG_DICTS = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
}

DEFAULT_APRILTAG_DICTIONARY = "36h11"


def build_detector_parameters(sensitivity: str = "relaxed") -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    if sensitivity == "default":
        return params

    if sensitivity == "relaxed":
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshConstant = 5.0
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        params.minMarkerDistanceRate = 0.02
        params.minOtsuStdDev = 1.0
        params.errorCorrectionRate = 0.75
        params.maxErroneousBitsInBorderRate = 0.45
        params.detectInvertedMarker = True
    elif sensitivity == "aggressive":
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 3.0
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.08
        params.minCornerDistanceRate = 0.01
        params.minDistanceToBorder = 0
        params.minMarkerDistanceRate = 0.01
        params.minOtsuStdDev = 0.5
        params.errorCorrectionRate = 0.85
        params.maxErroneousBitsInBorderRate = 0.55
        params.detectInvertedMarker = True
        params.useAruco3Detection = True
        params.minSideLengthCanonicalImg = 15
        params.minMarkerLengthRatioOriginalImg = 0.0
    else:
        raise ValueError(f"Unknown detection sensitivity {sensitivity!r}.")

    if sensitivity != "default":
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01

    if sensitivity == "aggressive":
        params.errorCorrectionRate = 0.9

    return params


def build_apriltag_detector(
    dictionary: str = DEFAULT_APRILTAG_DICTIONARY,
    sensitivity: str = "relaxed",
) -> cv2.aruco.ArucoDetector:
    if dictionary not in APRILTAG_DICTS:
        raise ValueError(f"Unknown dictionary {dictionary!r}.")
    apriltag_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[dictionary])
    return cv2.aruco.ArucoDetector(apriltag_dict, build_detector_parameters(sensitivity))
