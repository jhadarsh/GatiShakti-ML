"""
predictors/lanemonitoring.py

Bus lane violation detection logic.
Used by app.py's /predict/buslane endpoint.

Detects any vehicle inside a defined bus lane polygon that is NOT an
authorized class. Both "Bus" and "Truck" are treated as authorized -
many Indian buses (especially mini-buses / private operators) are visually
similar to trucks and get misclassified by a stock COCO-trained YOLO model,
so excluding "Truck" too avoids flooding false violations against actual
buses.

IMPORTANT - model limitation:
This uses a standard COCO-trained YOLO model (the same yolo11s.pt used by
predictors/parking.py). COCO does NOT have an "auto-rickshaw" class, so
auto-rickshaws will currently be detected as whatever the model finds
visually closest (commonly "Car" or "Bike"), not labeled "Auto-rickshaw"
specifically. True auto-rickshaw / Indian-vehicle classification requires a
model fine-tuned on a dataset like IDD (Indian Driving Dataset). To upgrade
later: point MODEL_PATH at the fine-tuned weights and update
VEHICLE_CLASS_NAMES / AUTHORIZED_CLASSES to match its class ids - nothing
else in this module needs to change.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11s.pt"

# Stock COCO class ids this module cares about, and how to label them in
# the API output. Update this mapping if you switch to a fine-tuned model
# with different/additional classes (e.g. a dedicated Auto-rickshaw class).
VEHICLE_CLASS_NAMES = {
    1: "Bicycle",
    2: "Car",
    3: "Bike",      # COCO "motorcycle"
    5: "Bus",
    7: "Truck",
}

# Classes allowed inside the bus lane - never flagged as violations even
# if detected inside the polygon. See module docstring for why Truck is
# included alongside Bus.
AUTHORIZED_CLASSES = {5, 7}  # Bus, Truck

# Annotation colors (BGR)
LANE_OUTLINE_COLOR = (0, 220, 220)     # yellow - bus lane boundary
VIOLATION_BOX_COLOR = (0, 0, 230)      # red - violating vehicle box
BOX_THICKNESS = 2

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                f"Run scripts/download_models.py first."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


# ---------------------------------------------------------------------------
# Image decoding (EXIF-safe - mirrors predictors/parking.py exactly)
# ---------------------------------------------------------------------------
def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR numpy array, applying EXIF
    orientation correction first. Keep this in sync with
    predictors/parking.py's decode_image() - both must behave identically
    so coordinates picked with tools/pick_lane.py line up correctly here.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Lane polygon helpers
# ---------------------------------------------------------------------------
def points_to_polygon(points: List[List[float]]) -> np.ndarray:
    """
    Converts a list of [x, y] pairs into an (N, 2) int32 numpy array
    suitable for cv2.pointPolygonTest / cv2.polylines.
    """
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(
            "bus_lane_coordinates must be a list of at least 3 [x, y] points."
        )
    polygon_points = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(
                "Each bus_lane_coordinates entry must be a [x, y] pair, "
                'e.g. [[x1,y1],[x2,y2],...].'
            )
        polygon_points.append((int(p[0]), int(p[1])))
    return np.array(polygon_points, np.int32)


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes an OpenCV (numpy) image to a base64 string."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode annotated image.")
    return base64.b64encode(buffer).decode("utf-8")


def draw_lane_annotations(
    image: np.ndarray,
    lane_polygon: np.ndarray,
    violations: List[dict],
) -> np.ndarray:
    """
    Draws the bus lane outline (yellow) and a red box + type label around
    every violating vehicle. Returns a new annotated image.
    """
    annotated = image.copy()

    cv2.polylines(
        annotated, [lane_polygon], isClosed=True,
        color=LANE_OUTLINE_COLOR, thickness=BOX_THICKNESS,
    )

    for v in violations:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), VIOLATION_BOX_COLOR, BOX_THICKNESS)
        cv2.putText(
            annotated,
            v["type"],
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            VIOLATION_BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def predict_bus_lane_violation(
    image_bytes: bytes,
    signal_id,
    bus_lane_coordinates: List[List[float]],
) -> dict:
    """
    Detects vehicles inside the given bus lane polygon and flags every
    vehicle that is NOT an authorized class (Bus / Truck) as a violation.

    bus_lane_coordinates: list of [x, y] pixel points outlining the bus
    lane region (8 points recommended - e.g. 4 along the near/wide edge
    and 4 along the far/narrow edge - to trace the lane's perspective
    accurately; any polygon with 3+ points is accepted).
    """
    image = decode_image(image_bytes)
    if image is None:
        raise ValueError("Could not decode uploaded image.")

    lane_polygon = points_to_polygon(bus_lane_coordinates)

    model = get_model()
    results = model(image, verbose=False)

    violations = []
    violation_confidences = []

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        class_id = int(class_id)

        if class_id not in VEHICLE_CLASS_NAMES:
            continue  # not a tracked vehicle class (e.g. person)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        inside_lane = cv2.pointPolygonTest(lane_polygon, (cx, cy), False) >= 0
        if not inside_lane:
            continue

        if class_id in AUTHORIZED_CLASSES:
            continue  # Bus / Truck - allowed in the bus lane

        violations.append({
            "type": VEHICLE_CLASS_NAMES[class_id],
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
        violation_confidences.append(conf)

    unauthorized_count = len(violations)
    confidence_score = (
        round(sum(violation_confidences) / len(violation_confidences), 2)
        if violation_confidences else 0.0
    )

    annotated_image = draw_lane_annotations(image, lane_polygon, violations)
    annotated_image_b64 = encode_image_to_base64(annotated_image)

    return {
        "unauthorized_count": unauthorized_count,
        "confidence_score": confidence_score,
        "violations": violations,
        "annotated_image": annotated_image_b64,
    }