"""
predictors/parking.py

Vehicle detection + parking slot occupancy logic.
Used by app.py's /predict/parking endpoint.
"""

import base64
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11s.pt"

# COCO class id for "car"
CAR_CLASS_ID = 2
# Other vehicle types that can occupy a parking slot (optional, broader match)
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Annotation colors (BGR, since we draw with OpenCV)
OCCUPIED_COLOR = (0, 200, 0)   # green
VACANT_COLOR = (0, 0, 220)     # red
BOX_THICKNESS = 2
FILL_ALPHA = 0.25              # transparency of the occupied-slot fill

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
# Slot occupancy logic
# ---------------------------------------------------------------------------
def coordinates_to_polygon(coordinates: List[float]) -> np.ndarray:
    """
    Converts a flat [x1, y1, x2, y2, x3, y3, x4, y4] list into an
    (N, 2) int32 numpy array suitable for cv2.pointPolygonTest.
    """
    if len(coordinates) % 2 != 0 or len(coordinates) < 6:
        raise ValueError(
            "Each slot's 'coordinates' must contain an even number of "
            "values (at least 3 points / 6 numbers)."
        )
    points = [
        (int(coordinates[i]), int(coordinates[i + 1]))
        for i in range(0, len(coordinates), 2)
    ]
    return np.array(points, np.int32)


def draw_slot_annotations(
    image: np.ndarray,
    parking_slots: List[dict],
    status_by_id: dict,
) -> np.ndarray:
    """
    Draws a filled, semi-transparent box + outline over each parking slot:
    green for occupied, red for vacant. Also labels each slot with its id
    and status. Returns a new annotated image (input image is left intact).
    """
    annotated = image.copy()
    overlay = image.copy()

    for slot in parking_slots:
        slot_id = slot["id"]
        polygon = coordinates_to_polygon(slot["coordinates"])
        status = status_by_id.get(slot_id, "Vacant")
        color = OCCUPIED_COLOR if status == "Occupied" else VACANT_COLOR

        # Semi-transparent fill on the overlay
        cv2.fillPoly(overlay, [polygon], color)

        # Solid outline on the annotated image
        cv2.polylines(
            annotated, [polygon], isClosed=True, color=color, thickness=BOX_THICKNESS
        )

        # Label near the first vertex of the polygon
        label_x, label_y = polygon[0]
        cv2.putText(
            annotated,
            f"{slot_id}: {status}",
            (int(label_x), max(int(label_y) - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Blend the filled overlay with the outlined image
    cv2.addWeighted(overlay, FILL_ALPHA, annotated, 1 - FILL_ALPHA, 0, annotated)
    return annotated


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes an OpenCV (numpy) image to a base64 string."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode annotated image.")
    return base64.b64encode(buffer).decode("utf-8")


def predict_parking(
    image_bytes: bytes,
    parking_id,
    parking_slots: List[dict],
) -> dict:
    """
    Runs YOLO detection on the given image bytes and checks each parking
    slot polygon for an occupying vehicle. Returns slot statuses plus a
    base64-encoded annotated image (green boxes = occupied, red = vacant).

    parking_slots: list of {"id": <int>, "coordinates": [x1,y1,x2,y2,x3,y3,x4,y4]}
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    if not parking_slots:
        raise ValueError("parking_slots list cannot be empty.")

    model = get_model()
    results = model(image, verbose=False)

    # Collect centers of detected vehicles
    vehicle_centers = []
    confidences = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        class_id = int(class_id)
        if class_id in VEHICLE_CLASSES:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            vehicle_centers.append((cx, cy))
            confidences.append(conf)

    slot_status = []
    status_by_id = {}
    occupied_slots = 0
    for slot in parking_slots:
        slot_id = slot["id"]
        polygon = coordinates_to_polygon(slot["coordinates"])
        is_occupied = False
        for cx, cy in vehicle_centers:
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                is_occupied = True
                break
        status = "Occupied" if is_occupied else "Vacant"
        if is_occupied:
            occupied_slots += 1
        slot_status.append({"id": slot_id, "status": status})
        status_by_id[slot_id] = status

    total_slots = len(parking_slots)
    vacant_slots = total_slots - occupied_slots
    occupancy_rate = round((occupied_slots / total_slots) * 100, 0) if total_slots else 0
    confidence_score = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    annotated_image = draw_slot_annotations(image, parking_slots, status_by_id)
    annotated_image_b64 = encode_image_to_base64(annotated_image)

    return {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "vacant_slots": vacant_slots,
        "occupancy_rate": int(occupancy_rate),
        "confidence_score": confidence_score,
        "slot_status": slot_status,
        "annotated_image": annotated_image_b64,
    }