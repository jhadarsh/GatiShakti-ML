"""
predictors/signal.py
Vehicle detection + adaptive traffic signal timing logic.
Used by app.py's /predict/signal endpoint.
"""

import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11s.pt"

# COCO class ids for vehicles: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Lazy-loaded singleton so the model is read from disk only once per process
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
# Signal timing logic
# ---------------------------------------------------------------------------
MIN_GREEN = 10
MAX_GREEN = 60
SECONDS_PER_VEHICLE = 1.0
YELLOW_TIME = 5.0


def compute_density(vehicle_count: int, road_width: float) -> str:
    """Very simple density classification based on count per metre of road width."""

    lanes = max(1, round(road_width/3.5))

    veh_per_lane = vehicle_count/lanes


    if veh_per_lane < 8:
        return "Low"

    elif veh_per_lane < 15:
        return "Medium"

    else:
        return "High"


def compute_signal_times(
    vehicle_count: int,
    road_width: float,
    previous_vehicle_count: int,
    previous_red_light_time: float,
):
    """Returns (green, yellow, red, confidence)."""
    base_green = vehicle_count * SECONDS_PER_VEHICLE
    # widen the road -> more lanes -> can clear more cars per second
    width_factor = max(road_width / 7.0, 0.5)  # 7m ~ standard 2-lane road
    green_time = base_green / width_factor
    green_time = max(MIN_GREEN, min(MAX_GREEN, green_time))

    yellow_time = YELLOW_TIME

    # Red time derived from how busy the *previous* cycle was (adaptive feedback)
    red_time = max(MIN_GREEN, previous_vehicle_count * 1.5)

    # crude confidence score: higher when current and previous counts agree (stable traffic)
    if previous_vehicle_count > 0:
        diff_ratio = abs(vehicle_count - previous_vehicle_count) / max(
            previous_vehicle_count, 1
        )
        confidence = max(0.5, 1.0 - diff_ratio)
    else:
        confidence = 0.75

    return round(green_time, 1), round(yellow_time, 1), round(red_time, 1), round(
        confidence, 2
    )


# ---------------------------------------------------------------------------
# Main entry point used by FastAPI route
# ---------------------------------------------------------------------------
def predict_signal(
    image_bytes: bytes,
    road_width: float,
    signal_id: str,
    timestamp: Optional[str],
    previous_vehicle_count: int = 0,
    previous_red_light_time: float = 0.0,
) -> dict:
    """
    Runs YOLO detection on the given image bytes and returns the full
    signal prediction payload (matches the API spec).
    """
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")

    model = get_model()
    results = model(image, verbose=False)

    vehicle_count = 0
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        class_id = int(class_id)
        if class_id in VEHICLE_CLASSES:
            vehicle_count += 1
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            label = f"{model.names[class_id]} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (int(x1), max(int(y1) - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    green_time, yellow_time, red_time, confidence = compute_signal_times(
        vehicle_count, road_width, previous_vehicle_count, previous_red_light_time
    )
    density = compute_density(vehicle_count, road_width)

    # Encode annotated image to base64 JPEG
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode output image.")
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "vehicle_count": vehicle_count,
        "annotated_image": annotated_b64,
        "traffic_density": density,
        "recommended_green_time": green_time,
        "recommended_yellow_time": yellow_time,
        "recommended_red_time": red_time,
        "confidence_score": confidence,
        "signal_id": signal_id,
        "timestamp": timestamp or str(int(time.time())),
    }