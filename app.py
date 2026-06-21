"""
app.py
FastAPI entrypoint for TrafficAI.
Run with: uvicorn app:app --reload --port 8000
"""
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictors.Signal import predict_signal
from predictors.parking import predict_parking
from predictors.lanemonitoring import predict_bus_lane_violation

app = FastAPI(title="TrafficAI", version="0.1.0")

# CORS: allows browser-based frontends on OTHER domains (e.g. your React/
# Vue app, or a hosted demo page) to call this API directly via fetch/axios.
# Postman, curl, and server-to-server calls are NEVER blocked by CORS - this
# only affects requests made from inside a browser page. "*" below means
# "any website may call this API" - fine for testing/hackathon use, but for
# a production deployment, replace it with your actual frontend's exact
# domain(s), e.g. allow_origins=["https://your-frontend.com"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class SignalPredictionResponse(BaseModel):
    vehicle_count: int
    annotated_image: str  # base64
    traffic_density: str
    recommended_green_time: float
    recommended_yellow_time: float
    recommended_red_time: float
    confidence_score: float
    signal_id: str
    timestamp: str

class SlotStatus(BaseModel):
    id: int
    status: str
 
 
class ParkingPredictionResponse(BaseModel):
    total_slots: int
    occupied_slots: int
    vacant_slots: int
    occupancy_rate: int
    confidence_score: float
    slot_status: List[SlotStatus]
    annotated_image: str  # base64, green boxes = occupied, red = vacant

class ViolationDetail(BaseModel):
    type: str
    bbox: List[int]  # [x1, y1, x2, y2]
 
 
class BusLaneViolationResponse(BaseModel):
    unauthorized_count: int
    confidence_score: float
    violations: List[ViolationDetail]
    annotated_image: str  # base64, red boxes = violating vehicles
  




@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/signal", response_model=SignalPredictionResponse)
async def predict_signal_endpoint(
    traffic_image: UploadFile = File(...),
    road_width: float = Form(...),
    signal_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    previous_vehicle_count: int = Form(0),
    previous_red_light_time: float = Form(0.0),
):
    if not traffic_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await traffic_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    ts = timestamp or datetime.utcnow().isoformat()

    try:
        result = predict_signal(
            image_bytes=image_bytes,
            road_width=road_width,
            signal_id=signal_id,
            timestamp=ts,
            previous_vehicle_count=previous_vehicle_count,
            previous_red_light_time=previous_red_light_time,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.post("/predict/parking", response_model=ParkingPredictionResponse)
async def predict_parking_endpoint(
    parking_image: UploadFile = File(...),
    parking_id: str = Form(...),
    parking_slots: str = Form(...),  # JSON string, see docstring below
):
    """
    parking_slots is sent as a JSON-encoded string field (multipart/form-data
    can't carry nested arrays directly), e.g.:
 
    [
      {"id": 1, "coordinates": [52,364,30,417,73,412,88,369]},
      {"id": 2, "coordinates": [105,353,86,428,137,427,146,358]}
    ]
    """
    if not parking_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
 
    image_bytes = await parking_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")
 
    try:
        slots = json.loads(parking_slots)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="parking_slots must be valid JSON, e.g. "
            '[{"id":1,"coordinates":[x1,y1,x2,y2,x3,y3,x4,y4]}]',
        )
 
    try:
        result = predict_parking(
            image_bytes=image_bytes,
            parking_id=parking_id,
            parking_slots=slots,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    return result


@app.post("/predict/buslane", response_model=BusLaneViolationResponse)
async def predict_bus_lane_endpoint(
    lane_image: UploadFile = File(...),
    signal_id: int = Form(...),
    bus_lane_coordinates: str = Form(...),  # JSON string, see docstring below
):
    """
    bus_lane_coordinates is sent as a JSON-encoded string field
    (multipart/form-data can't carry nested arrays directly), e.g.:
 
    [
      [x1, y1], [x2, y2], [x3, y3], [x4, y4],
      [x5, y5], [x6, y6], [x7, y7], [x8, y8]
    ]
 
    8 points (4 along the near/wide edge of the lane, 4 along the
    far/narrow edge) is recommended for an accurate perspective outline,
    but any polygon with 3+ [x, y] points is accepted.
 
    Any detected vehicle whose center point falls inside this polygon and
    is NOT classified as Bus or Truck is reported as a violation.
    """
    if not lane_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
 
    image_bytes = await lane_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")
 
    try:
        coordinates = json.loads(bus_lane_coordinates)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="bus_lane_coordinates must be valid JSON, e.g. "
            '[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]]',
        )
 
    try:
        result = predict_bus_lane_violation(
            image_bytes=image_bytes,
            signal_id=signal_id,
            bus_lane_coordinates=coordinates,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    return result
 