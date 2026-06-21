"""
tools/pick_lane.py

Interactive bus-lane coordinate picker.

Click points outlining the bus lane region in an image - 8 points
recommended (4 tracing the near/wide edge of the lane, 4 tracing the
far/narrow edge) to follow the road's perspective accurately. The tool
auto-stops accepting clicks after 8 points.

Coordinates are saved into a single shared JSON file (default:
lanecoordinates.json), keyed by the image's filename, so you can run this
against multiple test images one at a time and build up one coordinate
file covering all of them - each run only adds/updates its own image's
entry, existing entries for other images are preserved.

The image is decoded the EXACT same way (EXIF-safe) as
predictors/lanemonitoring.py, so coordinates picked here line up correctly
when sent to the API.

Run:
    python tools/pick_lane.py "C:\Users\adars\OneDrive\Desktop\HackProject\TrafficAI\testing\BusLane\Testing3.png"
    python tools/pick_lane.py path/to/test2.jpg
    python tools/pick_lane.py path/to/test5.jpg --output lanecoordinates.json

Controls:
    left-click  = add a point (ignored once 8 points are placed)
    u           = undo last point
    r           = reset all points for this image
    s           = save this image's points to the output JSON file & quit
    q           = quit without saving

Output format (lanecoordinates.json):
{
  "test2.jpg": [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]],
  "test5.jpg": [[x1,y1], ...]
}
"""
import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_POINTS = 8

# Max window size to fit on a normal screen. Image is only ever scaled
# DOWN for display - clicks are converted back to full original
# resolution before being stored, matching the API's image decoding.
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 800


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Same EXIF-safe decode used by predictors/lanemonitoring.py - keep
    these in sync. If you change one, change the other.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Pick bus lane corner coordinates for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image to mark.")
    parser.add_argument(
        "--output", type=str, default="lanecoordinates.json",
        help="Shared JSON file to save/update (default: lanecoordinates.json)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return

    output_path = Path(args.output)

    base_image = decode_image(image_path.read_bytes())
    orig_h, orig_w = base_image.shape[:2]
    print(f"Loaded {image_path.name}, original size: {orig_w}x{orig_h} (width x height)")

    scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
    if scale < 1.0:
        display_base = cv2.resize(base_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Display scaled to {display_base.shape[1]}x{display_base.shape[0]} (saved coordinates stay full-res)")
    else:
        display_base = base_image.copy()

    points = []  # stored in ORIGINAL image coordinates

    window_name = f"Pick bus lane ({MAX_POINTS} pts) | u=undo | r=reset | s=save&quit | q=quit - {image_path.name}"

    def to_display(x, y):
        return int(round(x * scale)), int(round(y * scale))

    def to_original(x, y):
        return int(round(x / scale)), int(round(y / scale))

    def redraw():
        display = display_base.copy()
        display_points = [to_display(x, y) for (x, y) in points]
        for i, (x, y) in enumerate(display_points):
            cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if i > 0:
                cv2.line(display, display_points[i - 1], (x, y), (0, 0, 255), 1)
        if len(display_points) == MAX_POINTS:
            cv2.line(display, display_points[-1], display_points[0], (0, 0, 255), 1)
        cv2.putText(display, f"{len(points)}/{MAX_POINTS} points", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= MAX_POINTS:
                print(f"Already have {MAX_POINTS} points - press 'u' to undo or 'r' to reset.")
                return
            points.append(to_original(x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_base.shape[1], display_base.shape[0])
    cv2.setMouseCallback(window_name, on_click)
    redraw()

    print(f"\nClick up to {MAX_POINTS} points outlining the bus lane.")
    print("u = undo | r = reset | s = save & quit | q = quit without saving\n")

    saved = False
    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            if points:
                points.pop()

        elif key == ord('r'):
            points = []

        elif key == ord('s'):
            if len(points) < 3:
                print("Need at least 3 points before saving.")
                continue
            if len(points) != MAX_POINTS:
                print(f"Warning: saving with {len(points)} points (recommended {MAX_POINTS}).")
            saved = True
            break

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if not saved:
        print("\nNo changes saved.")
        return

    # Load existing shared JSON (if any) and update only this image's entry
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            print(f"Warning: {output_path} was not valid JSON, starting fresh.")
            existing = {}
    else:
        existing = {}

    existing[image_path.name] = [[x, y] for (x, y) in points]
    output_path.write_text(json.dumps(existing, indent=2))

    print(f"\nSaved {len(points)} point(s) for '{image_path.name}' to {output_path.resolve()}")
    print(f"Total images in {output_path.name}: {len(existing)}")
    print("\nThis image's entry (paste into bus_lane_coordinates in Postman):\n")
    print(json.dumps(existing[image_path.name]))


if __name__ == "__main__":
    main()