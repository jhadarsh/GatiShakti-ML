"""
tools/pick_slots.py

Interactive parking-slot coordinate picker.

Click the corner points of each parking slot (in order around the box,
usually 4 points), then press 'n' to lock that slot in and start the next
one. The image is decoded the EXACT same way (EXIF-safe) as the
/predict/parking API, so the coordinates you pick here will line up
correctly when sent to the API - no more "shifted to the top" mismatch.

Run:
    python tools/pick_slots.py path/to/your_parking_lot.jpg

Controls:
    left-click  = add a corner point to the current slot
    n           = finish current slot (needs >= 3 points), start a new one
    u           = undo last point (or remove the last finished slot if no
                  points are pending)
    s           = finish current slot (if any points pending) and save+quit
    q           = quit without saving

Output:
    Writes slots.json in the current directory, and also prints a
    single-line JSON string you can paste directly into Postman's
    `parking_slots` form field.
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Max window size to fit on a normal screen. The image is only ever
# scaled DOWN for display - clicks are converted back to full original
# resolution before being stored/saved, so slots.json always matches the
# image's true pixel coordinates (what the API actually sees).
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 800


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Same EXIF-safe decode used by predictors/parking.py - keep these two
    in sync. If you change one, change the other.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def main():
    if len(sys.argv) < 2:
        print("Usage: python pick_slots.py <path_to_image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    base_image = decode_image(image_path.read_bytes())
    orig_h, orig_w = base_image.shape[:2]
    print(f"Loaded image, original size: {orig_w}x{orig_h} (width x height)")

    # Scale factor to fit the image on screen, never upscale small images
    scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
    if scale < 1.0:
        display_base = cv2.resize(base_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Display scaled to {display_base.shape[1]}x{display_base.shape[0]} (saved coordinates stay full-res)")
    else:
        display_base = base_image.copy()

    # All points/slots below are stored in ORIGINAL image coordinates.
    slots = []
    current_points = []  # original-resolution coordinates
    next_id = [1]

    window_name = "Pick slots | click corners | n=next slot | u=undo | s=save&quit | q=quit"

    def to_display(x, y):
        return int(round(x * scale)), int(round(y * scale))

    def to_original(x, y):
        return int(round(x / scale)), int(round(y / scale))

    def redraw():
        display = display_base.copy()
        for slot in slots:
            pts = np.array(
                [to_display(slot["coordinates"][i], slot["coordinates"][i + 1])
                 for i in range(0, len(slot["coordinates"]), 2)],
                np.int32,
            )
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)
            cv2.putText(display, str(slot["id"]), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        display_points = [to_display(x, y) for (x, y) in current_points]
        for i, (x, y) in enumerate(display_points):
            cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display, display_points[i - 1], (x, y), (0, 0, 255), 1)
        cv2.imshow(window_name, display)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append(to_original(x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_base.shape[1], display_base.shape[0])
    cv2.setMouseCallback(window_name, on_click)
    redraw()

    print("\nClick corner points for each slot, in order around the box.")
    print("n = save this slot & start next | u = undo | s = save all & quit | q = quit without saving\n")

    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):
            if len(current_points) < 3:
                print("Need at least 3 points before starting the next slot.")
                continue
            coords = []
            for (x, y) in current_points:
                coords.extend([x, y])
            slots.append({"id": next_id[0], "coordinates": coords})
            print(f"Slot {next_id[0]} saved: {coords}")
            next_id[0] += 1
            current_points = []

        elif key == ord('u'):
            if current_points:
                current_points.pop()
            elif slots:
                removed = slots.pop()
                next_id[0] = removed["id"]
                print(f"Removed slot {removed['id']}")

        elif key == ord('s'):
            if len(current_points) >= 3:
                coords = []
                for (x, y) in current_points:
                    coords.extend([x, y])
                slots.append({"id": next_id[0], "coordinates": coords})
            break

        elif key == ord('q'):
            slots = []
            break

    cv2.destroyAllWindows()

    if slots:
        out_path = Path("slots.json")
        out_path.write_text(json.dumps(slots, indent=2))
        print(f"\nSaved {len(slots)} slot(s) to {out_path.resolve()}")
        print("\nPaste this into the parking_slots field in Postman:\n")
        print(json.dumps(slots))
    else:
        print("\nNo slots saved.")


if __name__ == "__main__":
    main()