"""Entry point for the air-drawing app.

Gestures:
  * Index finger only      -> draw with the active brush
  * Index + middle finger  -> hover (no drawing); use this to pick a
                              brush or hold over CLEAR to wipe the canvas
  * Any other shape        -> idle
  * Press 'q'              -> quit
"""

import sys

import cv2

from hand_sensor import HandSensor, Intent
from paint_palette import Toolbar
from stroke_engine import StrokeEngine


WINDOW_NAME = "Air Draw"
CAMERA_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def _open_camera() -> cv2.VideoCapture:
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    if not cam.isOpened():
        print("Could not open the webcam.", file=sys.stderr)
        sys.exit(1)
    return cam


def _draw_status_bar(frame, brush_label: str, intent: Intent) -> None:
    msg = f"{intent.name.lower():<5}  |  brush: {brush_label}  |  q to quit"
    cv2.putText(frame, msg, (12, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)


def run() -> None:
    camera = _open_camera()

    ok, first_frame = camera.read()
    if not ok:
        print("Camera returned no frame.", file=sys.stderr)
        sys.exit(1)
    frame_height, frame_width = first_frame.shape[:2]

    sensor = HandSensor()
    toolbar = Toolbar(frame_width)
    canvas = StrokeEngine(frame_width, frame_height)

    last_tip = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reading = sensor.read(rgb_frame, frame_width, frame_height)

            if reading.intent == Intent.HOVER:
                action = toolbar.resolve_hover(reading.index_tip)
                if action == "clear":
                    canvas.reset()
                last_tip = None

            elif (reading.intent == Intent.DRAW
                  and reading.index_tip is not None
                  and reading.index_tip[1] > toolbar.height):
                if last_tip is not None:
                    canvas.stamp(last_tip, reading.index_tip,
                                 toolbar.active_brush)
                last_tip = reading.index_tip

            else:
                last_tip = None
                toolbar.resolve_hover(None)

            composed = canvas.compose(frame)
            toolbar.render(composed)

            if reading.index_tip is not None:
                if reading.intent == Intent.HOVER:
                    cursor_color = (255, 255, 255)
                else:
                    cursor_color = toolbar.active_brush.color
                cv2.circle(composed, reading.index_tip, 9, cursor_color, 2)

            _draw_status_bar(composed,
                             toolbar.active_brush.label,
                             reading.intent)

            cv2.imshow(WINDOW_NAME, composed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sensor.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
