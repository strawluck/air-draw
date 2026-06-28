# Air Draw

Draw in the air with your index finger. A webcam tracks your hand in real
time and turns finger movement into strokes on screen — pick colours,
switch to an eraser, or wipe the canvas, all with gestures.

## How it works

Each frame from the webcam runs through a short pipeline:

1. **Hand tracking** — MediaPipe's `HandLandmarker` returns 21 3D
   landmarks for the hand. From the relative position of each fingertip
   to its knuckle the app decides which fingers are extended.
2. **Intent** — the finger pattern is mapped to one of three states:
   - index finger only → **draw**
   - index + middle → **hover** (pick a tool or hold over CLEAR)
   - anything else → **idle** (pen lifted)
3. **Stroke engine** — when drawing, a line is stamped between the
   previous and current index-tip position. Strokes are kept in two
   layers: a crisp `core` layer and a `glow` layer that is
   Gaussian-blurred at composite time, so the neon brushes get a soft
   bloom while plain colours stay sharp.
4. **Composite** — every frame the canvas is composed onto the mirrored
   webcam feed and the toolbar is drawn on top.

## Tech

- **Python 3.9+**
- **MediaPipe** (Tasks API — `HandLandmarker`) for hand pose
- **OpenCV** for webcam I/O, drawing primitives, and Gaussian blur
- **NumPy** for the canvas buffers

The hand-landmarker model (`hand_landmarker.task`, ~7 MB) downloads
automatically the first time you run the app and is cached next to the
source.

## Requirements

- Python 3.9 or newer
- A webcam
- macOS, Linux, or Windows (developed on macOS arm64)
- Internet on the **first** launch only (to fetch the model)

## Install & run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On macOS, the first launch will prompt for camera access — approve it,
then re-run.

## Controls

| Gesture / key      | What it does                                          |
| ------------------ | ----------------------------------------------------- |
| Index finger only  | Draw with the active brush                            |
| Index + middle     | Hover — slide over a tool to switch                   |
| Hover on **CLEAR** | Hold ~0.6 s (cyan arc fills) to wipe the canvas       |
| Hand down / fist   | Pen lifts                                             |
| `q`                | Quit                                                  |

## Project layout

```
air-draw/
├── main.py             # webcam loop, owns the cv2 window
├── hand_sensor.py      # MediaPipe wrapper -> HandReading per frame
├── paint_palette.py    # Brush dataclass + Toolbar (slots + CLEAR button)
├── stroke_engine.py    # two-layer canvas with glow compositing
├── requirements.txt
└── hand_landmarker.task  # downloaded on first run, git-ignored
```

## Tuning

- Brush sizes and the colour kit: `DEFAULT_BRUSHES` in `paint_palette.py`
- Glow softness / strength: `_GLOW_BLUR_KERNEL` and
  `_GLOW_BLEND_STRENGTH` at the top of `stroke_engine.py`
- Detection sensitivity: `detection_confidence` /
  `tracking_confidence` passed to `HandSensor`
