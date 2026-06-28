"""Hand detection wrapper around MediaPipe's Tasks API.

Each call to `read` returns a `HandReading` summarising what the hand is
doing this frame: where the index/middle fingertips are and which gesture
the user is making (draw / hover / idle).

On first launch the hand-landmarker model is downloaded next to this
file (~7 MB) so the rest of the pipeline can run fully offline.
"""

import os
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


class Intent(Enum):
    IDLE = auto()
    DRAW = auto()
    HOVER = auto()


@dataclass
class HandReading:
    intent: Intent
    index_tip: Optional[Tuple[int, int]]
    middle_tip: Optional[Tuple[int, int]]
    fingers_extended: Tuple[bool, bool, bool, bool]


_LM_INDEX_TIP = 8
_LM_INDEX_PIP = 6
_LM_MIDDLE_TIP = 12
_LM_MIDDLE_PIP = 10
_LM_RING_TIP = 16
_LM_RING_PIP = 14
_LM_PINKY_TIP = 20
_LM_PINKY_PIP = 18

_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, _MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"Downloading {_MODEL_FILENAME} (~7 MB) ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Model ready.")
    return path


class HandSensor:
    def __init__(self, detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)
        # Tasks API requires monotonically increasing timestamps in VIDEO
        # mode — anchor them at construction.
        self._stream_origin = time.monotonic()

    def read(self, rgb_frame, width: int, height: int) -> HandReading:
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - self._stream_origin) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return HandReading(Intent.IDLE, None, None,
                               (False, False, False, False))

        landmarks = result.hand_landmarks[0]

        def to_pixel(idx: int) -> Tuple[int, int]:
            return (int(landmarks[idx].x * width),
                    int(landmarks[idx].y * height))

        def is_extended(tip_idx: int, pip_idx: int) -> bool:
            return landmarks[tip_idx].y < landmarks[pip_idx].y

        index_up = is_extended(_LM_INDEX_TIP, _LM_INDEX_PIP)
        middle_up = is_extended(_LM_MIDDLE_TIP, _LM_MIDDLE_PIP)
        ring_up = is_extended(_LM_RING_TIP, _LM_RING_PIP)
        pinky_up = is_extended(_LM_PINKY_TIP, _LM_PINKY_PIP)

        index_tip = to_pixel(_LM_INDEX_TIP)
        middle_tip = to_pixel(_LM_MIDDLE_TIP)

        if index_up and middle_up and not ring_up:
            intent = Intent.HOVER
        elif index_up and not middle_up and not ring_up:
            intent = Intent.DRAW
        else:
            intent = Intent.IDLE

        return HandReading(
            intent=intent,
            index_tip=index_tip,
            middle_tip=middle_tip,
            fingers_extended=(index_up, middle_up, ring_up, pinky_up),
        )

    def close(self) -> None:
        self._detector.close()
