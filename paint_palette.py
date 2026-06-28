"""Top-of-screen toolbar: brush slots plus a CLEAR button.

The toolbar is fully self-contained — it owns its layout, its hit-testing
and a small dwell timer so the user can't trigger CLEAR by accident.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2


@dataclass
class Brush:
    label: str
    color: Tuple[int, int, int]   # BGR
    size: int
    glow: bool = False
    is_eraser: bool = False


# Order matters — first entry is the brush selected at launch.
DEFAULT_BRUSHES: List[Brush] = [
    Brush("white",     (240, 240, 240), 6),
    Brush("red",       (60, 60, 220),   6),
    Brush("green",     (80, 200, 80),   6),
    Brush("blue",      (220, 130, 50),  6),
    Brush("neon-pink", (200, 80, 255),  7, glow=True),
    Brush("neon-cyan", (255, 220, 80),  7, glow=True),
    Brush("neon-lime", (90, 255, 180),  7, glow=True),
    Brush("eraser",    (0, 0, 0),       30, is_eraser=True),
]


_TOOLBAR_HEIGHT = 72
_CLEAR_BTN_WIDTH = 150
_CLEAR_DWELL_SECONDS = 0.6
_SLOT_INSET = 12


class Toolbar:
    def __init__(self, frame_width: int,
                 brushes: Optional[List[Brush]] = None):
        self._brushes = brushes if brushes is not None else DEFAULT_BRUSHES
        self._frame_width = frame_width
        self._slot_width = (frame_width - _CLEAR_BTN_WIDTH) // len(self._brushes)
        self.active_index = 0
        self._clear_dwell_start: Optional[float] = None

    @property
    def height(self) -> int:
        return _TOOLBAR_HEIGHT

    @property
    def active_brush(self) -> Brush:
        return self._brushes[self.active_index]

    def _slot_rect(self, idx: int) -> Tuple[int, int, int, int]:
        x1 = idx * self._slot_width
        return x1, 0, x1 + self._slot_width, _TOOLBAR_HEIGHT

    def _clear_rect(self) -> Tuple[int, int, int, int]:
        x1 = self._frame_width - _CLEAR_BTN_WIDTH
        return x1, 0, self._frame_width, _TOOLBAR_HEIGHT

    def resolve_hover(self, point: Optional[Tuple[int, int]]) -> Optional[str]:
        """Process a hover point.

        Returns ``"clear"`` once the user has held the CLEAR button long
        enough; returns the label of any brush that was just selected;
        otherwise returns ``None``.
        """
        if point is None or point[1] > _TOOLBAR_HEIGHT:
            self._clear_dwell_start = None
            return None

        x, _ = point
        cx1, _, cx2, _ = self._clear_rect()
        if cx1 <= x <= cx2:
            if self._clear_dwell_start is None:
                self._clear_dwell_start = time.monotonic()
            elapsed = time.monotonic() - self._clear_dwell_start
            if elapsed >= _CLEAR_DWELL_SECONDS:
                self._clear_dwell_start = None
                return "clear"
            return None

        self._clear_dwell_start = None
        for i in range(len(self._brushes)):
            x1, _, x2, _ = self._slot_rect(i)
            if x1 <= x <= x2:
                self.active_index = i
                return self._brushes[i].label
        return None

    def render(self, frame) -> None:
        cv2.rectangle(frame, (0, 0), (self._frame_width, _TOOLBAR_HEIGHT),
                      (28, 28, 28), -1)

        for i, brush in enumerate(self._brushes):
            x1, y1, x2, y2 = self._slot_rect(i)
            inner = (x1 + _SLOT_INSET, y1 + _SLOT_INSET,
                     x2 - _SLOT_INSET, y2 - _SLOT_INSET)

            if brush.is_eraser:
                cv2.rectangle(frame, (inner[0], inner[1]),
                              (inner[2], inner[3]), (200, 200, 200), 2)
                cv2.putText(frame, "ERASE",
                            (inner[0] + 4, inner[3] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (inner[0], inner[1]),
                              (inner[2], inner[3]), brush.color, -1)
                if brush.glow:
                    cv2.rectangle(frame, (inner[0], inner[1]),
                                  (inner[2], inner[3]),
                                  (255, 255, 255), 1)

            if i == self.active_index:
                cv2.rectangle(frame, (x1 + 4, y1 + 4),
                              (x2 - 4, y2 - 4),
                              (255, 255, 255), 2)

        cx1, cy1, cx2, cy2 = self._clear_rect()
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (38, 38, 38), -1)
        cv2.rectangle(frame, (cx1 + 4, cy1 + 4),
                      (cx2 - 4, cy2 - 4), (90, 90, 220), 2)
        cv2.putText(frame, "CLEAR", (cx1 + 28, cy1 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (220, 220, 255), 2, cv2.LINE_AA)

        if self._clear_dwell_start is not None:
            progress = min(1.0,
                           (time.monotonic() - self._clear_dwell_start)
                           / _CLEAR_DWELL_SECONDS)
            end_angle = int(360 * progress)
            center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
            cv2.ellipse(frame, center, (44, 28), 0,
                        -90, -90 + end_angle,
                        (120, 220, 255), 3)
