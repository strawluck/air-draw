"""Persistent drawing surface.

Strokes are split across two buffers: a crisp `core` layer and a soft
`glow` layer that is Gaussian-blurred at composite time. Together they
let plain colours and neon glow brushes coexist on the same canvas.
"""

import cv2
import numpy as np

from paint_palette import Brush


_GLOW_BLUR_KERNEL = (31, 31)
_GLOW_HALO_PADDING = 14       # extra px of width baked into the halo line
_GLOW_BLEND_STRENGTH = 0.75


class StrokeEngine:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._core = np.zeros((height, width, 3), dtype=np.uint8)
        self._glow = np.zeros((height, width, 3), dtype=np.uint8)
        self._mask = np.zeros((height, width), dtype=np.uint8)

    def stamp(self, prev_point, point, brush: Brush) -> None:
        if brush.is_eraser:
            # Black writes on both colour buffers actually clear pixels,
            # and the mask is cleared so the core layer no longer composites.
            cv2.line(self._core, prev_point, point, (0, 0, 0), brush.size)
            cv2.line(self._glow, prev_point, point, (0, 0, 0),
                     brush.size + _GLOW_HALO_PADDING)
            cv2.line(self._mask, prev_point, point, 0, brush.size)
            return

        if brush.glow:
            cv2.line(self._glow, prev_point, point, brush.color,
                     brush.size + _GLOW_HALO_PADDING)
        cv2.line(self._core, prev_point, point, brush.color, brush.size)
        cv2.line(self._mask, prev_point, point, 255, brush.size)

    def reset(self) -> None:
        self._core[:] = 0
        self._glow[:] = 0
        self._mask[:] = 0

    def compose(self, base_frame):
        out = base_frame.copy()

        if np.any(self._glow):
            blurred = cv2.GaussianBlur(self._glow, _GLOW_BLUR_KERNEL, 0)
            out = cv2.add(out, (blurred.astype(np.float32)
                                * _GLOW_BLEND_STRENGTH).astype(np.uint8))

        if np.any(self._mask):
            mask3 = (cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR)
                     .astype(np.float32) / 255.0)
            out = (out.astype(np.float32) * (1.0 - mask3)
                   + self._core.astype(np.float32) * mask3).astype(np.uint8)

        return out
