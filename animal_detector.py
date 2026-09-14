"""On-board thermal animal-presence detector (sketch).

This is "stage 1" of the animal -> RGB-trigger pipeline: a lightweight,
classical (non-ML) detector that runs on the 8-bit AGC'd thermal frames
the P1 already produces. No raw radiometric data is required - useful
right now since none has been saved yet - though see the note at the
bottom of this file for why raw data would make it more robust later.

It is deliberately NOT a neural network. At 160x120 the animal signature
in the example clip is a compact blob a few percent of the frame, clearly
distinguishable from the slowly-varying background by simple frame
differencing - a full ML detector would be solving a problem this small
input doesn't need solved, at a real cost of extra CPU, RAM, and (for a
battery-powered field unit) power. Start here; only add a learned model
if false positives from real deployments turn out to need it.

Algorithm:
  1. Mask out the burned-in timestamp strip. It changes every single
     frame (down to the millisecond digit) - the worst possible input
     for a frame-differencing detector, and the #1 source of false
     positives before this was fixed (see calibration notes below).
  2. Maintain a slow running-average background model per pixel.
  3. Threshold |frame - background| to find pixels that changed.
  4. Connected-component blob search on that mask, filtered by size.
  5. Reject frames with an implausible number of simultaneous blobs -
     a sudden sun/cloud/AGC-wide brightness shift looks like many small
     blobs appearing at once; a real animal looks like one (occasionally
     two, e.g. head separated from body by cooler fur).
  6. Require a candidate blob to keep reappearing near the same spot
     across several consecutive frames before calling it a confirmed
     detection. This is what actually separates a real multi-second
     animal crossing from single-frame sensor noise.

Feed it one frame at a time via process() - the same code path works
both offline against a recorded video (see test_on_video.py) and,
later, inline in record_p1_segmented.py's acquisition loop on the
Raspberry Pi: call detector.process(ir_brightness) right where that
loop already calls segment.write(ir_brightness, thermal_raw), and treat
result.confirmed as the RGB-camera trigger signal.

Calibration notes (from R:\\Share\\Alessandro\\recordings_field_test_20260819-20260826\\
thermal_20260826_041155.avi, 160x120 @ 10fps, un-rotated):
  - The default parameters below were tuned against that file: the known
    animal occurrence around 5:26 was confirmed continuously from 326.0s
    to 373.6s (about 48s - longer than the single moment originally
    noted), with ZERO false-positive confirmations across the rest of
    the ~400s of footage checked.
  - TIMESTAMP_FRAC_H must cover the *entire* burned-in timestamp width,
    not just where the text starts - at 160px wide the full
    "YYYY-MM-DD HH:MM:SS.mmm" string spans nearly the whole frame.
    Re-check this if you change --rotate-degrees or run the P3 (256x192).
  - If you point this at a different camera position, run
    test_on_video.py against a clip from it and look at the annotated
    output: any patch of the frame that is a static hot fixture (a
    bracket, a sunlit rock, part of the housing intruding into frame)
    settles into the background model within a few seconds and stops
    contributing - it does NOT need its own mask, unlike the timestamp,
    which is adversarial to background subtraction by construction
    (it changes every single frame, so it can never "settle").
"""

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    timestamp_frac_h: float = 0.20  # bottom fraction of frame height to mask (full width)
    bg_alpha: float = 0.02  # background adaptation rate; higher = adapts faster, less sensitive
    diff_thresh: int = 15  # |frame - background| threshold (0-255) to call a pixel "changed"
    min_area: int = 6  # smallest blob (pixels) to consider, at native P1 160x120 resolution
    max_area: int = 400  # largest blob to consider; bigger is a lighting change, not an animal
    max_simultaneous_blobs: int = 8  # more than this at once => global change, reject the frame
    persist_window: int = 5  # frames to look back over when confirming a detection
    persist_min_hits: int = 3  # of those, how many must have had a nearby blob
    persist_max_dist: float = 12.0  # pixels; how close counts as "the same" blob frame-to-frame


@dataclass
class DetectionResult:
    frame_index: int
    candidate_blobs: list = field(default_factory=list)  # [(x, y, area), ...] after size filtering
    confirmed: bool = False  # True once the persistence criteria are met -> trigger the RGB camera
    confirmed_centroid: tuple | None = None


class ThermalAnimalDetector:
    """Stateful, single-frame-at-a-time detector. One instance per camera stream."""

    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self._bg = None
        self._recent = deque(maxlen=self.cfg.persist_window)
        self._open_kernel = np.ones((2, 2), np.uint8)
        self._frame_index = -1

    def reset(self):
        """Call after a gap in acquisition (e.g. a new recording segment) so the
        background model doesn't treat a stale frame as ground truth."""
        self._bg = None
        self._recent.clear()
        self._frame_index = -1

    def process(self, gray_frame: np.ndarray) -> DetectionResult:
        """gray_frame: 2D uint8 array - a single-channel thermal IR-brightness frame,
        i.e. exactly what record_p1_segmented.py gets back as ir_brightness."""
        self._frame_index += 1
        h, w = gray_frame.shape[:2]
        ts_y = int(h * (1 - self.cfg.timestamp_frac_h))

        gray = gray_frame.astype(np.float32)
        if self._bg is None:
            self._bg = gray.copy()

        diff = cv2.absdiff(gray, self._bg).astype(np.uint8)
        diff[ts_y:, :] = 0  # the timestamp region never gets to drive a detection

        _, mask = cv2.threshold(diff, self.cfg.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)

        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if self.cfg.min_area <= area <= self.cfg.max_area:
                cx, cy = centroids[i]
                blobs.append((float(cx), float(cy), area))

        usable_blobs = blobs if len(blobs) <= self.cfg.max_simultaneous_blobs else []
        self._recent.append(usable_blobs)

        # Update the background AFTER computing the diff. bg_alpha is slow enough
        # relative to how briefly an animal is in frame that this doesn't erase it.
        cv2.accumulateWeighted(gray, self._bg, self.cfg.bg_alpha)

        confirmed = False
        confirmed_centroid = None
        for cx, cy, _area in usable_blobs:
            hits = sum(
                1
                for past in self._recent
                if any(
                    np.hypot(cx - px, cy - py) <= self.cfg.persist_max_dist
                    for px, py, _pa in past
                )
            )
            if hits >= self.cfg.persist_min_hits:
                confirmed = True
                confirmed_centroid = (cx, cy)
                break

        return DetectionResult(
            frame_index=self._frame_index,
            candidate_blobs=usable_blobs,
            confirmed=confirmed,
            confirmed_centroid=confirmed_centroid,
        )
