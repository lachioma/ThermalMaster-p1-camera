"""Continuous segmented thermal recording for the Thermal Master P1.

Built for unattended, multi-day field deployments on a battery-powered
Raspberry Pi:

- Recording is split into fixed-length segment files (``--segment-seconds``)
  instead of one giant file, so a power loss or crash only costs the
  in-progress segment, not the whole recording.
- Every segment filename carries a start timestamp, so files sort
  chronologically and a restart never overwrites previous data.
- Video defaults to AVI/XVID, which trades off file size against how much of
  a segment truncated mid-write (e.g. by a power loss) survives; see
  ``--format`` below for the other options (avi-mjpg, mp4).
- Each frame has a ``YYYY-MM-DD HH:MM:SS.mmm`` timestamp burned into the
  bottom-left corner (font size is computed per-frame so it always fits) and,
  by default, is rotated 90 degrees (``--rotate-degrees``) to match how the
  camera is mounted.
- Saving the raw 16-bit radiometric data is optional (``--no-raw``). When
  enabled it is streamed frame-by-frame to a flat ``.dat`` binary file
  instead of being buffered in memory and written with ``np.save`` at the
  end of the segment: a 10-minute segment at 25 fps would otherwise hold
  ~550 MB of raw frames in RAM, and buffering also means a power loss
  mid-segment would silently lose the entire segment's raw data instead
  of just the last fraction of a second. Reload it with:
  ``np.fromfile(path, dtype=np.uint16).reshape(-1, HEIGHT, WIDTH)``
  (actual shape/dtype, which depend on rotation, are also written to the
  segment's JSON sidecar). The raw data is rotated the same way as the
  video but never has the timestamp burned in, so temperature values stay
  unmodified. At full rate this is large - roughly 33 GB/day at 160x120,
  10 fps (fixed cost: resolution x 2 bytes x fps x 86400s, independent of
  scene content, unlike the compressed video). If you don't need every
  frame's raw data, ``--raw-every-n`` keeps only every Nth frame's raw data
  (the 1st frame of every segment is always kept), cutting that size by
  roughly N while the video stays at full frame rate.
- The video normally shows the camera's own hardware-AGC'd brightness
  (ir_brightness), which re-normalizes every frame to whatever's in the
  scene - great for a live viewer, but it means the same pixel value can
  represent different temperatures at different times, hiding a slow sensor
  drift. Passing ``--temp-min-c``/``--temp-max-c`` switches the video to a
  fixed absolute-temperature-to-brightness mapping instead (derived from the
  raw data, which is always read regardless of ``--no-raw``), so a real
  drift shows up directly as the video trending toward one end of the
  brightness range. Values outside the range saturate to black/white.
- Per-frame USB marker errors are skipped instead of crashing the whole
  recording, and the camera is automatically reconnected if the USB link
  drops entirely - but only for up to ``--max-reconnect-seconds``: while
  reconnecting, --segment-seconds/--duration/disk-space are not checked at
  all, so retrying forever can silently freeze the whole recording for as
  long as the camera stays unreachable. Past that limit the process exits
  instead, so a supervisor like systemd (Restart=always) can relaunch it -
  a visible, timestamped restart rather than a silent multi-day stall.
- A JSON sidecar per segment plus a running ``recording_log.csv`` record
  frame counts, achieved fps, and dropped/mismatched frames for later QA.
- Stops cleanly on SIGTERM (``systemctl stop``) as well as Ctrl+C, and can
  stop itself before the SD card / disk fills up (``--min-free-mb``).

Arguments:

    --duration N              Total recording duration in seconds.
                               Default: 0 (run until stopped by Ctrl+C,
                               SIGTERM, or disk space running low).
    --segment-seconds N       Length of each output file in seconds.
                               Default: 600 (10 minutes). Use something
                               small like 10-60 for debugging.
    --outdir PATH             Directory for segment files, JSON sidecars,
                               and recording_log.csv. Default: "recordings".
    --prefix NAME             Filename prefix for each segment; a
                               "_YYYYmmdd_HHMMSS" timestamp is appended
                               automatically. Default: "thermal".
    --format {avi,avi-mjpg,mp4}
                               Video container/codec (size vs. crash-safety
                               tradeoff, see above). Default: "avi" (XVID).
    --fps N                   Target frames per second: frames from the
                               camera (which streams at its own fixed
                               hardware rate, ~25-27 fps for the P1,
                               regardless of this setting) are thinned down
                               to this rate before being written out, so a
                               lower value actually reduces the number of
                               frames saved per real second instead of just
                               relabeling the video's playback rate.
                               Default: 25.0.
    --rotate-degrees {0,90,180,270}
                               Clockwise rotation applied to both the video
                               and the raw data before saving. Default: 90.
    --temp-min-c N             Bottom of a fixed Celsius range mapped to the
                               video's 0..255 brightness, replacing the
                               camera's hardware AGC. Must be given with
                               --temp-max-c. Default: unset (use AGC).
    --temp-max-c N             Top of that fixed Celsius range. Default:
                               unset (use AGC).
    --no-raw                  Disable saving the 16-bit raw radiometric
                               data (_raw.dat). Default: off (raw is saved).
    --raw-every-n N           Persist only every Nth video frame's raw data
                               instead of every frame (the 1st frame of
                               every segment is always kept). No effect if
                               --no-raw is set. Default: 1 (every frame).
    --no-timestamps           Disable the per-frame timestamp log
                               (_timestamps.txt). Default: off (saved).
                               Independent of the burned-in video timestamp,
                               which is always drawn.
    --min-free-mb N           Stop recording once free disk space on
                               --outdir drops below this many MB.
                               Default: 500.0.
    --connect-retry-interval N
                               Seconds to wait between camera
                               connect/reconnect attempts. Default: 5.0.
    --max-reconnect-seconds N Give up and exit the process after this many
                               seconds of failing to (re)connect, instead of
                               retrying forever (which silently ignores
                               --segment-seconds/--duration/disk-space the
                               whole time it's stuck). 0 = retry forever.
                               Default: 300.0 (5 minutes).

Example:

    python record_p1_segmented.py --segment-seconds 600 --no-raw \\
        --outdir /media/usb/recordings
"""

import argparse
import csv
import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import usb.core

from p3_camera import (
    FrameMarkerMismatchError,
    Model,
    P3Camera,
    celsius_to_raw,
    get_model_config,
)

# Native resolution of the P1 sensor's IR/thermal output.
WIDTH, HEIGHT = 160, 120

FOURCC_BY_FORMAT = {
    "avi": "XVID",
    "avi-mjpg": "MJPG",
    "mp4": "mp4v",
}

# cv2.rotate() codes, keyed by clockwise rotation in degrees.
ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def _rotate(img, degrees):
    code = ROTATE_CODES[degrees]
    return img if code is None else cv2.rotate(img, code)


def _map_temperature_range(thermal_raw, raw_min, raw_max):
    """Render raw 16-bit temperature data to 8-bit using a *fixed* range.

    Unlike the camera's own hardware AGC (ir_brightness), which re-normalizes
    every frame to whatever range is currently in the scene, this always maps
    the same [raw_min, raw_max] window to [0, 255]. A pixel value therefore
    means the same absolute temperature in every frame of every segment, so a
    genuine sensor drift over hours/days shows up directly as the video
    trending toward one end of the brightness range - the hardware AGC would
    otherwise silently re-stretch it away. The tradeoff: anything outside
    [raw_min, raw_max] saturates to 0 or 255, so pick a range that comfortably
    covers the scene's real extremes.
    """
    scaled = (thermal_raw.astype(np.float32) - raw_min) * (255.0 / (raw_max - raw_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)


_TIMESTAMP_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TIMESTAMP_MARGIN = 2


def _draw_timestamp(bgr_frame, ts):
    """Burn a date/time stamp (with milliseconds) into the bottom-left corner.

    The P1's native frame is only 120-160 px on a side, so the font scale is
    computed from the frame width rather than fixed, guaranteeing the text
    (including milliseconds) never grows wider or taller than the frame
    itself, at any --rotate-degrees setting.
    """
    dt = datetime.fromtimestamp(ts)
    text = dt.strftime("%Y-%m-%d %H:%M:%S") + f".{dt.microsecond // 1000:03d}"
    frame_h, frame_w = bgr_frame.shape[:2]
    max_width = frame_w - 2 * _TIMESTAMP_MARGIN
    max_height = frame_h - 2 * _TIMESTAMP_MARGIN

    # Start from a scale that comfortably fits typical frames, then shrink
    # further if this particular frame size still can't fit the text.
    scale = 0.35
    (text_w, text_h), baseline = cv2.getTextSize(text, _TIMESTAMP_FONT, scale, 1)
    if text_w > max_width or (text_h + baseline) > max_height:
        scale *= min(max_width / text_w, max_height / (text_h + baseline))
        scale = max(scale, 0.1)  # never shrink to the point of being invisible
        (text_w, text_h), baseline = cv2.getTextSize(text, _TIMESTAMP_FONT, scale, 1)

    org = (_TIMESTAMP_MARGIN, frame_h - _TIMESTAMP_MARGIN - baseline)
    # Black outline first, then white fill, so the text stays legible
    # regardless of how bright/dark the thermal image is underneath it.
    cv2.putText(bgr_frame, text, org, _TIMESTAMP_FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bgr_frame, text, org, _TIMESTAMP_FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


class SegmentWriter:
    """Owns the video writer (and optional raw/timestamp buffers) for one segment file."""

    def __init__(
        self,
        outdir,
        prefix,
        fmt,
        fps,
        save_raw,
        save_timestamps,
        rotate_degrees,
        temp_range_c=None,
        raw_every_n=1,
    ):
        self.outdir = outdir
        self.prefix = prefix
        self.fmt = fmt
        self.fps = fps
        self.save_raw = save_raw
        self.save_timestamps = save_timestamps
        self.rotate_degrees = rotate_degrees
        # Only every raw_every_n-th video frame's raw data is persisted (the
        # 1st frame of every segment always is), to cut the raw stream's size
        # when full temporal resolution isn't needed for it. 1 = every frame.
        self.raw_every_n = raw_every_n
        # (Tmin, Tmax) in Celsius for a fixed absolute-temperature display
        # mapping, or None to use the camera's own hardware-AGC'd
        # ir_brightness. Converted once to raw sensor units (an affine
        # transform of Celsius) since write() runs per-frame.
        self.temp_range_c = temp_range_c
        self.temp_raw_bounds = (
            (celsius_to_raw(temp_range_c[0]), celsius_to_raw(temp_range_c[1]))
            if temp_range_c is not None
            else None
        )
        if rotate_degrees in (90, 270):
            self.out_width, self.out_height = HEIGHT, WIDTH
        else:
            self.out_width, self.out_height = WIDTH, HEIGHT

        self.video_writer = None
        self.video_path = None
        self.raw_path = None
        self.ts_path = None
        self.meta_path = None
        self._raw_fh = None
        self._ts_fh = None
        self.frame_count = 0
        self.raw_frame_count = 0
        self.marker_mismatches = 0
        self.start_time = None

    def open(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.prefix}_{stamp}"
        self.video_path = self.outdir / f"{base}.{self.fmt}"
        self.raw_path = self.outdir / f"{base}_raw.dat"
        self.ts_path = self.outdir / f"{base}_timestamps.txt"
        self.meta_path = self.outdir / f"{base}.json"

        fourcc = cv2.VideoWriter_fourcc(*FOURCC_BY_FORMAT[self.fmt])
        # Written as color (grayscale replicated across BGR channels) since
        # single-channel output is unreliable across OpenCV/ffmpeg backends
        # for MP4; this keeps the file directly playable in VLC etc.
        self.video_writer = cv2.VideoWriter(
            str(self.video_path), fourcc, self.fps, (self.out_width, self.out_height), True
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.video_path}")

        # Raw radiometric data and per-frame timestamps are streamed to disk
        # frame-by-frame (rather than buffered and written at close()) to
        # keep memory bounded and to limit how much is lost on a mid-segment
        # power loss to the last unflushed frame instead of the whole segment.
        self._raw_fh = open(self.raw_path, "wb") if self.save_raw else None
        self._ts_fh = open(self.ts_path, "w") if self.save_timestamps else None

        self.frame_count = 0
        self.raw_frame_count = 0
        self.marker_mismatches = 0
        self.start_time = time.time()

        extra = []
        if self.save_raw:
            extra.append(self.raw_path.name)
        if self.save_timestamps:
            extra.append(self.ts_path.name)
        suffix = f" (+ {', '.join(extra)})" if extra else ""
        print(f"[segment] recording -> {self.video_path.name}{suffix}")

    def write(self, ir_brightness, thermal_raw):
        now = time.time()

        if self.temp_raw_bounds is not None:
            raw_min, raw_max = self.temp_raw_bounds
            display = _map_temperature_range(thermal_raw, raw_min, raw_max)
        else:
            display = ir_brightness

        frame = _rotate(display, self.rotate_degrees)
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        _draw_timestamp(bgr, now)
        self.video_writer.write(bgr)

        if self._raw_fh is not None and self.frame_count % self.raw_every_n == 0:
            # Rotated the same way as the video so pixel (r, c) still refers
            # to the same physical spot in both files, but never stamped:
            # this must stay the camera's true temperature reading.
            self._raw_fh.write(_rotate(thermal_raw, self.rotate_degrees).tobytes())
            self._raw_fh.flush()
            self.raw_frame_count += 1
        if self._ts_fh is not None:
            self._ts_fh.write(f"{now!r}\n")
            self._ts_fh.flush()
        self.frame_count += 1

    def close(self):
        end_time = time.time()
        self.video_writer.release()
        if self._raw_fh is not None:
            self._raw_fh.close()
        if self._ts_fh is not None:
            self._ts_fh.close()

        duration = end_time - self.start_time
        meta = {
            "video_file": self.video_path.name,
            "raw_file": self.raw_path.name if self.save_raw else None,
            "raw_dtype": "uint16" if self.save_raw else None,
            "raw_every_n": self.raw_every_n if self.save_raw else None,
            "raw_shape": [self.raw_frame_count, self.out_height, self.out_width]
            if self.save_raw
            else None,
            "timestamps_file": self.ts_path.name if self.save_timestamps else None,
            "rotate_degrees": self.rotate_degrees,
            "video_source": "fixed_temp_range" if self.temp_range_c else "hardware_agc",
            "temp_min_c": self.temp_range_c[0] if self.temp_range_c else None,
            "temp_max_c": self.temp_range_c[1] if self.temp_range_c else None,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": round(duration, 2),
            "frame_count": self.frame_count,
            "target_fps": self.fps,
            "achieved_fps": round(self.frame_count / duration, 2) if duration > 0 else 0.0,
            "marker_mismatches": self.marker_mismatches,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))
        print(
            f"[segment] closed {self.video_path.name}: {self.frame_count} frames, "
            f"{meta['achieved_fps']:.1f} fps, {self.marker_mismatches} marker mismatches"
        )
        return meta


def append_run_log(log_path, meta):
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(meta)


def free_space_mb(path):
    return shutil.disk_usage(path).free / (1024 * 1024)


def connect_camera(retry_interval, max_reconnect_seconds=300.0):
    """Connect + init + start streaming, retrying on failure.

    On an unattended field device nobody is around to restart the script if
    the camera enumerates a moment late after power-on, so this retries
    rather than giving up immediately. But it must not retry *forever*:
    while this loop is running, the caller's --segment-seconds/--duration/
    disk-space checks are not being evaluated at all, so an unbounded retry
    here can silently freeze the whole recording (a real deployment once
    lost ~2.8 days this way, stuck retrying every few seconds with nothing
    written and no visible sign anything was wrong). Past
    max_reconnect_seconds (0 = retry forever, the old behavior) this gives
    up and returns None, so the caller can exit the process - letting a
    supervisor like systemd (Restart=always) relaunch it, which turns a
    silent multi-day stall into a series of visible, timestamped restarts.
    """
    attempt = 0
    deadline = time.time() + max_reconnect_seconds if max_reconnect_seconds > 0 else None
    while not _stop_requested:
        if deadline is not None and time.time() >= deadline:
            print(
                f"[connect] giving up after {max_reconnect_seconds}s without a camera."
            )
            return None
        attempt += 1
        try:
            config = get_model_config(Model.P1)
            camera = P3Camera(config=config)
            camera.connect()
            name, version = camera.init()
            camera.start_streaming()
            print(f"Connected to {name} (firmware {version}).")
            return camera
        except Exception as e:
            print(f"[connect] attempt {attempt} failed ({e!r}); retrying in {retry_interval}s")
            time.sleep(retry_interval)
    return None


def record(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_log_path = outdir / "recording_log.csv"
    save_raw = not args.no_raw
    save_timestamps = not args.no_timestamps
    temp_range_c = (
        (args.temp_min_c, args.temp_max_c) if args.temp_min_c is not None else None
    )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    camera = connect_camera(args.connect_retry_interval, args.max_reconnect_seconds)
    if camera is None:
        print("Could not connect to the camera; exiting so a process supervisor "
              "(e.g. systemd) can restart cleanly, or stop was requested first.")
        return

    overall_start = time.time()
    consecutive_errors = 0
    ERROR_RECONNECT_THRESHOLD = 20
    # The camera streams at its own fixed hardware rate (~25-27 fps for the
    # P1) regardless of --fps; read_frame_both() must still be called every
    # loop to drain that stream, but only frames that land on this schedule
    # get written out, so --fps actually thins the output instead of just
    # relabeling the container's declared playback rate.
    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    next_frame_due = time.time()

    try:
        while not _stop_requested:
            if args.duration and (time.time() - overall_start) >= args.duration:
                print("Requested total duration reached.")
                break

            if free_space_mb(outdir) < args.min_free_mb:
                print(
                    f"Free space below {args.min_free_mb} MB; stopping recording "
                    "to avoid filling the disk."
                )
                break

            segment = SegmentWriter(
                outdir,
                args.prefix,
                args.format,
                args.fps,
                save_raw,
                save_timestamps,
                args.rotate_degrees,
                temp_range_c,
                args.raw_every_n,
            )
            segment.open()
            segment_deadline = time.time() + args.segment_seconds
            next_frame_due = time.time()
            give_up = False

            while (
                not _stop_requested
                and time.time() < segment_deadline
                and not (args.duration and (time.time() - overall_start) >= args.duration)
            ):
                try:
                    ir_brightness, thermal_raw = camera.read_frame_both()
                except FrameMarkerMismatchError:
                    segment.marker_mismatches += 1
                    ir_brightness, thermal_raw = None, None
                except usb.core.USBError as e:
                    print(f"[usb] read error: {e!r}")
                    ir_brightness, thermal_raw = None, None

                if ir_brightness is not None and thermal_raw is not None:
                    consecutive_errors = 0
                    now = time.time()
                    if frame_interval <= 0 or now >= next_frame_due:
                        segment.write(ir_brightness, thermal_raw)
                        next_frame_due += frame_interval
                        if next_frame_due < now:
                            # Fell behind (e.g. right after a reconnect stall);
                            # resync instead of bursting out queued frames
                            # back-to-back to "catch up".
                            next_frame_due = now + frame_interval
                    continue

                # Frame missing or corrupted (exception above, or a silent
                # None/None return from read_frame_both). Count it and, past
                # the threshold, assume the USB link is wedged and reconnect.
                consecutive_errors += 1
                if consecutive_errors >= ERROR_RECONNECT_THRESHOLD:
                    print(
                        f"[usb] {consecutive_errors} consecutive frame errors; "
                        "reconnecting to camera..."
                    )
                    try:
                        camera.stop_streaming()
                        camera.disconnect()
                    except Exception:
                        pass
                    camera = connect_camera(args.connect_retry_interval, args.max_reconnect_seconds)
                    consecutive_errors = 0
                    if camera is None:
                        give_up = True
                        break

            meta = segment.close()
            append_run_log(run_log_path, meta)
            if give_up:
                print(
                    "Could not reconnect within --max-reconnect-seconds; exiting so a "
                    "process supervisor (e.g. systemd) can restart cleanly."
                )
                break

    finally:
        try:
            camera.stop_streaming()
            camera.disconnect()
        except Exception:
            pass
        print("Camera disconnected. Recording stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Segmented, unattended thermal video recording for the Thermal Master P1."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Total recording duration in seconds. 0 (default) means run until "
        "stopped (Ctrl+C, SIGTERM, or disk full).",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=600,
        help="Length of each output file in seconds (default: 600 = 10 minutes). "
        "Use a small value like 10-60 for debugging.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="recordings",
        help="Directory to write segment files, sidecar metadata, and the run log into.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="thermal",
        help="Filename prefix for each segment (timestamp is appended automatically).",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FOURCC_BY_FORMAT),
        default="avi",
        help="Video container/codec, trading off file size against how much a segment "
        "truncated mid-write (e.g. by a power loss) survives. avi (XVID, default): "
        "inter-frame compressed, small files; a truncated segment still plays back "
        "to shortly before the cut (loses at most the last unfinished group of "
        "frames). avi-mjpg: every frame is an independent JPEG, so a truncated "
        "segment plays back to the exact last complete frame, at the cost of "
        "noticeably larger files (no inter-frame compression). mp4 (mp4v): "
        "smallest files, but its index is only written when the segment closes, "
        "so a truncated segment is typically unplayable.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target frames per second. The camera streams at its own fixed hardware "
        "rate regardless of this value; frames are thinned down to --fps before being "
        "written, so e.g. --fps 2 actually saves ~2 frames per real second (not just "
        "25 fps of frames relabeled to play back at 2 fps).",
    )
    parser.add_argument(
        "--rotate-degrees",
        type=int,
        choices=sorted(ROTATE_CODES),
        default=90,
        help="Rotate each frame clockwise by this many degrees before saving "
        "(applies to both the video and the raw data). Use 0 to disable.",
    )
    parser.add_argument(
        "--temp-min-c",
        type=float,
        default=None,
        help="Bottom of a fixed absolute-temperature range (Celsius) to map to the video's "
        "0..255 brightness, replacing the camera's own hardware AGC (ir_brightness). "
        "The AGC re-normalizes every frame to whatever's currently in the scene, so the "
        "same pixel value can mean different temperatures at different times - useless "
        "for spotting a slow sensor drift, since the AGC just stretches it away. A fixed "
        "range has no such adaptation: a real drift shows up directly as the video "
        "trending toward one end of the brightness range. Anything outside "
        "[--temp-min-c, --temp-max-c] saturates to black/white, so pick a range that "
        "comfortably covers the scene's real extremes. Requires --temp-max-c too; only "
        "affects the video, never the raw data.",
    )
    parser.add_argument(
        "--temp-max-c",
        type=float,
        default=None,
        help="Top of the fixed absolute-temperature range (Celsius). See --temp-min-c.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not save the 16-bit radiometric raw data (_raw.dat) alongside the video.",
    )
    parser.add_argument(
        "--raw-every-n",
        type=int,
        default=1,
        help="Only persist every Nth video frame's raw data (the 1st frame of every "
        "segment is always kept), to cut the raw stream's size when full temporal "
        "resolution isn't needed for it - e.g. only enough to periodically check for "
        "sensor drift. The raw file's frame count/shape and this value are recorded in "
        "the segment's JSON sidecar so you can map a raw sample back to its video frame "
        "index (raw sample k = video frame k * --raw-every-n) and, via the timestamps "
        "file, its wall-clock time. No effect if --no-raw is set. Default: 1 (every frame).",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not save a per-frame wall-clock timestamp file (_timestamps.txt) per segment.",
    )
    parser.add_argument(
        "--min-free-mb",
        type=float,
        default=500.0,
        help="Stop recording once free disk space drops below this many MB.",
    )
    parser.add_argument(
        "--connect-retry-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between camera connection/reconnection attempts.",
    )
    parser.add_argument(
        "--max-reconnect-seconds",
        type=float,
        default=300.0,
        help="Give up and exit the process after this many seconds of failing to "
        "(re)connect to the camera, instead of retrying forever. While retrying, "
        "--segment-seconds/--duration/disk-space checks are not evaluated at all, so "
        "an unbounded retry can silently freeze the whole recording for as long as the "
        "camera stays unreachable. Exiting instead lets a process supervisor like "
        "systemd (Restart=always) relaunch cleanly, turning a silent multi-day stall "
        "into visible, timestamped restarts in the logs. 0 = retry forever (old "
        "behavior). Default: 300 (5 minutes).",
    )

    args = parser.parse_args()
    if (args.temp_min_c is None) != (args.temp_max_c is None):
        parser.error("--temp-min-c and --temp-max-c must be given together.")
    if args.temp_min_c is not None and args.temp_max_c <= args.temp_min_c:
        parser.error("--temp-max-c must be greater than --temp-min-c.")
    if args.raw_every_n < 1:
        parser.error("--raw-every-n must be at least 1.")
    if args.max_reconnect_seconds < 0:
        parser.error("--max-reconnect-seconds must be >= 0.")
    record(args)


if __name__ == "__main__":
    main()
