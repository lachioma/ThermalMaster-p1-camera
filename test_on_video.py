"""Run animal_detector.py against a recorded thermal video for calibration/review.

Works on your PC (this is how the detector was tuned - against real field
footage) and on the Raspberry Pi unchanged, since it's plain OpenCV/numpy.

Usage:

    python test_on_video.py VIDEO.avi
        Print each confirmed detection episode (start/end time, duration,
        peak blob size) found in the whole file.

    python test_on_video.py VIDEO.avi --start 300 --end 400
        Only scan that time range (seconds) - much faster for checking one
        known event instead of a full 10-minute segment.

    python test_on_video.py VIDEO.avi --annotate OUT.mp4
        Also write an annotated copy of the video: every frame gets the
        candidate blobs outlined in yellow, confirmed detections in red
        with a "ANIMAL" label, and the masked timestamp strip shaded blue
        so you can visually confirm it's excluded. Use this to sanity-check
        parameters on a new camera position or a suspicious clip.

    python test_on_video.py VIDEO.avi --annotate OUT.mp4 --start 300 --end 400
        Combine both - the fast way to get a short, reviewable clip of one
        known event.
"""

import argparse
import sys

import cv2

from animal_detector import DetectorConfig, ThermalAnimalDetector


def merge_close_episodes(episodes, fps, max_gap_s=2.0):
    """Merge episodes separated by a brief gap (e.g. the animal paused, or one
    frame briefly failed persistence) into a single reported event."""
    if not episodes:
        return []
    max_gap_frames = max_gap_s * fps
    merged = [dict(episodes[0])]
    for ep in episodes[1:]:
        if ep["start_frame"] - merged[-1]["end_frame"] <= max_gap_frames:
            merged[-1]["end_frame"] = ep["end_frame"]
            if ep["peak_area"] > merged[-1]["peak_area"]:
                merged[-1].update(peak_area=ep["peak_area"], peak_x=ep["peak_x"], peak_y=ep["peak_y"])
        else:
            merged.append(dict(ep))
    return merged


def format_episodes(episodes, fps, time_offset_s=0.0):
    lines = []
    for ep in episodes:
        start_s = time_offset_s + ep["start_frame"] / fps
        end_s = time_offset_s + ep["end_frame"] / fps
        lines.append(
            f"  {start_s:7.2f}s - {end_s:7.2f}s  (duration {end_s - start_s:5.2f}s, "
            f"{ep['end_frame'] - ep['start_frame'] + 1} frames, "
            f"peak area {ep['peak_area']}px at ~({ep['peak_x']:.0f},{ep['peak_y']:.0f}))"
        )
    return "\n".join(lines) if lines else "  (none)"


def annotate_frame(frame_bgr, result, ts_y):
    overlay = frame_bgr.copy()
    overlay[ts_y:, :] = (overlay[ts_y:, :] * 0.5 + (255, 128, 0)).astype(overlay.dtype)
    frame_bgr = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    for cx, cy, area in result.candidate_blobs:
        r = max(3, int((area**0.5)))
        color = (0, 0, 255) if result.confirmed else (0, 255, 255)
        cv2.circle(frame_bgr, (int(cx), int(cy)), r, color, 1)

    if result.confirmed:
        cx, cy = result.confirmed_centroid
        cv2.putText(
            frame_bgr, "ANIMAL", (max(0, int(cx) - 20), max(10, int(cy) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to a thermal .avi/.mp4 segment")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: end of file)")
    parser.add_argument("--annotate", type=str, default=None, help="Write an annotated copy to this path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video}: {w}x{h} @ {fps:.2f}fps")

    if args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    detector = ThermalAnimalDetector(DetectorConfig())
    ts_y = int(h * (1 - detector.cfg.timestamp_frac_h))

    writer = None
    if args.annotate:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotate, fourcc, fps, (w, h), True)

    episodes = []
    open_episode = None
    frame_idx = 0

    while True:
        pos_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if args.end is not None and pos_s > args.end:
            break
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detector.process(gray)

        if result.confirmed:
            cx, cy = result.confirmed_centroid
            area = next((a for x, y, a in result.candidate_blobs if (x, y) == (cx, cy)), 0)
            if open_episode is None:
                open_episode = {
                    "start_frame": frame_idx, "end_frame": frame_idx,
                    "peak_area": area, "peak_x": cx, "peak_y": cy,
                }
            else:
                open_episode["end_frame"] = frame_idx
                if area > open_episode["peak_area"]:
                    open_episode.update(peak_area=area, peak_x=cx, peak_y=cy)
        else:
            if open_episode is not None:
                episodes.append(open_episode)
                open_episode = None

        if writer is not None:
            writer.write(annotate_frame(frame, result, ts_y))

        frame_idx += 1

    if open_episode is not None:
        episodes.append(open_episode)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video written to {args.annotate}")

    episodes = merge_close_episodes(episodes, fps)
    print(f"\nScanned {frame_idx} frames ({frame_idx / fps:.1f}s starting at {args.start:.1f}s). "
          f"Confirmed detection episodes: {len(episodes)}")
    print(format_episodes(episodes, fps, time_offset_s=args.start))


if __name__ == "__main__":
    main()
