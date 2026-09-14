"""Batch-run animal_detector.py over every video in a folder and export
annotated clips for each confirmed detection episode.

Two passes per source file:
  1. Run the detector over the whole file to find confirmed episodes
     (same logic as test_on_video.py).
  2. For each episode, re-open the file, seek to a bit before the episode
     start (--warmup extra hidden seconds so the background model has
     settled before the visible clip begins, plus --pad seconds of
     visible lead-in/lead-out), and write an annotated clip with
     bounding boxes.

Files with zero detections cost only the first pass (fast: no video
writer, no second decode). Only files that actually contain something
get a second pass and an output clip - so runtime scales with how much
is actually in the footage, not just its total length.

Also writes detections_summary.csv in the output folder: one row per
episode, with source file, timing, and peak blob size, independent of
the video clips - useful for a quick scan without opening every clip.

Usage:
    python batch_detect.py INPUT_DIR OUTPUT_DIR
    python batch_detect.py INPUT_DIR OUTPUT_DIR --pad 3 --pattern "*.avi"
"""

import argparse
import csv
import glob
import os
import time

import cv2

from animal_detector import DetectorConfig, ThermalAnimalDetector
from test_on_video import annotate_frame, merge_close_episodes


def find_episodes(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = ThermalAnimalDetector(DetectorConfig())
    episodes = []
    open_episode = None
    frame_idx = 0
    while True:
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
        frame_idx += 1
    if open_episode is not None:
        episodes.append(open_episode)
    cap.release()

    episodes = merge_close_episodes(episodes, fps)
    return episodes, fps, w, h, frame_idx


def render_episode_clip(video_path, episode, fps, w, h, pad_s, warmup_s, out_path):
    start_frame = max(0, int(episode["start_frame"] - pad_s * fps))
    warmup_frame = max(0, int(start_frame - warmup_s * fps))
    end_frame = int(episode["end_frame"] + pad_s * fps)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_frame)

    detector = ThermalAnimalDetector(DetectorConfig())
    ts_y = int(h * (1 - detector.cfg.timestamp_frac_h))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), True)

    idx = warmup_frame
    while idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detector.process(gray)
        if idx >= start_frame:
            writer.write(annotate_frame(frame, result, ts_y))
        idx += 1

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--pattern", default="*.avi")
    parser.add_argument(
        "--pad", type=float, default=3.0,
        help="Extra seconds of clip before/after the confirmed episode (default: 3.0)",
    )
    parser.add_argument(
        "--warmup", type=float, default=3.0,
        help="Extra hidden seconds decoded before the clip so the background "
        "model has settled before the visible part starts (default: 3.0)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Found {len(files)} files to scan.", flush=True)

    summary_path = os.path.join(args.output_dir, "detections_summary.csv")
    summary_rows = []

    t_start = time.time()
    total_episodes = 0
    failed = []
    for i, video_path in enumerate(files):
        fname = os.path.basename(video_path)
        result = find_episodes(video_path)
        if result is None:
            print(f"[{i+1}/{len(files)}] {fname}: FAILED TO OPEN", flush=True)
            failed.append(fname)
            continue
        episodes, fps, w, h, nframes = result
        elapsed = time.time() - t_start
        eta_min = elapsed / (i + 1) * (len(files) - i - 1) / 60
        print(
            f"[{i+1}/{len(files)}] {fname}: {len(episodes)} episode(s)  "
            f"(elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min)",
            flush=True,
        )

        for j, ep in enumerate(episodes):
            start_s = ep["start_frame"] / fps
            end_s = ep["end_frame"] / fps
            clip_name = f"{os.path.splitext(fname)[0]}_ep{j+1}_t{start_s:.1f}-{end_s:.1f}.mp4"
            clip_path = os.path.join(args.output_dir, clip_name)
            render_episode_clip(video_path, ep, fps, w, h, args.pad, args.warmup, clip_path)
            print(
                f"    -> {clip_name}  ({start_s:.1f}s-{end_s:.1f}s, peak {ep['peak_area']}px)",
                flush=True,
            )
            summary_rows.append({
                "source_file": fname, "episode": j + 1,
                "start_s": round(start_s, 2), "end_s": round(end_s, 2),
                "duration_s": round(end_s - start_s, 2),
                "peak_area_px": ep["peak_area"],
                "peak_x": round(ep["peak_x"], 1), "peak_y": round(ep["peak_y"], 1),
                "clip_file": clip_name,
            })
            total_episodes += 1

    with open(summary_path, "w", newline="") as f:
        fieldnames = ["source_file", "episode", "start_s", "end_s", "duration_s",
                      "peak_area_px", "peak_x", "peak_y", "clip_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. {len(files)} files scanned ({len(failed)} failed to open), "
          f"{total_episodes} detection episode(s) found.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
