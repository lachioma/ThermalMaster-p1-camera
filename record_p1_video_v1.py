"""

Stand-alone script that:

connects to the camera

sets the P1 model

records frames for a specified duration

saves them to a video file

does no visualization

saves the raw thermal frames (no colormap)



How to run it


Activate your environment:

source venv/bin/activate

Run the script:

python record_p1_video_v1.py --duration 60 --output test.avi

Example result:

Recording for 60 seconds...
Camera resolution: 256x192
Recording finished.
Frames captured: 1500
Saved to: test.avi

"""

import argparse
import time
import cv2
import numpy as np

from p3_camera import P3Camera
from p3_camera import (
    get_model_config,
)
from p3_viewer import (
    AGCMode,
    ScaleMode,
)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="p1", help="Camera model")
    parser.add_argument("--duration", type=int, required=True, help="Recording duration in seconds")
    parser.add_argument("--output", default="thermal.avi", help="Output video file")
    parser.add_argument("--fps", type=int, default=25)

    args = parser.parse_args()

    # Get camera configuration exactly like p3_viewer
    config = get_model_config(args.model)

    camera = P3Camera(config=config)
    camera.connect()


    name, version = camera.init()
    print(f"Device: {name}, Firmware: {version}")
    
    
    width  = 160
    height = 120
    
    

    print(f"Camera resolution: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height), False)

    camera.start_streaming()
    
    start = time.time()
    frame_count = 0

    print(f"Recording for {args.duration} seconds...")

    while time.time() - start < args.duration:

        # ir_brightness, thermal = self.camera.read_frame_both()
        # if thermal is None:
            # continue
        # self._ir_brightness = ir_brightness
        
        ir_brightness = camera.read_frame_both()

        _last_display = ir_brightness
        
        # _last_display = camera._render(ir_brightness)

        _last_display = np.asarray(_last_display, dtype=np.uint8)
        # # normalize to 8-bit for video storage
        _last_display = cv2.normalize(_last_display, None, 0, 255, cv2.NORM_MINMAX)
        # _last_display = frame.astype(np.uint8)
        

        writer.write(_last_display)

        frame_count += 1

    writer.release()
    camera.stop_streaming()

    print("Finished.")
    print(f"Frames recorded: {frame_count}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()



def _render(thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Render thermal frame to display image."""

    frame_stats = {}

    # AGC: normalize to 8-bit based on selected mode
    if self.agc_mode == AGCMode.FACTORY:
        # Use hardware AGC'd IR brightness from camera (already 8-bit)
        if self._ir_brightness is not None:
            img = self._ir_brightness.copy()
        else:
            img = agc_temporal(thermal, pct=1.0)
    elif self.agc_mode == AGCMode.FIXED_RANGE:
        img = agc_fixed(thermal)
    else:
        pct = AGC_PERCENTILES.get(self.agc_mode, 1.0)
        img = agc_temporal(thermal, pct=pct)

    # Optional 2x upscaling
    if self.scale_mode != ScaleMode.OFF:
        h, w = img.shape[:2]
        resized: Any = cv2.resize(
            img, (w * 2, h * 2), interpolation=SCALE_INTERP[self.scale_mode]
        )
        # Ensure result is numpy array (cv2.resize may return cv2.UMat on some platforms)
        img = np.asarray(resized, dtype=np.uint8)

    # Optional CLAHE for local contrast enhancement
    if self.use_clahe:
        clahe_result: Any = self._clahe.apply(img)
        # Ensure result is a numpy array (CLAHE may return cv2.UMat on some platforms)
        img = np.asarray(clahe_result, dtype=np.uint8)

    # DDE: edge enhancement
    img = dde(img, strength=self.dde_strength)

    # # Temperature values (with emissivity correction)
    # cy, cx = self._get_spot_coords(thermal)
    # env = self.camera.env_params
    # frame_stats['tspot'] = float(raw_to_celsius_corrected(thermal[cy, cx], env))
    # frame_stats['cmax'] = thermal.argmax()
    # frame_stats['cmin'] = thermal.argmin()
    # frame_stats['tmin'] = float(raw_to_celsius_corrected(thermal.ravel()[frame_stats['cmin']], env))
    # frame_stats['tmax'] = float(raw_to_celsius_corrected(thermal.ravel()[frame_stats['cmax']], env))
    # # value range
    # frame_stats['range_min'] = int(np.min(img))
    # frame_stats['range_max'] = int(np.max(img))

    # # Apply colormap
    # img = apply_colormap(img, self.colormap_idx)

    # # Mirror
    # if self.mirror:
        # img = cv2.flip(img, 1)

    # # Rotate
    # if self.rotation == 90:
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # elif self.rotation == 180:
        # img = cv2.rotate(img, cv2.ROTATE_180)
    # elif self.rotation == 270:
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # # Zoom
    # h, w = img.shape[:2]
    # result = cast(
        # NDArray[np.uint8],
        # cv2.resize(
            # img, (w * self.zoom, h * self.zoom), interpolation=cv2.INTER_LINEAR
        # ),
    # )

    # if self.show_colorbar:
        # self._draw_colorbar(result, frame_stats)

    # # Overlays
    # self._draw_overlays(result, thermal, frame_stats)

    # # If lock-in results exist, composite two panes to the right
    # if self.lockin_controller is not None:
        # in_phase, quad, amplitude, angle = self.lockin_controller.get_latest()
        # if in_phase is not None and quad is not None and amplitude is not None and angle is not None:
            # try:
                # result = self._composite_lockin_panes(result, in_phase, quad, amplitude, angle)
            # except Exception:
                # # Don't let lock-in display errors break rendering
                # pass

    return result
