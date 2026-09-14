import time
import argparse
import cv2
import numpy as np
from p3_camera import Model, P3Camera, get_model_config

def record_thermal_video(duration_seconds, output_filename, fps=25.0):
    # Initialize the camera configuration for the P1 model
    config = get_model_config(Model.P1)
    camera = P3Camera(config=config)
    
    print("Connecting to Thermal Master P1...")
    camera.connect()
    camera.init()
    camera.start_streaming()
    
    # Native Resolution for the P1 camera
    width, height = 160, 120
    
    # Set up OpenCV VideoWriter
    # Using XVID codec for a standard AVI file (8-bit grayscale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
    
    # List to hold the true 16-bit radiometric data
    raw_frames = []

    print(f"Recording started for {duration_seconds} seconds. Please wait...")
    start_time = time.time()
    frames_recorded = 0
    
    try:
        # Loop until the specified duration has passed
        while (time.time() - start_time) < duration_seconds:
            # Read frames: 
            # ir_brightness is 8-bit (visual), thermal_raw is 16-bit (radiometric temperature)
            ir_brightness, thermal_raw = camera.read_frame_both()
            
            if ir_brightness is not None:
                # 1. Write the 8-bit visual image to the standard video file
                out.write(ir_brightness)
                
                # 2. Append the true 16-bit raw data to our list
                raw_frames.append(thermal_raw)
                
                frames_recorded += 1
                
    except KeyboardInterrupt:
        print("\nRecording interrupted manually by user.")
    
    # Cleanup and release hardware resources
    out.release()
    camera.stop_streaming()
    camera.disconnect()
    
    print(f"Recording complete. {frames_recorded} frames saved.")
    print(f"Visual video saved to: {output_filename}")
    
    # Save the 16-bit raw radiometric data as a NumPy file (.npy)
    # This allows you to reconstruct the exact temperatures later in Python
    raw_output_filename = output_filename.rsplit('.', 1)[0] + "_raw.npy"
    np.save(raw_output_filename, np.array(raw_frames))
    print(f"16-bit raw thermal data saved to: {raw_output_filename}")

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Headless thermal video recording for Thermal Master P1.")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the recording in seconds")
    parser.add_argument("--out", type=str, default="output.avi", help="Output video filename (must end in .avi)")
    
    args = parser.parse_args()
    record_thermal_video(args.duration, args.out)
