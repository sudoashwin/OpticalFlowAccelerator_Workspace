import numpy as np
import cv2
import os

# --- Parameters ---
# Grid Size 4x4 (GridSize 2) means resolution is Input_Res / 4
WIDTH = 962
HEIGHT = 542
CHANNELS = 2  # X and Y vectors
BYTES_PER_INT16 = 2
FRAME_SIZE_BYTES = WIDTH * HEIGHT * CHANNELS * BYTES_PER_INT16

# File paths
INPUT_BIN = 'front_flowoutput.bin'
OUTPUT_MP4 = 'flow_visualizationOutputOnly.mp4'

def main():
    if not os.path.exists(INPUT_BIN):
        print(f"Error: Could not find {INPUT_BIN}")
        return

    # Setup the OpenCV Video Writer
    # We use mp4v codec for standard MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_MP4, fourcc, 30.0, (WIDTH, HEIGHT))
    
    # Pre-allocate an HSV image array (Hue, Saturation, Value)
    hsv = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set maximum saturation for vibrant colors

    frames_processed = 0

    print(f"Opening {INPUT_BIN} and rendering video...")

    with open(INPUT_BIN, 'rb') as f:
        while True:
            # Read exact byte chunk for one frame
            raw_data = f.read(FRAME_SIZE_BYTES)
            
            # Break if we hit the end of the file or an incomplete frame
            if not raw_data or len(raw_data) != FRAME_SIZE_BYTES:
                break
                
            # 1. Parse the binary data as 16-bit integers
            flow_data = np.frombuffer(raw_data, dtype=np.int16).reshape((HEIGHT, WIDTH, 2))
            
            # 2. Hardware scaling conversion
            # The OFA hardware outputs vectors in S10.5 fixed-point format.
            # To get true pixel displacement, we convert to float and divide by 2^5 (32).
            flow_data_float = flow_data.astype(np.float32) / 32.0
            
            flow_x = flow_data_float[..., 0]
            flow_y = flow_data_float[..., 1]
            
            # 3. Convert Cartesian (X,Y) to Polar (Magnitude, Angle)
            mag, ang = cv2.cartToPolar(flow_x, flow_y)
            
            # 4. Map the math to colors
            # OpenCV Hue range is [0, 179] for 8-bit images, so we scale the angle
            hsv[..., 0] = ang * 180 / np.pi / 2
            
            # Normalize the magnitude to a [0, 255] brightness value
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert HSV back to standard BGR for the video writer
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Write the colored frame to the MP4 file
            out.write(bgr)
            frames_processed += 1
            
            # Simple progress tracker
            if frames_processed % 10 == 0:
                print(f"Rendered {frames_processed} frames...")

    out.release()
    print(f"\nDone! Successfully rendered {frames_processed} frames.")
    print(f"Saved output to: {OUTPUT_MP4}")

if __name__ == "__main__":
    main()
