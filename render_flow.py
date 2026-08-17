import numpy as np
import cv2
import os

# --- Parameters ---
# Input resolution 3848x2168, gridSize 2 (4x4) -> flow resolution = input / 4
WIDTH = 962
HEIGHT = 542

FLOW_FRAME_BYTES = WIDTH * HEIGHT * 2 * 2   # 2 channels (X,Y), int16
COST_FRAME_BYTES = WIDTH * HEIGHT           # 1 byte per grid cell

# Fixed magnitude scale: displacement (px) that maps to full brightness.
# Tune to your data (use the printed per-frame stats as a guide).
MAX_FLOW_PX = 8.0

# Cells with magnitude below this (px) are rendered black (direction is noise)
MIN_MAG_PX = 0.5

# Cells with cost above this are considered unreliable and rendered black.
# Cost is uint8, lower = better match. Tune using printed cost stats.
MAX_COST = 100

FPS = 30.0

# File paths
FLOW_BIN = 'front_flowoutput.bin'
COST_BIN = 'front_flowcostoutput.bin'   # set to None to disable cost masking
OUTPUT_MP4 = 'flow_visualization.mp4'


def main():
    if not os.path.exists(FLOW_BIN):
        print(f"Error: Could not find {FLOW_BIN}")
        return

    use_cost = COST_BIN is not None and os.path.exists(COST_BIN)
    if COST_BIN is not None and not use_cost:
        print(f"WARNING: {COST_BIN} not found — rendering without cost mask.")

    # --- Sanity checks ---
    flow_size = os.path.getsize(FLOW_BIN)
    if flow_size % FLOW_FRAME_BYTES != 0:
        print(f"WARNING: flow file size {flow_size} not a multiple of "
              f"{FLOW_FRAME_BYTES} — stride/resolution mismatch likely.")
    n_flow_frames = flow_size // FLOW_FRAME_BYTES
    print(f"Flow file: {n_flow_frames} frames.")

    if use_cost:
        cost_size = os.path.getsize(COST_BIN)
        n_cost_frames = cost_size // COST_FRAME_BYTES
        if cost_size % COST_FRAME_BYTES != 0:
            print(f"WARNING: cost file size {cost_size} not a multiple of "
                  f"{COST_FRAME_BYTES}.")
        if n_cost_frames != n_flow_frames:
            print(f"WARNING: cost frames ({n_cost_frames}) != flow frames "
                  f"({n_flow_frames}) — masking may misalign.")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_MP4, fourcc, FPS, (WIDTH, HEIGHT))

    hsv = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    brightness_scale = 255.0 / MAX_FLOW_PX
    frames_processed = 0

    print(f"Rendering from {FLOW_BIN}"
          + (f" with cost mask from {COST_BIN}" if use_cost else "") + " ...")

    f_flow = open(FLOW_BIN, 'rb')
    f_cost = open(COST_BIN, 'rb') if use_cost else None

    try:
        while True:
            raw_flow = f_flow.read(FLOW_FRAME_BYTES)
            if not raw_flow or len(raw_flow) != FLOW_FRAME_BYTES:
                break

            if use_cost:
                raw_cost = f_cost.read(COST_FRAME_BYTES)
                if not raw_cost or len(raw_cost) != COST_FRAME_BYTES:
                    print("WARNING: cost file ended early — "
                          "continuing without mask.")
                    use_cost = False

            # 1. Parse flow: int16, S10.5 fixed-point -> divide by 32
            flow = np.frombuffer(raw_flow, dtype=np.int16).reshape(
                (HEIGHT, WIDTH, 2)).astype(np.float32) / 32.0

            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # 2. Cartesian -> Polar
            mag, ang = cv2.cartToPolar(flow_x, flow_y)

            # 3. Hue = direction, Value = fixed-scale magnitude
            hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
            hsv[..., 2] = np.clip(mag * brightness_scale, 0, 255).astype(
                np.uint8)

            # 4. Mask: near-zero magnitude (direction is noise)
            hsv[mag < MIN_MAG_PX, 2] = 0

            # 5. Mask: high-cost (unreliable) cells
            if use_cost:
                cost = np.frombuffer(raw_cost, dtype=np.uint8).reshape(
                    (HEIGHT, WIDTH))
                hsv[cost > MAX_COST, 2] = 0

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out.write(bgr)
            frames_processed += 1

            # Per-frame stats to guide MAX_FLOW_PX / MAX_COST tuning
            if frames_processed % 10 == 0:
                msg = (f"frame {frames_processed}: "
                       f"mag mean={mag.mean():.2f} "
                       f"p95={np.percentile(mag, 95):.2f} "
                       f"max={mag.max():.2f}")
                if use_cost:
                    msg += (f" | cost mean={cost.mean():.0f} "
                            f"p95={np.percentile(cost, 95):.0f}")
                print(msg)
    finally:
        f_flow.close()
        if f_cost:
            f_cost.close()

    out.release()
    print(f"\nDone! Rendered {frames_processed} frames -> {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
