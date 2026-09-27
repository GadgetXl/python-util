import numpy as np
from PIL import Image
from scipy import ndimage
import sys

if len(sys.argv) < 2:
    print("Usage: python height_fix.py INPUT_BIN")
    sys.exit(1)

def process_heightmap(
    input_bin=sys.argv[1],
    output_bin="fixed_"+sys.argv[1],
    output_png="heightmap_fixed.png",
):
    data = np.fromfile(input_bin, dtype=np.float32)

    side = int(np.sqrt(data.size))
    data = data.reshape((side, side))

    mask = data == 0.0

    if np.any(mask):
        indices = ndimage.distance_transform_edt(
            mask, return_distances=False, return_indices=True
        )
        filled = data[tuple(indices)].copy()

        smoothed = filled.copy()
        for _ in range(25):
            blurred = ndimage.gaussian_filter(smoothed, sigma=2)
            smoothed[mask] = blurred[mask]

        final_data = np.where(mask, smoothed, data)
    else:
        final_data = data

    final_data.astype(np.float32).tofile(output_bin)

    mn, mx = final_data.min(), final_data.max()
    img_array = (
        ((final_data - mn) / (mx - mn) * 255).astype(np.uint8)
        if mx > mn
        else np.zeros_like(final_data, dtype=np.uint8)
    )
    Image.fromarray(img_array).save(output_png)

    print(f"Process completed. Size: {side}x{side}")


if __name__ == "__main__":
    process_heightmap()