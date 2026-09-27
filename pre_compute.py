import time
import numpy as np
from numba import njit
import sys

if len(sys.argv) < 3:
    print("Usage: python pre_compute.py INPUT_FILE OUTPUT_FILE_NAME") 
    sys.exit(1)

# -- Config --
MAP_FILE = sys.argv[1] # Raw binary map file (.bin)
OUTPUT_FILE = sys.argv[2] if sys.argv[2].endswith(".npz") else sys.argv[2] + ".npz"
MAP_SHAPE = (2048, 2048)
WALKABLE_VALUE = 255

@njit
def _compute_diagonals_numba(jps_dist, walkable, H, W):
    configs = [
        (4, 1, -1, 2, 0, 0, H, 1, W - 1, -1, -1),  # NE
        (5, -1, -1, 3, 0, 0, H, 1, 0, W, 1),      # NW
        (6, 1, 1, 2, 1, H - 1, -1, -1, W - 1, -1, -1),  # SE
        (7, -1, 1, 3, 1, H - 1, -1, -1, 0, W, 1),  # SW
    ]

    for cfg in configs:
        d_idx, dx, dy, h_dir, v_dir = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
        y_s, y_e, y_st = cfg[5], cfg[6], cfg[7]
        x_s, x_e, x_st = cfg[8], cfg[9], cfg[10]

        for y in range(y_s, y_e, y_st):
            for x in range(x_s, x_e, x_st):
                if not walkable[y, x]:
                    continue

                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if (
                        walkable[ny, nx]
                        and walkable[y, nx]
                        and walkable[ny, x]
                    ):
                        is_diag_jp = (jps_dist[ny, nx, h_dir] > 0) or (
                            jps_dist[ny, nx, v_dir] > 0
                        )
                        if is_diag_jp:
                            jps_dist[y, x, d_idx] = 1
                        else:
                            nd = jps_dist[ny, nx, d_idx]
                            if nd > 0:
                                jps_dist[y, x, d_idx] = nd + 1
                            else:
                                jps_dist[y, x, d_idx] = nd - 1
                    else:
                        jps_dist[y, x, d_idx] = -1
                else:
                    jps_dist[y, x, d_idx] = -1


def precompute_jps():
    print(f"[*] '{MAP_FILE}' reading...")
    start_time = time.time()

    try:
        with open(MAP_FILE, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except FileNotFoundError:
        print("[*] Input file not found")
        sys.exit(1)

    grid = data.reshape(MAP_SHAPE)
    H, W = MAP_SHAPE
    walkable = grid == WALKABLE_VALUE

    padded = np.pad(
        walkable, pad_width=1, mode="constant", constant_values=False
    )
    jps_dist = np.zeros((H, W, 8), dtype=np.int16)

    # Cardinal directions
    is_jp_E = walkable & (
        ((~padded[0:H, 1 : W + 1]) & padded[0:H, 2 : W + 2])
        | ((~padded[2 : H + 2, 1 : W + 1]) & padded[2 : H + 2, 2 : W + 2])
    )
    is_jp_W = walkable & (
        ((~padded[0:H, 1 : W + 1]) & padded[0:H, 0:W])
        | ((~padded[2 : H + 2, 1 : W + 1]) & padded[2 : H + 2, 0:W])
    )
    is_jp_S = walkable & (
        ((~padded[1 : H + 1, 0:W]) & padded[2 : H + 2, 0:W])
        | ((~padded[1 : H + 1, 2 : W + 2]) & padded[2 : H + 2, 2 : W + 2])
    )
    is_jp_N = walkable & (
        ((~padded[1 : H + 1, 0:W]) & padded[0:H, 0:W])
        | ((~padded[1 : H + 1, 2 : W + 2]) & padded[0:H, 2 : W + 2])
    )

    # East (2)
    jps_dist[walkable[:, W - 1], W - 1, 2] = -1
    for x in range(W - 2, -1, -1):
        nw, jp, nd = (
            walkable[:, x + 1],
            is_jp_E[:, x + 1],
            jps_dist[:, x + 1, 2],
        )
        d = np.zeros(H, dtype=np.int16)
        d[~nw] = -1
        d[nw & jp] = 1
        d[nw & (~jp) & (nd > 0)] = nd[nw & (~jp) & (nd > 0)] + 1
        d[nw & (~jp) & (nd <= 0)] = nd[nw & (~jp) & (nd <= 0)] - 1
        jps_dist[walkable[:, x], x, 2] = d[walkable[:, x]]

    # West (3)
    jps_dist[walkable[:, 0], 0, 3] = -1
    for x in range(1, W):
        nw, jp, nd = (
            walkable[:, x - 1],
            is_jp_W[:, x - 1],
            jps_dist[:, x - 1, 3],
        )
        d = np.zeros(H, dtype=np.int16)
        d[~nw] = -1
        d[nw & jp] = 1
        d[nw & (~jp) & (nd > 0)] = nd[nw & (~jp) & (nd > 0)] + 1
        d[nw & (~jp) & (nd <= 0)] = nd[nw & (~jp) & (nd <= 0)] - 1
        jps_dist[walkable[:, x], x, 3] = d[walkable[:, x]]

    # South (1)
    jps_dist[H - 1, walkable[H - 1, :], 1] = -1
    for y in range(H - 2, -1, -1):
        nw, jp, nd = (
            walkable[y + 1, :],
            is_jp_S[y + 1, :],
            jps_dist[y + 1, :, 1],
        )
        d = np.zeros(W, dtype=np.int16)
        d[~nw] = -1
        d[nw & jp] = 1
        d[nw & (~jp) & (nd > 0)] = nd[nw & (~jp) & (nd > 0)] + 1
        d[nw & (~jp) & (nd <= 0)] = nd[nw & (~jp) & (nd <= 0)] - 1
        jps_dist[y, walkable[y, :], 1] = d[walkable[y, :]]

    # North (0)
    jps_dist[0, walkable[0, :], 0] = -1
    for y in range(1, H):
        nw, jp, nd = (
            walkable[y - 1, :],
            is_jp_N[y - 1, :],
            jps_dist[y - 1, :, 0],
        )
        d = np.zeros(W, dtype=np.int16)
        d[~nw] = -1
        d[nw & jp] = 1
        d[nw & (~jp) & (nd > 0)] = nd[nw & (~jp) & (nd > 0)] + 1
        d[nw & (~jp) & (nd <= 0)] = nd[nw & (~jp) & (nd <= 0)] - 1
        jps_dist[y, walkable[y, :], 0] = d[walkable[y, :]]

    # Diagonal directions
    _compute_diagonals_numba(jps_dist, walkable, H, W)

    elapsed = time.time() - start_time
    print(f"[+] Pre compute completed in {elapsed:.2f} seconds.")

    np.savez_compressed(
        OUTPUT_FILE, grid=grid, jps_dist=jps_dist, walkable_val=WALKABLE_VALUE
    )
    print(f"[+] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    precompute_jps()