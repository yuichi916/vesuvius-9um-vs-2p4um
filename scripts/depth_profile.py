"""Per-tile depth profiles of flat surface volumes.

For each on-surface 128px tile: mean intensity per plane, normalised by the
tile's overall std. Reports the profile's contrast ((max - min) / std), the
plane of the brightest and darkest mean, and the median profile per segment.

usage: python depth_profile.py OUT_JSON NAME=ZARR [NAME=ZARR ...]
"""
import json
import sys

import numpy as np
import zarr

T = 128


def tile_profiles(path, max_tiles=600, seed=0):
    a = zarr.open_array(path + "/0", mode="r")
    D, H, W = a.shape
    cands = [(y, x) for y in range(0, H - T + 1, T) for x in range(0, W - T + 1, T)]
    rng = np.random.default_rng(seed)
    rng.shuffle(cands)
    out = []
    for y, x in cands:
        if len(out) >= max_tiles:
            break
        mid = np.asarray(a[D // 2, y:y + T, x:x + T])
        if (mid > 0).mean() < 0.98:
            continue
        v = np.asarray(a[:, y:y + T, x:x + T]).astype(np.float32)
        if (v == 0).mean() > 0.02:
            continue
        prof = v.reshape(D, -1).mean(1)
        sd = v.std()
        out.append(((prof - prof.mean()) / max(sd, 1e-6)).tolist())
    return np.array(out)


def main():
    out_json = sys.argv[1]
    res = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        P = tile_profiles(path)
        if len(P) == 0:
            print(name, "no tiles"); continue
        contrast = P.max(1) - P.min(1)
        res[name] = {"n": len(P), "contrast_median": float(np.median(contrast)),
                     "contrast_p10": float(np.percentile(contrast, 10)),
                     "contrast_p90": float(np.percentile(contrast, 90)),
                     "median_profile": np.median(P, 0).round(3).tolist(),
                     "argmax_hist": np.bincount(P.argmax(1), minlength=P.shape[1]).tolist(),
                     "contrast": contrast.round(4).tolist()}
        print(f"{name:40s} n={len(P):4d} contrast median {np.median(contrast):.3f} "
              f"(p10 {np.percentile(contrast, 10):.3f}, p90 {np.percentile(contrast, 90):.3f}) "
              f"median profile {np.round(np.median(P, 0)[::3], 2).tolist()}", flush=True)
    json.dump(res, open(out_json, "w"))


if __name__ == "__main__":
    main()
