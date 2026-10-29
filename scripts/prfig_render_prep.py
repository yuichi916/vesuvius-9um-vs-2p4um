"""Crop the published and the Python-rendered PHerc0139 w035 surface volumes to the same window for the PR figure."""
import numpy as np
import zarr

OUT = "E:/vesuvius/prfig"
pub = zarr.open("E:/vesuvius/n9/w035/w035.zarr", mode="r")["0"]
mine = zarr.open("E:/vesuvius/rtest/w035_full.zarr", mode="r")["0"]
assert pub.shape == mine.shape, (pub.shape, mine.shape)

# window with the most joint coverage among a few candidates
S = 2048
best = None
for y0 in range(256, pub.shape[1] - S, 512):
    for x0 in range(256, pub.shape[2] - S, 512):
        a = np.asarray(pub[14, y0:y0 + S:8, x0:x0 + S:8]); b = np.asarray(mine[14, y0:y0 + S:8, x0:x0 + S:8])
        cov = ((a > 0) & (b > 0)).mean()
        if best is None or cov > best[0]:
            best = (cov, y0, x0)
cov, y0, x0 = best
print("window", y0, x0, "coverage", round(cov, 3))
for name, src in (("pub", pub), ("mine", mine)):
    v = np.asarray(src[:, y0:y0 + S, x0:x0 + S])
    g = zarr.open_group(f"{OUT}/w035_{name}.zarr", mode="w", zarr_format=2)
    g.create_array("0", data=v, chunks=(v.shape[0], 128, 128))
np.save(f"{OUT}/window.npy", np.array([y0, x0, S]))
