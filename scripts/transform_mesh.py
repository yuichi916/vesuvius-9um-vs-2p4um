"""Map a tifxyz from the fixed scan into the moving scan of a volume-registration transform.

usage: python transform_mesh.py SRC_TIFXYZ_DIR DST_DIR TRANSFORM_JSON
TRANSFORM_JSON: foundation/volume-registration schema (XYZ, p_fixed = M @ p_moving),
e.g. from auto_register.py with the 9 um scan fixed and the 2.4 um scan moving.
"""
import json, sys
from pathlib import Path
import numpy as np, tifffile
src, dst, aff = Path(sys.argv[1]), Path(sys.argv[2]), json.load(open(sys.argv[3]))
Mx = np.array(aff["transformation_matrix"])          # XYZ
perm = [2, 1, 0]                                      # work in ZYX below
M = np.zeros((3, 4)); M[:, :3] = Mx[:, :3][np.ix_(perm, perm)]; M[:, 3] = Mx[:, 3][perm]
A, t = M[:, :3], M[:, 3]
Ainv = np.linalg.inv(A)
x, y, z = (tifffile.imread(src / f"{c}.tif").astype(np.float64) for c in "xyz")
bad = (x == -1) | (y == -1) | (z <= 0) | ~np.isfinite(x)
p9 = np.stack([z, y, x], -1)                       # zyx
p2 = (p9 - t) @ Ainv.T
dst.mkdir(parents=True, exist_ok=True)
for i, c in enumerate("zyx"):
    v = p2[..., i].astype(np.float32); v[bad] = -1
    tifffile.imwrite(dst / f"{c}.tif", v)
meta = json.load(open(src / "meta.json"))
k = float(np.cbrt(abs(np.linalg.det(Ainv))))       # 2.4 voxels per 9um voxel
meta["scale"] = [s / k for s in meta["scale"]]
meta["transformed_from"] = {"source": str(src), "transform": sys.argv[3]}
json.dump(meta, open(dst / "meta.json", "w"), indent=1)
inside = (~bad) & (p2[..., 0] >= 0) & (p2[..., 0] < 15137)
print(f"{src.name}: scale {meta['scale'][0]:.5f}, vertices inside 2.4 band {inside.sum() / max((~bad).sum(), 1):.2%}")
