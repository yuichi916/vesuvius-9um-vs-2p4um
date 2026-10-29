"""Map a 9um tifxyz through a registration and measure its distance to the published 2.4um tifxyz.

usage: python validate_transfer.py MESH_9UM_DIR MESH_2P4_DIR TRANSFORM_JSON
TRANSFORM_JSON: volume-registration schema, fixed = 9um scan, moving = 2.4um scan (XYZ, level 0).
Reports the normal-direction and in-plane distance (2.4um voxels and um) from each mapped
9um vertex to the nearest point of the published 2.4um mesh.
"""
import json
import sys

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def load(d):
    x, y, z = (tifffile.imread(f"{d}/{c}.tif").astype(np.float64) for c in "xyz")
    ok = (x != -1) & (z > 0) & np.isfinite(x)
    return np.stack([x, y, z], -1), ok


src, ok_s = load(sys.argv[1]); dst, ok_d = load(sys.argv[2])
M = np.array(json.load(open(sys.argv[3]))["transformation_matrix"])
A, t = M[:, :3], M[:, 3]
mapped = (src[ok_s] - t) @ np.linalg.inv(A).T          # 9um xyz -> 2.4um xyz

# densify the published mesh 4x so nearest-vertex quantisation is ~1 voxel
H, W = dst.shape[:2]
gy, gx = np.mgrid[0:H - 1:0.25, 0:W - 1:0.25]
dense = np.stack([map_coordinates(dst[..., i], [gy, gx], order=1) for i in range(3)], -1)
dok = map_coordinates(ok_d.astype(float), [gy, gx], order=1) > 0.999
pts = dense[dok]
# normals of the dense grid
gyv = np.gradient(dense, axis=0); gxv = np.gradient(dense, axis=1)
nrm = np.cross(gxv, gyv); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
nrm = nrm[dok]
tree = cKDTree(pts)
dist, j = tree.query(mapped, k=1, distance_upper_bound=400)
hit = np.isfinite(dist)
d = mapped[hit] - pts[j[hit]]
dn = np.abs((d * nrm[j[hit]]).sum(-1))
dp = np.linalg.norm(d - (d * nrm[j[hit]]).sum(-1, keepdims=True) * nrm[j[hit]], axis=-1)
um = 2.403
print(f"mapped vertices {len(mapped)}, within 400 vox of the published mesh {hit.mean():.1%}")
for name, v in (("normal", dn), ("in-plane (incl. 1-vox grid quantisation)", dp)):
    print(f"{name:42s} median {np.median(v):.2f} vox ({np.median(v) * um:.1f} um), p90 {np.percentile(v, 90):.2f} vox ({np.percentile(v, 90) * um:.1f} um)")
