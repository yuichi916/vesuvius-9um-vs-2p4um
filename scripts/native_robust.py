import sys, json
import numpy as np, zarr, tifffile
from scipy.ndimage import affine_transform
sys.path.insert(0, ".")
from sweep_proto import auc
nat_pred, seg, nat_um, l2_um = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
s = nat_um / l2_um
sh = json.load(open(seg + "/native_compare.json"))["shift"]
rd = lambda p: np.asarray(zarr.open_array(tifffile.imread(p, aszarr=True), mode="r")[:]).astype(np.float32)
lab = np.asarray(zarr.open_array(seg + "/inklabels.zarr/2", mode="r")[:]) > 0
sup = np.asarray(zarr.open_array(seg + "/supervision.zarr/2", mode="r")[:]) > 0
H, W = lab.shape
p = rd(nat_pred); q = rd(seg + "/pred.tif")
to_l2 = lambda img, d: affine_transform(img, [1 / s, 1 / s], offset=[-d[0] / s, -d[1] / s], output_shape=(H, W), order=1)
cov = to_l2(np.ones_like(p), sh) > 0.99
mm = sup & cov & (q > 0)
best = {}
for dy in range(-2, 3):
    for dx in range(-2, 3):
        best[(dy, dx)] = auc(to_l2(p, (sh[0] + dy, sh[1] + dx))[mm], lab[mm])
k = max(best, key=best.get)
# round trip pooled pred through the native grid so both sides get the same interpolation
q_nat = affine_transform(q, [s, s], offset=[sh[0], sh[1]], output_shape=p.shape, order=1)
q_rt = to_l2(q_nat, sh)
print(json.dumps({"native_at_shift": best[(0, 0)], "native_best_local_shift": [k, best[k]],
                  "pooled": auc(q[mm], lab[mm]), "pooled_roundtrip": auc(q_rt[mm], lab[mm])}))
