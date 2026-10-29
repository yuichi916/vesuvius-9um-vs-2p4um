"""Align a native-9um surface volume canvas to the 2.4um level-2 canvas (scale + shift) and score ink AUC on it.
usage: python native_align.py NATIVE_ZARR NATIVE_PRED_PREFIX SEG_DIR NATIVE_UM
"""
import sys, json
import numpy as np, zarr, tifffile
from scipy.ndimage import affine_transform
from numpy.fft import fft2, ifft2
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from sweep_proto import auc

nat_path, pred_prefix, seg, nat_um = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
l2_um = float(sys.argv[5]) if len(sys.argv) > 5 else 9.612
s = nat_um / l2_um   # native px -> L2 px
nat = zarr.open_array(nat_path + "/0", mode="r"); pooled = zarr.open_array(seg + "/pooled.zarr/0", mode="r")
A = np.asarray(nat[12:16]).astype(np.float32).mean(0); Bimg = np.asarray(pooled[9:12]).astype(np.float32).mean(0)
H, W = Bimg.shape
def to_l2(img, shift=(0.0, 0.0), order=1):
    # output L2 pixel p samples native at (p - shift) / s
    return affine_transform(img, [1 / s, 1 / s], offset=[-shift[0] / s, -shift[1] / s], output_shape=(H, W), order=order, cval=0)
def hp(x):
    from scipy.ndimage import gaussian_filter
    m = x > 0; x = x - gaussian_filter(x, 8); x[~m] = 0; return x
R = to_l2(A)
c = np.real(ifft2(fft2(hp(Bimg)) * np.conj(fft2(hp(R)))))
dy, dx = np.unravel_index(c.argmax(), c.shape); dy = (dy + H // 2) % H - H // 2; dx = (dx + W // 2) % W - W // 2
peak = c.max() / c.std()
R2 = to_l2(A, (dy, dx)); m = (R2 > 0) & (Bimg > 0)
r = np.corrcoef(hp(R2)[m], hp(Bimg)[m])[0, 1]
print(f"shift {dy},{dx} peak/std {peak:.1f} highpass corr after align {r:.3f}")
lab = np.asarray(zarr.open_array(seg + "/inklabels.zarr/2", mode="r")[:]) > 0
sup = np.asarray(zarr.open_array(seg + "/supervision.zarr/2", mode="r")[:]) > 0
out = {"shift": [int(dy), int(dx)], "corr": float(r)}
for tag in ["", "_reverse"]:
    p = np.asarray(zarr.open_array(tifffile.imread(pred_prefix + tag + ".tif", aszarr=True), mode="r")[:]).astype(np.float32)
    P = to_l2(p, (dy, dx))
    mm = sup & m & (to_l2(np.ones_like(p), (dy, dx)) > 0.99)
    out["auc" + (tag or "_fwd")] = auc(P[mm], lab[mm]); out["px"] = int(mm.sum())
    # pooled pred on the same pixels for a like-for-like comparison
    q = np.asarray(zarr.open_array(tifffile.imread(seg + "/pred" + tag + ".tif", aszarr=True), mode="r")[:]).astype(np.float32)
    out["pooled_auc" + (tag or "_fwd")] = auc(q[mm], lab[mm])
print(json.dumps(out))
json.dump(out, open(seg + "/native_compare.json", "w"))
