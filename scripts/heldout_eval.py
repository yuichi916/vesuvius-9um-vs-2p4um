"""Per-tile held-out evaluation: ink AUC vs label-free depth-profile features.

usage: python heldout_eval.py SEG_DIR OUT_JSON
SEG_DIR holds pooled.zarr (21-plane 9.6 um input), pred.tif, pred_reverse.tif,
inklabels.zarr and supervision.zarr (level "2" matches the pooled canvas).
"""
import json
import sys

import numpy as np
import tifffile
import zarr

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from sweep_proto import auc  # noqa: E402

T = 128


def read_pred(path):
    a = zarr.open_array(tifffile.imread(path, aszarr=True), mode="r")
    return np.asarray(a[:]).astype(np.float32)


def profile_features(v):
    """v: (D, T, T) float32. Depth-profile features normalised by tile std."""
    D = v.shape[0]
    prof = v.reshape(D, -1).mean(1)
    sd = max(float(v.std()), 1e-6)
    p = (prof - prof.mean()) / sd
    return {"contrast": float(p.max() - p.min()), "argmax": int(p.argmax()), "argmin": int(p.argmin()),
            "front_minus_back": float(p[: D // 3].mean() - p[-(D // 3):].mean()),
            "center_minus_edges": float(p[D // 3: 2 * D // 3].mean() - 0.5 * (p[: D // 3].mean() + p[-(D // 3):].mean()))}


def main():
    seg, out_json = sys.argv[1], sys.argv[2]
    vol = zarr.open_array(seg + "/pooled.zarr/0", mode="r")
    lab = np.asarray(zarr.open_array(seg + "/inklabels.zarr/2", mode="r")[:]) > 0
    sup = np.asarray(zarr.open_array(seg + "/supervision.zarr/2", mode="r")[:]) > 0
    fwd, rev = read_pred(seg + "/pred.tif"), read_pred(seg + "/pred_reverse.tif")
    H, W = lab.shape
    assert vol.shape[1:] == (H, W) == fwd.shape, (vol.shape, lab.shape, fwd.shape)
    s = sup & (np.asarray(vol[vol.shape[0] // 2]) > 0)
    res = {"seg": seg, "auc_fwd": auc(fwd[s], lab[s]), "auc_rev": auc(rev[s], lab[s]),
           "ink_frac": float(lab[s].mean()), "sup_px": int(s.sum()), "tiles": []}
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            m = s[y:y + T, x:x + T]
            if m.mean() < 0.9:
                continue
            l = lab[y:y + T, x:x + T][m]
            if not 0.03 < l.mean() < 0.97:
                continue
            v = np.asarray(vol[:, y:y + T, x:x + T]).astype(np.float32)
            rec = {"y": y, "x": x, "ink": float(l.mean()),
                   "auc_fwd": auc(fwd[y:y + T, x:x + T][m], l), "auc_rev": auc(rev[y:y + T, x:x + T][m], l)}
            rec.update(profile_features(v))
            res["tiles"].append(rec)
    json.dump(res, open(out_json, "w"))
    print(f"{seg}: AUC fwd {res['auc_fwd']:.4f} rev {res['auc_rev']:.4f} ink {res['ink_frac']:.3f} tiles {len(res['tiles'])}")


if __name__ == "__main__":
    main()
