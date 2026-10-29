"""Stack: surface render (mean of layers around the surface) over ink predictions.

usage: python view_pred.py VOLUME_ZARR OUT_PNG PRED_TIF [PRED_TIF...] [--ds 4] [--layers 10 18]
Predictions are display-rescaled with (p - 0.25) / 0.5 (ink_9um label smoothing floor).
"""
import argparse

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("volume"); ap.add_argument("out"); ap.add_argument("preds", nargs="*")
ap.add_argument("--ds", type=int, default=4)
ap.add_argument("--layers", type=int, nargs=2, default=[10, 18])
ap.add_argument("--roi", type=int, nargs=4)
ap.add_argument("--vertical", action="store_true")
a = ap.parse_args()

vol = zarr.open_array(a.volume + "/0", mode="r")
D, H, W = vol.shape
y0, y1, x0, x1 = a.roi if a.roi else (0, H, 0, W)
ds = a.ds


def small(arr):
    im = Image.fromarray(arr)
    return im.resize((max(1, arr.shape[1] // ds), max(1, arr.shape[0] // ds)), Image.BILINEAR)


v = np.asarray(vol[a.layers[0]:a.layers[1], y0:y1, x0:x1]).astype(np.float32).mean(0)
nz = v[v > 0]
lo, hi = np.percentile(nz, [1, 99]) if nz.size else (0, 255)
panels = [("render", small(np.clip((v - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)))]
for p in a.preds:
    pr = zarr.open_array(tifffile.imread(p, aszarr=True), mode="r")
    q = np.asarray(pr[y0:y1, x0:x1]).astype(np.float32)
    if q.max() > 1.5:
        q /= 255.0
    panels.append((p.replace("\\", "/").split("/")[-1], small(np.clip((q - 0.25) / 0.5 * 255, 0, 255).astype(np.uint8))))
w, h = panels[0][1].size
gap = 24
if a.vertical:
    canvas = Image.new("L", (w, (h + gap) * len(panels)), 255)
else:
    canvas = Image.new("L", ((w + 10) * len(panels), h + gap), 255)
dr = ImageDraw.Draw(canvas)
for i, (name, im) in enumerate(panels):
    ox, oy = (0, i * (h + gap)) if a.vertical else (i * (w + 10), 0)
    dr.text((ox + 4, oy + 4), name, fill=0)
    canvas.paste(im, (ox, oy + gap))
canvas.save(a.out)
print("saved", a.out, canvas.size)
