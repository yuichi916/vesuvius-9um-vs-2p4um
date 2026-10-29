"""Checkerboard of PHerc1203's 9.362 um scan and its 2.403 um scan resampled through the automatic transform."""
import json

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/"
FIXED = B + "20250820131727-9.362um-1.2m-113keV-masked.zarr"
MOVING = B + "20260319130212-2.403um-0.2m-77keV-masked.zarr"
T = json.load(open("C:/projects/vesuvius/submission/transforms/pherc1203_2p403_to_9p362.json"))
M = np.array(T["transformation_matrix"]); A, t = M[:, :3], M[:, 3]
OUT = "E:/vesuvius/prfig"

fixed = zarr.open(FIXED, mode="r")["1"]      # 18.7 um
moving = zarr.open(MOVING, mode="r")["3"]    # 19.2 um
print(fixed.shape, moving.shape)

# fixed level-1 plane through the middle of the 2.4 um band
zc_m = (zarr.open(MOVING, mode="r")["0"].shape[0] - 1) / 2
zc_f0 = (A @ np.array([0, 0, zc_m]) + t)[2]
z1 = int(round((zc_f0 + 0.5) / 2 - 0.5))
S = 768
y0, x0 = 1842 - S // 2, 1722 - S // 2
F = np.asarray(fixed[z1, y0:y0 + S, x0:x0 + S]).astype(np.float32)

# level-1 fixed pixel centres -> level-0 fixed XYZ -> level-0 moving XYZ -> level-3 moving ZYX
yy, xx = np.mgrid[y0:y0 + S, x0:x0 + S].astype(np.float64)
pf = np.stack([(xx + 0.5) * 2 - 0.5, (yy + 0.5) * 2 - 0.5, np.full_like(xx, (z1 + 0.5) * 2 - 0.5)], -1)
pm = (pf - t) @ np.linalg.inv(A).T
c3 = (pm + 0.5) / 8 - 0.5                       # x, y, z at moving level 3
lo = np.floor(c3.reshape(-1, 3).min(0)).astype(int) - 1
hi = np.ceil(c3.reshape(-1, 3).max(0)).astype(int) + 2
blk = np.asarray(moving[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
Mv = map_coordinates(blk, [c3[..., 2] - lo[2], c3[..., 1] - lo[1], c3[..., 0] - lo[0]], order=1)
print("fixed z1", z1, "moving block", lo, hi)


def norm(a):
    v = a[a > 0]
    p1, p99 = np.percentile(v, [1, 99.5]) if v.size else (0, 1)
    return np.clip((a - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


Fn, Mn = norm(F), norm(Mv)
tile = 96
cb = ((np.arange(S)[:, None] // tile + np.arange(S)[None, :] // tile) % 2).astype(bool)
check = np.where(cb, Mn, Fn)
r = np.corrcoef(F[(F > 0) & (Mv > 0)], Mv[(F > 0) & (Mv > 0)])[0, 1]
print("pixel r", round(r, 3))

gap, top = 12, 50
font = ImageFont.truetype("arial.ttf", 28)
canvas = Image.new("L", (S * 3 + gap * 2, S + top), 255)
for i, im in enumerate((Fn, Mn, check)):
    canvas.paste(Image.fromarray(im), (i * (S + gap), top))
d = ImageDraw.Draw(canvas)
for i, s in enumerate(("PHerc1203 9.362 um scan", "2.403 um scan, auto transform", f"checkerboard ({tile} px tiles)")):
    d.text((i * (S + gap) + 4, 10), s, fill=0, font=font)
bar = int(round(2000 / 18.72))                   # 2 mm at level 1
x, y = S * 2 + gap * 2 + 20, S + top - 30
d.rectangle((x, y, x + bar, y + 7), fill=255)
d.text((x, y - 34), "2 mm", fill=255, font=font)
canvas.save(f"{OUT}/pherc1203_registration_checkerboard.png")
