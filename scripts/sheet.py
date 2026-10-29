"""Crop each segment to its rendered area; rows of render | fwd | rev. usage: sheet.py OUT_PREFIX DIR SEG [SEG...]"""
import sys, numpy as np, zarr, tifffile
from PIL import Image, ImageDraw
out, d, segs = sys.argv[1], sys.argv[2], sys.argv[3:]
DS = 4; rows = []
for s in segs:
    v = zarr.open_array(f"{d}/{s}.zarr/0", mode="r"); D = v.shape[0]
    mid = np.asarray(v[D // 2]); ys, xs = np.nonzero(mid[::8, ::8])
    y0, y1, x0, x1 = ys.min() * 8, (ys.max() + 1) * 8, xs.min() * 8, (xs.max() + 1) * 8
    r = np.asarray(v[D // 2 - 2:D // 2 + 2, y0:y1, x0:x1]).astype(np.float32).mean(0)
    nz = r[r > 0]; lo, hi = np.percentile(nz, [1, 99])
    ims = [np.clip((r - lo) / (hi - lo) * 255, 0, 255)]
    for tag in ("", "_reverse"):
        p = np.asarray(zarr.open_array(tifffile.imread(f"{d}/pred_{s}{tag}.tif", aszarr=True), mode="r")[y0:y1, x0:x1]).astype(np.float32)
        if p.max() > 1.5: p /= 255
        ims.append(np.clip((p - 0.25) / 0.5 * 255, 0, 255) * (r > 0))
    ims = [Image.fromarray(i.astype(np.uint8)).resize(((x1 - x0) // DS, (y1 - y0) // DS), Image.BILINEAR) for i in ims]
    w, h = ims[0].size; row = Image.new("L", (3 * w + 20, h + 16), 255); dr = ImageDraw.Draw(row)
    dr.text((2, 2), f"{s}  render | fwd | rev   ({(x1-x0)*0.0096:.0f}x{(y1-y0)*0.0096:.0f} mm)", fill=0)
    for i, im in enumerate(ims): row.paste(im, (i * (w + 10), 16))
    rows.append(row)
# pack rows into pages of max ~2400 px tall
page, H, n = [], 0, 0
def flush(page, n):
    W = max(r.width for r in page); c = Image.new("L", (W, sum(r.height for r in page)), 255); y = 0
    for r in page: c.paste(r, (0, y)); y += r.height
    c.save(f"{out}_{n}.png"); print("saved", f"{out}_{n}.png", c.size)
for r in rows:
    if page and H + r.height > 2400: flush(page, n); n += 1; page, H = [], 0
    page.append(r); H += r.height
if page: flush(page, n)
