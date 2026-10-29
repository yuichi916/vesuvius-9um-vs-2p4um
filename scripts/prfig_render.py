"""PR figure: published vs Python-rendered PHerc0139 w035 (middle plane, |difference|) and ink_9um on each."""
import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw, ImageFont

D = "E:/vesuvius/prfig"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
pub = np.asarray(zarr.open(f"{D}/w035_pub.zarr", mode="r")["0"][14]).astype(np.float32)
mine = np.asarray(zarr.open(f"{D}/w035_mine.zarr", mode="r")["0"][14]).astype(np.float32)


def pred(name):
    fwd = tifffile.imread(f"{D}/pred_{name}.tif").astype(np.float32)
    rev = tifffile.imread(f"{D}/pred_{name}_reverse.tif").astype(np.float32)
    return fwd, rev


pf, pr = pred("pub"); mf, mr = pred("mine")
print("pred shapes", pf.shape, mf.shape, "dtype", tifffile.imread(f"{D}/pred_pub.tif").dtype)
for tag, a, b in (("forward", pf, mf), ("reverse", pr, mr)):
    print(tag, "pred r", round(np.corrcoef(a.ravel(), b.ravel())[0, 1], 4), "mean |d|/range",
          round(float(np.abs(a - b).mean() / (a.max() - a.min() + 1e-9)), 4))
print("plane 14 r", round(np.corrcoef(pub.ravel(), mine.ravel())[0, 1], 4), "mean |d|", round(float(np.abs(pub - mine).mean()), 2))

# show the direction with the stronger response
use_rev = pr.std() > pf.std()
a, b = (pr, mr) if use_rev else (pf, mf)
lo, hi = np.percentile(a, [1, 99.5])
g = lambda x: np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
diff = np.clip(np.abs(pub - mine) * 4, 0, 255).astype(np.uint8)
tiles = [(pub.astype(np.uint8), "published render, plane 14"), (mine.astype(np.uint8), "this PR, plane 14"),
         (diff, "|difference| x4"), (g(a), "ink_9um on published"), (g(b), "ink_9um on this PR")]
S, gap, top = 640, 14, 50
canvas = Image.new("L", (3 * S + 2 * gap, 2 * (S + top) + gap), 255)
d = ImageDraw.Draw(canvas)
for i, (im, label) in enumerate(tiles):
    r, c = divmod(i, 3)
    x, y = c * (S + gap), r * (S + top + gap)
    canvas.paste(Image.fromarray(im).resize((S, S), Image.LANCZOS), (x, y + top))
    d.text((x + 4, y + 8), label, fill=0, font=FONT)
d.text((2 * (S + gap) + 4, S + top + gap + top + 10),
       "PHerc0139 w035, 2048 x 2048 px window\n(19 x 19 mm), seed42+seed43\nstep-075000, " +
       ("reverse" if use_rev else "forward") + " direction", fill=0, font=FONT)
canvas.save(f"{D}/w035_render_vs_published.png")
print("direction", "reverse" if use_rev else "forward")
