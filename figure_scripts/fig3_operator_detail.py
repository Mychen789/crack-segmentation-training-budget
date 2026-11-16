"""Fig. 3: what the 2x2 operators do on a real crack image, after CBRF Figs. 3-4.

Rules: figures/插图修改经验记录_2026-11-05.md, section 5.2.
The operators are applied to an illustrative non-negative map, r = max(0, (t - g) / t),
with g the grey level of a synthetic crack image (tools_fig/make_synthetic_crack.py) and t its 13th percentile, rounded to 0.01.
The clipping mimics post-ReLU features (most values zero); it is not a network feature map.
alpha = 0.75 is for illustration only (the trained alpha is learned, initial value 0.5).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
from PIL import Image

from style3 import MM, PAL, ROOT, C, save

mpl.rcParams.update({"font.family": ["Times New Roman", "SimSun"],
                     "mathtext.fontset": "stix"})

ASSETS = ROOT / "figures" / "assets"
INK, GREY = C["ink"], C["dark"]
RED, BLUE = PAL["red"], PAL["blue"]
ALPHA = 0.75
Y0, X0, S = 122, 338, 24          # zoomed patch in the image (rows, cols, size in pixels)
WI, WJ = 7, 6                     # the highlighted 2x2 window, in pooled coordinates

rgb = np.asarray(Image.open(ASSETS / "synthetic_crack.png").convert("RGB"))
g = np.asarray(Image.open(ASSETS / "synthetic_crack.png").convert("L"), float) / 255
t = np.percentile(g, 13)
r = np.round(np.clip((t - g) / t, 0, 1), 2)   # two decimals, so the printed window arithmetic is exact


def windows(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return np.sort(a[:h, :w].reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4), -1)


def pool(s):
    top2 = s[..., 2:].mean(-1)
    low2 = s[..., :2].mean(-1)
    return {"max": s[..., 3], "avg": s.mean(-1), "Avg-Top2": top2,
            "BTSP": ALPHA * top2 + (1 - ALPHA) * low2, "diff": (1 - ALPHA) * low2}


full = pool(windows(r))
share_zero = float((full["diff"] == 0).mean())
P = r[Y0:Y0 + S, X0:X0 + S]
out = pool(windows(P))
win = P[2 * WI:2 * WI + 2, 2 * WJ:2 * WJ + 2]
srt = np.sort(win.ravel())

W_MM, H_MM = 180, 84
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
base = fig.add_axes([0, 0, 1, 1])
base.set_xlim(0, W_MM)
base.set_ylim(0, H_MM)
base.set_axis_off()


def img_ax(x, y, w, h):
    a = fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(INK)
    return a


def text(x, y, s, size=8, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    base.text(x, y, s, fontsize=size, **kw)


def arrow(p, q, color=INK, lw=0.8):
    base.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                   shrinkA=0, shrinkB=0))


def link(a_from, xy_from, a_to, xy_to, color):
    fig.add_artist(ConnectionPatch(xyA=xy_from, coordsA=a_from.transData, xyB=xy_to, coordsB=a_to.transData,
                                   color=color, lw=0.6))


# ------------------------------------------------------------------ top row
TOP = 46
a_in = img_ax(3, TOP, 46, 46 * 384 / 544)
a_in.imshow(rgb)
a_in.add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), S, S, fill=False, ec=RED, lw=0.9))
text(26, TOP - 3.5, "Synthetic crack image (schematic)")

ZS = 30
a_z = img_ax(58, TOP + 1.5, ZS, ZS)
a_z.imshow(rgb[Y0:Y0 + S, X0:X0 + S], interpolation="nearest")
for sp in a_z.spines.values():
    sp.set_color(RED)
    sp.set_linewidth(0.9)
link(a_in, (X0 + S - 0.5, Y0 - 0.5), a_z, (-0.5, -0.5), RED)
link(a_in, (X0 + S - 0.5, Y0 + S - 0.5), a_z, (-0.5, S - 0.5), RED)
text(73, TOP - 3.5, "Local crop (24 × 24 pixels)")

a_r = img_ax(97, TOP + 1.5, ZS, ZS)
a_r.imshow(P, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
a_r.add_patch(Rectangle((2 * WJ - 0.5, 2 * WI - 0.5), 2, 2, fill=False, ec=BLUE, lw=1.1))
arrow((89.2, TOP + 16.5), (95.8, TOP + 16.5))
text(112, TOP - 3.5, "Nonnegative response (schematic)")

# window detail
cs = 5.6
wx, wy = 136, TOP + 19
for i in range(2):
    for j in range(2):
        v = win[i, j]
        base.add_patch(Rectangle((wx + j * cs, wy + (1 - i) * cs), cs, cs, fc=plt.cm.gray(v), ec=BLUE, lw=0.9))
        text(wx + j * cs + cs / 2, wy + (1 - i) * cs + cs / 2, f"{v:.2f}", size=6.5,
             color=INK if v > 0.5 else "white")
link(a_r, (2 * WJ + 1.5, 2 * WI - 0.5), base, (wx, wy + 2 * cs), BLUE)
link(a_r, (2 * WJ + 1.5, 2 * WI + 1.5), base, (wx, wy), BLUE)
arrow((wx + 2 * cs + 1.5, wy + cs), (wx + 2 * cs + 7.5, wy + cs))
text(wx + 2 * cs + 4.5, wy + cs + 2.2, "sort", size=7, color=GREY)
sx = wx + 2 * cs + 9
for k, v in enumerate(srt):
    base.add_patch(Rectangle((sx + k * cs, wy + cs / 2), cs, cs, fc=plt.cm.gray(v), ec=INK, lw=0.6))
    text(sx + k * cs + cs / 2, wy + cs, f"{v:.2f}", size=6.5, color=INK if v > 0.5 else "white")
    text(sx + k * cs + cs / 2, wy + cs / 2 - 2.4, f"$x_{{({k + 1})}}$", size=7, color=GREY)
text(sx + cs, wy + cs / 2 - 6.2, "Two smallest = 0", size=7, color=RED)
base.plot([sx + 0.4, sx + 2 * cs - 0.4], [wy + cs / 2 - 4.3] * 2, color=RED, lw=0.7)

top2 = srt[2:].mean()
lines = [("max", f"{srt[3]:.2f}"), ("avg", f"{srt.mean():.2f}"), ("Avg-Top2", f"{top2:.2f}"),
         ("BTSP", f"{ALPHA * top2 + (1 - ALPHA) * srt[:2].mean():.2f} = α × Avg-Top2")]
for k, (name, val) in enumerate(lines):
    yy = wy - 11 - k * 4.4
    text(wx, yy, name, size=7.5, ha="left")
    text(wx + 15, yy, "= " + val, size=7.5, ha="left", color=RED if name == "BTSP" else INK)

# ------------------------------------------------------------------ bottom row: pooled patch
BS, BY = 24, 7
names = ["max", "avg", "Avg-Top2", "BTSP", "diff"]
labels = ["max", "avg", "Avg-Top2", "BTSP", "BTSP − α × Avg-Top2"]
bx = [8, 42, 76, 110, 148]
for k, n in enumerate(names):
    a = img_ax(bx[k], BY, BS, BS)
    a.imshow(out[n], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.add_patch(Rectangle((WJ - 0.5, WI - 0.5), 1, 1, fill=False, ec=BLUE, lw=1.0))
    text(bx[k] + BS / 2, BY - 3.2, labels[k], size=7.5)
text(bx[4] + BS / 2, BY + BS + 2.6, f"Zero difference in {share_zero:.0%} of windows", size=7, color=RED)
base.plot([bx[4] - 6, bx[4] - 6], [BY - 1, BY + BS + 1], color=C["mid"], lw=0.6, ls=(0, (2, 2)))
text(8, BY + BS + 3.4, "Outputs after 2 × 2, stride-2 downsampling of the response (box marks the window above)",
     size=7.5, ha="left")

save(fig, "fig3_operator_detail")
print("alpha", ALPHA, "t", round(t, 4), "share of windows with zero lower tail", round(share_zero, 4))
print("window", win.round(3).tolist())
