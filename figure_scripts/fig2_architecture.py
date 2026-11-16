"""Fig. 2: overall structure only, after Context-CrackNet Fig. 1.

Rules: figures/插图修改经验记录_2026-11-05.md, section 5.1.
Input (dashed box) -> encoder levels E0..E4 as 3-D slabs, slab height ~ resolution,
with the compared down-sampling operator D (the only accent colour) between levels
-> decoder column (decoder blocks with grey up-sampling pills) -> 1x1 convolution
-> predicted mask (dashed box). No channel numbers, no ruler.
v3.17: the slabs sit close together and each skip leaves its slab top and runs just
above the lower levels into its decoder block, so no line climbs across the figure;
a row of operator chips under the encoder shows that D is the part being swapped.
Structure follows resnet_unet.py / models.py. Drawn in millimetre coordinates.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from style3 import MM, ROOT, save

mpl.rcParams.update({"font.family": ["Times New Roman", "SimSun"],
                     "mathtext.fontset": "stix"})

ASSETS = ROOT / "figures" / "assets"
W_MM, H_MM = 180, 98
INK = "#262626"
GREY = "#555555"
EDGE = "#3A3A3A"
D_FC = ("#F08A3C", "#F7B98A", "#C8641E")       # front, top, side
ENC_FC = ("#C6DAF0", "#E1ECF8", "#97B5D6")
DEC_FC = ("#DCEBCF", "#EEF5E7", "#AFCB98")
OUT_FC = ("#A9D18E", "#C9E3B6", "#7FAE62")
PILL = "#E4E4E4"

fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H_MM)
ax.set_axis_off()
ax.set_autoscale_on(False)


def text(x, y, s, size=8, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    ax.text(x, y, s, fontsize=size, zorder=kw.pop("zorder", 6), **kw)


def box3d(x, y, w, h, fc, d=1.6, lw=0.6, z=3):
    """Front face at (x, y, w, h); top and right faces recede by d."""
    front, top, side = fc
    ax.add_patch(Rectangle((x, y), w, h, fc=front, ec=EDGE, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=top, ec=EDGE, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=side, ec=EDGE, lw=lw, zorder=z))


def arrow(pts, lw=0.8, color=INK):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="projecting", zorder=2)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=2))


def pill(xc, yc, s, w=15, h=4.2):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=PILL, ec=EDGE, lw=0.6, zorder=3))
    text(xc, yc, s, size=7.5)


def dashed_frame(x, y, w, h, pad=1.2):
    ax.add_patch(Rectangle((x - pad, y - pad), w + 2 * pad, h + 2 * pad, fill=False, ec=EDGE,
                           lw=0.6, ls=(0, (3, 2)), zorder=1))


# ------------------------------------------------------------------ legend (top left)
ax.add_patch(Rectangle((2, 63.9), 44, 32.1, fill=False, ec=EDGE, lw=0.6, ls=(0, (3, 2))))
box3d(4.5, 87.5, 5, 4.2, D_FC, d=1.2)
text(7.0, 89.6, "D", size=8, color="white", fontweight="bold")
text(13, 89.6, "Compared operator", size=7.5, ha="left")
box3d(4.5, 76.5, 5, 6.5, ENC_FC, d=1.2)
text(7.0, 79.75, "$E$", size=8)
text(13, 82.4, "Encoder block", size=7.5, ha="left")
text(13, 79.0, "UNet: 2 convolutions/level", size=7, ha="left", color=GREY)
text(13, 75.8, "ResNet18: 1 convolution at E₀;", size=7, ha="left", color=GREY)
text(13, 72.6, "2 residual blocks at E₁–E₄", size=7, ha="left", color=GREY)
text(13, 69.4, "ConvNeXt: 2 blocks/level", size=7, ha="left", color=GREY)
text(13, 66.2, "", size=7, ha="left", color=GREY)

# ------------------------------------------------------------------ encoder
BASE = 24
SW, SD = 4.4, 2.2                          # slab front width, depth
heights = [56, 43, 30, 14, 9]              # ~ spatial resolution, not to scale
# E0 left edge and E4 right edge stay where they were (50 and 117.6 mm); thinner slabs spread
# evenly in between so each E-D-E gap is wider and the D-to-E arrow shows in full
enc_x = [50 + k * (117.6 - SW - SD - 50) / 4 for k in range(5)]
for k in range(5):
    box3d(enc_x[k], BASE, SW, heights[k], ENC_FC, d=SD)
    text(enc_x[k] + SW / 2, BASE + heights[k] / 2, f"$E_{k}$", size=8)

# D between levels, low on the slabs
DW, DY = 3.6, BASE + 4.5
d_centres = []
for k in range(4):
    x0, x1 = enc_x[k] + SW + SD, enc_x[k + 1]
    xc = (x0 + x1) / 2 - 0.5              # slightly left: more room for the D-to-E arrow
    d_centres.append(xc)
    ax.plot([x0, xc - DW / 2], [DY, DY], color=INK, lw=0.8, zorder=2)
    box3d(xc - DW / 2, DY - 1.9, DW, 3.8, D_FC, d=0.8, z=4)
    text(xc, DY, "D", size=7, color="white", fontweight="bold", zorder=7)
    arrow([(xc + DW / 2 + 0.8, DY), (x1 - 0.3, DY)])

# ------------------------------------------------------------------ input image
img = plt.imread(str(ASSETS / "synthetic_crack.png"))
IW, IH = 27, 27 * 384 / 544
ix, iy = 6, 34
ax.imshow(img, extent=(ix, ix + IW, iy, iy + IH), zorder=3)
dashed_frame(ix, iy, IW, IH)
text(ix + IW / 2, iy + IH + 4.2, "Input image", size=8)
arrow([(ix + IW + 1.2, iy + IH / 2), (enc_x[0] - 0.4, iy + IH / 2)])

# ------------------------------------------------------------------ decoder column
DX, DBW, DBH = 123, 16, 4.6               # decoder block left, width, height
dcx = DX + DBW / 2
dec_y = [42, 58, 74, 90]                   # decoder blocks for levels 3, 2, 1, 0
up_y = [30, 50, 66, 82]                    # up-sampling pills, bottom to top
arrow([(enc_x[4] + SW + SD, up_y[0]), (dcx - 7.5, up_y[0])])
for i in range(4):
    pill(dcx, up_y[i], "Upsample")
    box3d(DX, dec_y[i] - DBH / 2, DBW, DBH, DEC_FC, d=1.3)
    text(dcx, dec_y[i], "Decoder", size=7.5)
    arrow([(dcx, up_y[i] + 2.1), (dcx, dec_y[i] - DBH / 2)])
    if i < 3:
        arrow([(dcx, dec_y[i] + DBH / 2 + 1.3), (dcx, up_y[i + 1] - 2.1)])
# skips: up from the top face of E_{3-i}, then right just above the lower levels
for i in range(4):
    k = 3 - i
    sx = enc_x[k] + SW / 2 + SD / 2
    arrow([(sx, BASE + heights[k] + SD), (sx, dec_y[i]), (DX - 0.2, dec_y[i])])
text(96, dec_y[3] + 2.6, "Skip connections (concatenation)", size=7, color=GREY)

# decoder pathway arrow
ax.add_patch(FancyArrowPatch((142.6, 26), (142.6, 86), arrowstyle="-|>", mutation_scale=7,
                             lw=0.6, color=INK))
text(145.2, 56, "Decoder path", size=7.5, rotation=90)

# ------------------------------------------------------------------ output
OX, OY, OW, OH = 152, dec_y[3] - 2.5, 18, 5
box3d(OX, OY, OW, OH, OUT_FC, d=1.6)
text(OX + OW / 2, OY + OH / 2, "1×1 conv.", size=7.5)
arrow([(DX + DBW + 1.3, dec_y[3]), (OX - 0.2, dec_y[3])])
mask = plt.imread(str(ASSETS / "synthetic_crack_mask.png"))
MW, MH = 25, 25 * 384 / 544
mx, my = OX + OW / 2 - MW / 2 + 0.8, 48
ax.imshow(mask, cmap="gray", vmin=0, vmax=1, extent=(mx, mx + MW, my, my + MH), zorder=3)
dashed_frame(mx, my, MW, MH)
arrow([(OX + OW / 2, OY), (OX + OW / 2, my + MH + 1.4)])
text(mx + MW / 2, my - 4.5, "Output mask (schematic)", size=8)

# ------------------------------------------------------------------ what D can be
CH_Y, CH_H = 9.5, 4.6
chips = ["max", "avg", "min", "Avg-Top2", "BTSP", "Strided conv.", "…"]
widths = [8.5, 8.5, 8.5, 12.5, 9.5, 16.5, 5.5]
gap = 1.6
total = sum(widths) + gap * (len(widths) - 1)
cx0 = (d_centres[0] + d_centres[-1]) / 2 - total / 2
bus_y = CH_Y + CH_H / 2 + 4.0
for xc in d_centres:                       # each D drops a thin lead to the bus
    ax.plot([xc, xc], [DY - 1.9, bus_y], color=D_FC[2], lw=0.6, ls=(0, (2, 1.5)), zorder=2)
ax.plot([d_centres[0], d_centres[-1]], [bus_y, bus_y], color=D_FC[2], lw=0.6, zorder=2)
mid = (d_centres[0] + d_centres[-1]) / 2
ax.plot([mid, mid], [bus_y, CH_Y + CH_H / 2 + 0.6], color=D_FC[2], lw=0.6, zorder=2)
x = cx0
for s, w in zip(chips, widths):
    if s == "…":
        text(x + w / 2, CH_Y, s, size=8, color=GREY)
    else:
        ax.add_patch(FancyBboxPatch((x, CH_Y - CH_H / 2), w, CH_H,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    fc="white", ec=D_FC[2], lw=0.7, zorder=3))
        text(x + w / 2, CH_Y, s, size=7)
    x += w + gap
text(mid, CH_Y - CH_H / 2 - 3.2, "One of 11 operators is used at all 4 locations in each run", size=7.5)

save(fig, "fig2_architecture")
