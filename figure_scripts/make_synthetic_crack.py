"""Generate a synthetic concrete-crack image and its mask for schematic figures (Fig. 2 and Fig. S1).

The image is produced procedurally (random texture plus random-walk cracks), so it carries no third-party
copyright. It is used only for illustration; no experiment uses it.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "figures" / "assets"
W, H = 544, 384
rng = np.random.default_rng(20261116)


def smooth_noise(scale, amp):
    small = rng.normal(0, 1, (max(2, H // scale), max(2, W // scale)))
    img = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-9) * 255).astype(np.uint8))
    return (np.asarray(img.resize((W, H), Image.BICUBIC), float) / 255 - 0.5) * amp


tex = 0.62 + smooth_noise(64, 0.10) + smooth_noise(16, 0.08) + smooth_noise(4, 0.07) + rng.normal(0, 0.035, (H, W))
# Small aggregate pits.
for _ in range(260):
    y, x = rng.integers(0, H), rng.integers(0, W)
    r = rng.integers(1, 4)
    yy, xx = np.ogrid[-y:H - y, -x:W - x]
    tex[(yy * yy + xx * xx) <= r * r] -= rng.uniform(0.05, 0.18)


def walk(x0, y0, angle, steps, step=4.0, wiggle=0.35):
    pts, a = [(x0, y0)], angle
    for _ in range(steps):
        a += rng.normal(0, wiggle)
        a = 0.75 * a + 0.25 * angle
        x, y = pts[-1]
        pts.append((x + step * np.cos(a), y + step * np.sin(a)))
    return pts


mask = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(mask)
main = walk(-5, 170, 0.0, 150)
for i in range(len(main) - 1):
    w = int(round(9 + 3.5 * np.sin(i / 17) + rng.normal(0, 0.8)))
    draw.line([main[i], main[i + 1]], fill=255, width=max(3, w))
for start, ang, n in [(40, -0.9, 30), (85, 0.8, 26), (115, -0.6, 22)]:
    x0, y0 = main[start]
    br = walk(x0, y0, ang, n, step=3.5, wiggle=0.45)
    for i in range(len(br) - 1):
        draw.line([br[i], br[i + 1]], fill=255, width=max(1, int(3 - 2 * i / len(br)) + 1))
m = np.asarray(mask, float) / 255
soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.5)), float) / 255

img = tex * (1 - 0.95 * soft) + 0.02 * soft
img = np.clip(img, 0, 1)
rgb = np.stack([img * 1.0, img * 0.98, img * 0.94], -1)
OUT.mkdir(parents=True, exist_ok=True)
Image.fromarray((rgb * 255).astype(np.uint8)).save(OUT / "synthetic_crack.png")
Image.fromarray(((m > 0.5) * 255).astype(np.uint8)).save(OUT / "synthetic_crack_mask.png")
print("saved", OUT / "synthetic_crack.png")
