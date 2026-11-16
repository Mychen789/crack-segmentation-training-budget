"""Manuscript figures: reuse figstyle (data access, export to figures/) and set the colours and fonts."""

import figstyle
from figstyle import *  # noqa: F401,F403

# One palette for every figure, anchored on Figs 2 and 3. One meaning per colour:
#   orange = the operator a panel is about (the D block in Fig. 2; min in Figs 1, 4-6)
#   blue   = the reference it is compared with (stride conv, the window in Fig. 3, rank shading)
#   red    = conclusion changed / criterion failed / emphasis in Fig. 3
#   green  = criterion passed (the decoder green in Fig. 2)
#   grey   = everything else, in five fixed levels
PAL = {
    "orange": "#F08A3C", "orange_dark": "#C8641E", "orange_soft": "#F7B98A",
    "blue": "#1F5FBF", "blue_mid": "#97B5D6", "blue_soft": "#C6DAF0",
    "red": "#D62728", "red_soft": "#F6D5D2",
    "green": "#A9D18E", "green_soft": "#DCEBCF",
}
figstyle.C.update({
    "ink": "#262626", "dark": "#555555", "mid": "#8A8A8A", "light": "#D9D9D9", "faint": "#EFEFEF",
    "signal": PAL["red"], "signal_soft": PAL["red_soft"],
    "blue": PAL["blue"], "blue_soft": PAL["blue_soft"], "pass": PAL["green_soft"],
})

CHANGED = PAL["red"]            # conclusion changed / criterion failed
FOCUS = PAL["blue"]             # the reference operator or quantity
OPER = PAL["orange"]            # the operator a panel is about (min)
BTSP_INK = figstyle.C["ink"]    # the preregistered operator
BAND = figstyle.C["faint"]      # +/- smallest effect size of interest
PASS_BG = PAL["green_soft"]
FAIL_BG = PAL["red_soft"]
NA_BG = "#F5F5F5"


def tag(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", ha="left", va="bottom")


def panel(ax, letter, title, pad=4, x=0.0):
    """Panel label in one line above the axes: bold (a) followed by the panel title."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import offset_copy
    import matplotlib as mpl

    fs = FontProperties(size=mpl.rcParams["axes.titlesize"]).get_size_in_points()
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pad, units="points")
    ax.text(x, 1, f"({letter})", transform=tr, fontsize=fs, fontweight="bold",
            ha="left", va="baseline")
    tr2 = offset_copy(ax.transAxes, fig=ax.figure, x=fs * 1.45 + 3, y=pad, units="points")
    ax.text(x, 1, title, transform=tr2, fontsize=fs, ha="left", va="baseline")


# Chinese manuscript: SimSun for CJK, Times New Roman for Latin and digits (per-glyph fallback).
import matplotlib as _mpl  # noqa: E402

_mpl.rcParams.update({"font.family": ["Times New Roman"], "mathtext.fontset": "stix",
                      "axes.unicode_minus": True})
