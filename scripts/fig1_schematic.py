"""Fig. 1: schematic of the construction.
(a) Embedding sketch of the near-extremal magnetically charged pair: the near-AdS2 x S2 throat of radius r_e between
the two boundaries of the strip, the mouths opening into the two asymptotic regions, magnetic field lines threading
the throat and a crossing null ray.
(b) The throat in global coordinates (sigma across, global time t upward, null rays at 45 degrees). Outside each
boundary, one lane holds the opening channels O^j and one the holding channels P^j; the filled width of a lane is
the coupling strength at that global time, from the stabilized run of transfer_model.py. Each crossing ray is a
ribbon whose two widths are the shares of its negative null energy from the two sectors (rays_fig in
results_transfer.json). The mercury switch and the hydrogen maser are control devices drawn outside the strip."""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse, Polygon, Rectangle, FancyBboxPatch
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import transfer_model as T

apply_style()
FS, FS2 = 7.0, 6.4                 # label and secondary text sizes (pt)
INK, INK2 = "#000000", "#404040"   # text tokens
SURF, OPEN_ = "#EFEFEF", "#D9D9D9"  # surface and mouth-opening fills
RING, FIELD = "#A8A8A8", "#5A5A5A"

W, H = TEXTWIDTH_IN - 0.05, 3.0      # the saved page adds 0.02 in on each side and stays within the text width
B0 = 0.075                         # bottom margin (figure fraction) for the panel letters
fig = plt.figure(figsize=(W, H))
wa, ha = W / 2.0, H * (1.0 - B0)   # each panel in inches
ax_a = fig.add_axes([0.0, B0, 0.5, 1.0 - B0])
ax_b = fig.add_axes([0.5, B0, 0.5, 1.0 - B0])
for ax in (ax_a, ax_b):
    ax.set_xlim(0, wa); ax.set_ylim(0, ha); ax.set_aspect("equal"); ax.axis("off")


def leader(ax, p, q, color=INK2, lw=0.5, arrow=False, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>" if arrow else "-", color=color, lw=lw, ls=ls, mutation_scale=5.5,
        shrinkA=0, shrinkB=0))


# ============================================================================================ (a) geometry
a = ax_a
y0 = 1.33                         # axis of the throat
xm = wa / 2.0
re_, Lt, Lb, Rr = 0.15, 1.25, 0.86, 0.92      # throat radius, throat length, bell length, rim radius
xa, xb = xm - Lt / 2, xm + Lt / 2             # the two boundaries of the strip
lam = Lb / math.acosh(Rr / re_)
ep = 0.28                                      # perspective factor of the cross sections


def r_of(x):
    s = np.maximum(0.0, np.maximum(xa - x, x - xb))
    return re_ * np.cosh(s / lam)


xs = np.linspace(xa - Lb, xb + Lb, 1200)
rs = r_of(xs)
g_of = lambda x: np.tanh((np.asarray(x) - xm) / 0.45)   # foreshortening seen from the front of the center
xL, xR = xa - Lb, xb + Lb                                # rims of the mouths
ps = np.linspace(0, np.pi, 200)
# surface: silhouettes closed by the front halves of the two rims
outline = np.r_[np.c_[xs, y0 + rs],
                np.c_[xR + ep * Rr * g_of(xR) * np.sin(ps), y0 + Rr * np.cos(ps)],
                np.c_[xs[::-1], y0 - rs[::-1]],
                np.c_[xL + ep * Rr * g_of(xL) * np.sin(ps[::-1]), y0 + Rr * np.cos(ps[::-1])]]
a.add_patch(Polygon(outline, closed=True, fc=SURF, ec=INK, lw=0.8, zorder=2, joinstyle="round"))


def ring(x, r, color, lw, z, back=True, back_ls=(0, (1.2, 1.4))):
    q = np.linspace(0, 2 * np.pi, 400)
    X, Y = x + ep * r * float(g_of(x)) * np.sin(q), y0 + r * np.cos(q)
    front = np.sin(q) >= 0
    a.plot(np.where(front, X, np.nan), np.where(front, Y, np.nan), color=color, lw=lw, zorder=z,
           solid_capstyle="butt")
    if back:
        a.plot(np.where(~front, X, np.nan), np.where(~front, Y, np.nan), color=color, lw=lw * 0.9, ls=back_ls,
               zorder=z)


for s_ in (0.36, 0.62):                      # cross sections of the mouths
    for x in (xa - s_, xb + s_):
        ring(x, float(r_of(np.array([x]))[0]), RING, 0.45, 3)
for x in (xm - 0.31, xm + 0.31):             # cross sections of the throat
    ring(x, re_, RING, 0.45, 3)
for x in (xa, xb):                           # the boundaries sigma = 0 and sigma = pi
    ring(x, re_, INK, 0.95, 6, back_ls=(0, (1.0, 1.1)))
for x in (xL, xR):                           # hidden halves of the rims
    ring(x, Rr, INK, 0.55, 3, back_ls=(0, (1.0, 1.6)))

# magnetic field lines: front meridians of the surface, threading the throat from L to R
for psi in np.radians([38, 70, 110, 142]):
    X = xs + ep * rs * g_of(xs) * math.sin(psi)
    Y = y0 + rs * math.cos(psi)
    a.plot(X, Y, color=FIELD, lw=0.55, zorder=4)
    for xc in (xa - 0.50, xm, xb + 0.50):
        i = np.searchsorted(xs, xc)
        leader(a, (X[i - 6], Y[i - 6]), (X[i + 6], Y[i + 6]), color=FIELD, lw=0.55, arrow=True)

# crossing null ray along the front meridian
xr0, xr1 = xm - 1.37, xm + 1.37
a.plot([xr0, xr1 - 0.05], [y0, y0], color=INK, lw=1.2, zorder=8, solid_capstyle="butt")
leader(a, (xr1 - 0.12, y0), (xr1, y0), color=INK, lw=1.2, arrow=True)

# labels
a.text(0.03, 2.50, "asymptotic region L", fontsize=FS, color=INK, ha="left", va="center")
a.text(wa - 0.03, 2.50, "asymptotic region R", fontsize=FS, color=INK, ha="right", va="center")
for x, lab, dx in ((xa, "boundary L, $\\sigma=0$", -0.17), (xb, "boundary R, $\\sigma=\\pi$", 0.17)):
    a.text(x + dx, 2.06, lab, fontsize=FS, color=INK, ha="center", va="center")
    leader(a, (x, 2.00), (x, y0 + re_ + 0.015))
a.text(xm, y0 + re_ + 0.24, "near-AdS$_2\\times S^2$", fontsize=FS, color=INK, ha="center", va="bottom")
a.text(xm, y0 + re_ + 0.08, "throat, radius $r_{\\rm e}$", fontsize=FS, color=INK, ha="center", va="bottom")
a.text(xm - 0.15, 0.93, "crossing null ray", fontsize=FS, color=INK, ha="center", va="top")
leader(a, (xm - 0.15, 0.95), (xm - 0.15, y0 - 0.012))
xf = xm + 0.45
a.text(xf, 0.69, "magnetic field line", fontsize=FS, color=INK, ha="center", va="top")
yf = y0 + re_ * math.cos(math.radians(142))
leader(a, (xf, 0.71), (xf + ep * re_ * float(g_of(xf)) * math.sin(math.radians(142)), yf - 0.01))
a.text(0.50, 0.30, "magnetically charged", fontsize=FS, color=INK, ha="center", va="center")
a.text(0.50, 0.17, "mouth L", fontsize=FS, color=INK, ha="center", va="center")
a.text(wa - 0.50, 0.30, "magnetically charged", fontsize=FS, color=INK, ha="center", va="center")
a.text(wa - 0.50, 0.17, "mouth R", fontsize=FS, color=INK, ha="center", va="center")

# ============================================================================================ (b) couplings
b = ax_b
res = json.load(open(os.path.join(HERE, "results_transfer.json")))
rows = np.array(res["S3"]["rays_fig"]["open_hold_f"])      # t_in, u_in, u_out, opening, holding, constraint
pick = rows[0:37:6]
share = pick[:, 4] / (pick[:, 3] + pick[:, 4])               # holding share of each ray's null energy
R = T.run(T.protocols()["open_hold_f"])
t_lo, t_hi = pick[0, 0] - 0.25, pick[-1, 0] + math.pi + 0.25
yb, yt = 0.33, 2.58
st = (yt - yb) / (t_hi - t_lo)                                # inches per unit of global time
Ws = math.pi * st                                             # 45-degree null rays
xs0, xs1 = wa / 2 - Ws / 2, wa / 2 + Ws / 2
Y = lambda t: yb + (np.asarray(t) - t_lo) * st
Xs = lambda sg: xs0 + np.asarray(sg) / math.pi * Ws

b.add_patch(Rectangle((xs0, yb), Ws, yt - yb, fc=SURF, ec="none", zorder=1))
for x in (xs0, xs1):
    b.plot([x, x], [yb, yt], color=INK, lw=1.0, zorder=6, solid_capstyle="butt")

# channel lanes outside both boundaries: holding lane next to the boundary, opening lane outside it
wl, gap = 0.12, 0.035
lanes = {"PL": (xs0 - gap - wl, -1), "OL": (xs0 - 2 * gap - 2 * wl, -1),
         "PR": (xs1 + gap, +1), "OR": (xs1 + 2 * gap + wl, +1)}
m = (R["t"] >= t_lo) & (R["t"] <= t_hi) & (np.arange(R["t"].size) % 10 == 0)   # every 10th sample
tt, yy = R["t"][m], Y(R["t"][m])
prof = {"O": R["etas"][0][m] / T.P["eta_o"], "P": R["etas"][1][m] / R["etas"][1][m].max()}
colr = {"O": COL["red"], "P": COL["green"]}
for key, (x0, side) in lanes.items():
    b.add_patch(Rectangle((x0, yb), wl, yt - yb, fc="none", ec="#B8B8B8", lw=0.4, zorder=2))
    f = np.clip(prof[key[0]], 0, 1) * wl
    inner = x0 + wl if side < 0 else x0
    b.fill_betweenx(yy, inner, inner + side * f, color=colr[key[0]], lw=0, zorder=3)
    lab = "$O^j$" if key[0] == "O" else "$P^j$"
    b.text(x0 + wl / 2, yt + 0.035, lab, fontsize=FS, color=INK, ha="center", va="bottom")

# crossing rays as two-colored ribbons: opening share on the lower side, holding share on the upper side
Wr = 2.7
strip_clip = Rectangle((xs0, yb), Ws, yt - yb, transform=b.transData)
for (t_in, *_), h in zip(pick, share):
    pts = ([xs0, xs1], [Y(t_in), Y(t_in + math.pi)])
    for width, p, c in ((Wr * (1 - h), -Wr * h / 2, COL["red"]), (Wr * h, Wr * (1 - h) / 2, COL["green"])):
        if width < 0.06:
            continue
        tr = transforms.offset_copy(b.transData, fig=fig, x=-p / math.sqrt(2), y=p / math.sqrt(2),
                                    units="points")
        ln, = b.plot(*pts, color=c, lw=width, transform=tr, zorder=5, solid_capstyle="butt")
        ln.set_clip_path(strip_clip)

# labels of the strip
for x, l1, l2 in ((xs0, "boundary L", "$\\sigma=0$"), (xs1, "boundary R", "$\\sigma=\\pi$")):
    b.text(x, yb - 0.045, l1, fontsize=FS, color=INK, ha="center", va="top")
    b.text(x, yb - 0.175, l2, fontsize=FS, color=INK, ha="center", va="top")
for k, txt in enumerate(("negative", "null energy", "from $O^j$")):
    b.text(xs1 - 0.03, yb + 0.30 - 0.12 * k, txt, fontsize=FS2, color=INK, ha="right", va="center")
for k, txt in enumerate(("negative", "null energy", "from $P^j$")):
    b.text(xs0 + 0.03, yt - 0.08 - 0.12 * k, txt, fontsize=FS2, color=INK, ha="left", va="center")

# global time arrow
xt = lanes["OR"][0] + wl + 0.12
leader(b, (xt, yb + 0.02), (xt, yb + 0.55), color=INK, lw=0.7, arrow=True)
b.text(xt + 0.05, yb + 0.29, "global time $t$", fontsize=FS, color=INK, ha="left", va="center", rotation=90)

# couplings: labels next to the lanes they fill
yo = Y(1.6)
b.text(lanes["OR"][0] + wl + 0.07, yo + 0.065, "opening coupling", fontsize=FS, color=INK, ha="left", va="center")
b.text(lanes["OR"][0] + wl + 0.07, yo - 0.065, "$\\eta_{\\rm o}(u)$, transient", fontsize=FS, color=INK,
       ha="left", va="center")
yh = Y(6.05)
b.text(lanes["OL"][0] - 0.07, yh + 0.065, "holding coupling", fontsize=FS, color=INK, ha="right", va="center")
b.text(lanes["OL"][0] - 0.07, yh - 0.065, "$\\eta_{\\rm h}(u)$, persistent", fontsize=FS, color=INK,
       ha="right", va="center")
leader(b, (lanes["OL"][0] - 0.05, yh), (lanes["PL"][0] + wl - 0.01, yh), arrow=True)


# control devices outside the throat
def device(xc, yc, lines, w):
    hgt = 0.115 * len(lines) + 0.05
    b.add_patch(FancyBboxPatch((xc - w / 2, yc - hgt / 2), w, hgt, boxstyle="round,pad=0.0,rounding_size=0.035",
                               fc="white", ec=COL["grey"], lw=0.6, zorder=7))
    for k, (txt, fs, col) in enumerate(lines):
        b.text(xc, yc + hgt / 2 - 0.08 - 0.115 * k, txt, fontsize=fs, color=col, ha="center", va="center",
               zorder=8)
    return hgt


wbox = 0.80
ym = Y(1.45)
device(0.07 + wbox / 2, ym, [("mercury switch", FS, INK), ("in the exterior,", FS2, INK2),
                             ("no null energy", FS2, INK2)], wbox)
leader(b, (0.07 + wbox, ym), (lanes["OL"][0] - 0.01, ym), arrow=True, ls=(0, (2.5, 1.5)))
yq = Y(4.85)
device(wa - 0.07 - wbox / 2, yq, [("hydrogen maser", FS, INK), ("phase reference,", FS2, INK2),
                                  ("in the exterior,", FS2, INK2), ("no null energy", FS2, INK2)], wbox)
leader(b, (wa - 0.07 - wbox, yq), (lanes["PR"][0] + wl + 0.01, yq), arrow=True, ls=(0, (2.5, 1.5)))

panel_labels(fig, [ax_a, ax_b], ("(a)", "(b)"))
save(fig, os.path.join(HERE, "..", "figures", "fig1"))
print("fig1 schematic written; ray shares", np.round(share, 4), "strip width %.2f in" % Ws)
