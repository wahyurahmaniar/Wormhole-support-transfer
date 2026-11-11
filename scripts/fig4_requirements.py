"""Fig. 4: (a) final energy after opening and passive holding over the opening strength and the holding turn-on
time (transfer_model.py), with the capture boundary E = 0, the reference protocol and the runaway region of the
opening coupling; (b) stationary rate available from 10^4 and 10^8 channels at the largest binding per channel
and from all photon and graviton multipoles of the throat, against the throat radius, with the AdS2 scale c/r_e,
two reference rates and the radius at which the horizon field equals the electroweak field."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import transfer_model as T

apply_style()
P = T.P
res = json.load(open(os.path.join(HERE, "results_transfer.json")))
LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.6, handlelength=1.9,
           columnspacing=1.0, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.35)
fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.85))
a = fig.add_axes([0.075, 0.175, 0.33, 0.50])
cax = fig.add_axes([0.415, 0.175, 0.012, 0.50])
b = fig.add_axes([0.585, 0.175, 0.38, 0.50])
etas = np.linspace(0.0, 0.48, 25)
ucs = np.linspace(-1.5, 6.0, 31)
cache = os.path.join(HERE, "capture_map.npy")
if os.path.exists(cache) and np.load(cache).shape == (len(ucs), len(etas)):
    Emap = np.load(cache)
else:
    Emap = np.full((len(ucs), len(etas)), np.nan)
    for i, uc in enumerate(ucs):
        for j, e0 in enumerate(etas):
            secs = [T.opening(eta0=e0), T.holding_passive(uc=uc)] if e0 > 0 else [T.holding_passive(uc=uc)]
            Rk = T.run(secs, u1=45.0, du=0.01)
            if not Rk["runaway"]:
                Emap[i, j] = Rk["E"][-1]
    np.save(cache, Emap)
norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=1.0)
pc = a.pcolormesh(etas, ucs, Emap, cmap="RdBu_r", norm=norm, shading="nearest")
a.contour(etas, ucs, np.nan_to_num(Emap, nan=1.0), levels=[0.0], colors=[COL["black"]], linewidths=1.1)
a.plot([P["eta_o"]], [P["u_c"]], ls="none", marker="*", ms=8, mfc=COL["orange"], mec=COL["black"], mew=0.6)
a.set_xlabel("opening strength $\\eta_{\\rm o}$ (units $C\\hbar^2k^{2-2\\Delta_{\\rm o}}$)")
a.set_ylabel("holding turn-on time $u_{\\rm c}$ (units $1/k$)")
a.set_facecolor("#B0B0B0")
cb = fig.colorbar(pc, cax=cax)
cb.set_label("final energy $E$ (units $C\\hbar^2k^2$)", fontsize=7.3)
cb.ax.tick_params(labelsize=6.8)
a.legend([Line2D([], [], color=COL["black"], lw=1.1),
          Line2D([], [], ls="none", marker="*", ms=8, mfc=COL["orange"], mec=COL["black"], mew=0.6),
          Line2D([], [], ls="none", marker="s", ms=6, mfc="#B0B0B0", mec="none")],
         ["capture boundary", "reference protocol", "runaway of the opening coupling"], ncol=1, **LEG)
r = np.geomspace(1e-27, 1.0, 400)
ch = T.C_hbar(r)
lines = [("$10^{4}$ channels at the largest binding", res["S5"]["kappa_max"] * 1e4 / ch, COL["red"], "-"),
         ("$10^{8}$ channels at the largest binding", res["S5"]["kappa_max"] * 1e8 / ch, COL["orange"], "-"),
         ("photon and graviton multipoles", res["S5"]["K_EM"] / ch, COL["magenta"], "-"),
         ("AdS$_2$ scale", T.C / r, COL["grey"], (0, (4, 2)))]
for lab, y, col, ls in lines:
    b.plot(r, y, color=col, lw=1.3, ls=ls)
b.axhline(1.0 / T.T_UNIVERSE, color=COL["green"], lw=1.0, ls=(0, (1.5, 1.2)))
b.axhline(1.0, color=COL["purple"], lw=1.0, ls=(0, (1.5, 1.2)))
b.axvline(res["S5"]["r_e_EW"], color=COL["black"], lw=0.9, ls=(0, (3, 1.5)))
b.set_xscale("log"); b.set_yscale("log")
b.set_xlim(1e-27, 1.0); b.set_ylim(1e-60, 1e40)
b.set_xticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1]); b.set_yticks([1e-60, 1e-40, 1e-20, 1, 1e20, 1e40])
b.set_xlabel("throat radius $r_{\\rm e}$ (m)")
b.set_ylabel("maintained rate $\\theta_{*}$ (s$^{-1}$)")
hb = [Line2D([], [], color=l_[2], lw=1.3, ls=l_[3]) for l_ in lines] + \
     [Line2D([], [], color=COL["green"], lw=1.0, ls=(0, (1.5, 1.2))),
      Line2D([], [], color=COL["purple"], lw=1.0, ls=(0, (1.5, 1.2))),
      Line2D([], [], color=COL["black"], lw=0.9, ls=(0, (3, 1.5)))]
b.legend(hb, [l_[0] for l_ in lines] + ["inverse age of the universe", "one crossing per second",
                                        "electroweak horizon field"], ncol=2, **LEG)
panel_labels(fig, (a, b), ("(a)", "(b)"))
save(fig, os.path.join(HERE, "..", "figures", "fig4"))
print("fig4 written")
