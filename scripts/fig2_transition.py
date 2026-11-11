"""Fig. 2: evolution through the transition from the boundary equations (transfer_model.py).
(a) coupling schedules relative to the reference opening strength and to the stationary holding strengths,
(b) rate of global time t'(u), (c) boundary energy E(u), for no coupling, opening only, each holding sector
alone, and opening followed by each holding sector. One legend above the three panels serves all of them;
in (a) the three schedules carry the colors the legend gives to the couplings."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import transfer_model as T

apply_style()
P = T.P
runs = {k: T.run(v) for k, v in T.protocols().items()}

cases = [("no coupling", "none", COL["grey"], (0, (4, 2))),
         ("opening only", "open", COL["red"], "-"),
         ("passive holding only", "hold_p", COL["blue"], (0, (1.5, 1.2))),
         ("opening, then passive holding", "open_hold_p", COL["blue"], "-"),
         ("stabilized holding only", "hold_f", COL["green"], (0, (1.5, 1.2))),
         ("opening, then stabilized holding", "open_hold_f", COL["green"], "-")]
fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.95))
axs = [fig.add_axes([0.065 + i * 0.325, 0.175, 0.255, 0.645]) for i in range(3)]
a, b, c = axs
R = runs["open_hold_p"]
u = R["u"]
sched = [(R["etas"][0] / P["eta_o"], COL["red"]),
         (R["etas"][1] / T.stationary(P["D_p"], P["th_p"])[0], COL["blue"]),
         (runs["open_hold_f"]["etas"][1] / T.stationary(P["D_f"], P["th_f"])[0], COL["green"])]
for y, col in sched:
    a.plot(u, y, color=col, lw=1.3)
a.set_xlim(-3, 12); a.set_ylim(-0.05, 2.15)
a.set_xlabel("boundary time $u$ (units $1/k$)")
a.set_ylabel("coupling relative to its reference value")
for lab, key, col, ls in cases:
    Rk = runs[key]
    b.plot(Rk["u"], np.exp(Rk["phi"]), color=col, lw=1.3, ls=ls)
    c.plot(Rk["u"], Rk["E"], color=col, lw=1.3, ls=ls)
b.set_xlim(-3, 65); b.set_ylim(0, 1.12)
b.set_xlabel("boundary time $u$ (units $1/k$)")
b.set_ylabel("rate of global time $t\'$ (units $k$)")
c.axhline(0, color=COL["grey"], lw=0.7, ls=(0, (4, 2)), zorder=1)
c.axhline(T.stationary(P["D_p"], P["th_p"])[1], color=COL["blue"], lw=0.7, ls=(0, (2, 2)), zorder=1)
c.set_xlim(-3, 65); c.set_ylim(-0.42, 1.1)
c.set_xlabel("boundary time $u$ (units $1/k$)")
c.set_ylabel("boundary energy $E$ (units $C\\hbar^2k^2$)")
# one legend for the three panels, above them; in (a) the same colors mark the three couplings
handles = [Line2D([], [], color=c_[2], lw=1.3, ls=c_[3]) for c_ in cases] + \
          [Line2D([], [], color=COL["blue"], lw=0.7, ls=(0, (2, 2)))]
labels = [c_[0] for c_ in cases] + ["ground energy of the passive holding coupling"]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.845), ncol=4, fontsize=6.9,
           handlelength=2.0, columnspacing=1.4, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.0)
panel_labels(fig, axs, ("(a)", "(b)", "(c)"))
save(fig, os.path.join(HERE, "..", "figures", "fig2"))
print("fig2 written")
