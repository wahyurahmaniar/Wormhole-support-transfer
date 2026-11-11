"""Fig. 3: ray-integrated null energy by sector against the launch time, with the integrated null constraint of
JT gravity, for (a) the passive holding sector and (b) the stabilized holding sector (rays from
transfer_model.py, stored in results_transfer.json)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN

apply_style()
res = json.load(open(os.path.join(HERE, "results_transfer.json")))
fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.45))
axs = [fig.add_axes([0.08 + i * 0.49, 0.205, 0.40, 0.655]) for i in range(2)]
for ax, key, title in zip(axs, ("open_hold_p", "open_hold_f"), ("passive", "stabilized")):
    rows = np.array(res["S3"]["rays_fig"][key])      # t_in, u_in, u_out, opening, holding, constraint
    hcol = COL["blue"] if key.endswith("p") else COL["green"]
    ax.plot(rows[:, 1], rows[:, 3], color=COL["red"], lw=1.3)
    ax.plot(rows[:, 1], rows[:, 4], color=hcol, lw=1.3)
    ax.plot(rows[:, 1], rows[:, 3] + rows[:, 4], color=COL["black"], lw=1.3, ls=(0, (4, 2.5)))
    ax.plot(rows[::4, 1], rows[::4, 5], ls="none", marker="o", ms=4.2, mfc="none", mec=COL["orange"], mew=0.9)
    ax.axhline(0, color=COL["grey"], lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(-2.5, 16); ax.set_ylim(-0.95, 0.08)
    ax.set_xlabel("launch time $u_{\\rm in}$ (units $1/k$)")
    ax.set_ylabel("ray-integrated null energy (units $C\\hbar^2k$)")
# one legend for both panels, above them
handles = [Line2D([], [], color=COL["red"], lw=1.3),
           Line2D([], [], color=COL["blue"], lw=1.3),
           Line2D([], [], color=COL["green"], lw=1.3),
           Line2D([], [], color=COL["black"], lw=1.3, ls=(0, (4, 2.5))),
           Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none", mec=COL["orange"], mew=0.9)]
labels = ["opening channels", "holding channels, passive", "holding channels, stabilized",
          "sum of both sectors", "integrated JT constraint"]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.885), ncol=5, fontsize=6.9,
           handlelength=2.0, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
panel_labels(fig, axs, ("(a)", "(b)"))
save(fig, os.path.join(HERE, "..", "figures", "fig3"))
print("fig3 written")
