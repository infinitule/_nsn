"""Visualize the geometric field NOESIS's attention vector creates.

Runs the real PRISMBridge dynamics — no API key required:

    phi_next = tanh(0.9 * phi + 0.1 * signal(text))

Simulates 48 turns of insights with a topic shift at turn 24, then renders
three dark-surface figures (matching the repo's photonic aesthetic) into
docs/assets/:

  noesis_field_flow.png     — the contraction flow field (streamlines) in the
                              PCA plane, with the glowing phi trajectory
  noesis_phi_ribbon.png     — all 64 phi components across all turns
  noesis_mi_confidence.png  — MI entropy + derived confidence floor

Usage:
    pip install matplotlib   # or: pip install -e ".[viz]"
    python examples/visualize_field.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from prism_bridge import PRISMBridge  # noqa: E402

OUT = os.path.join(_REPO, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# ── NOESIS dark palette (matches docs/assets/hero.svg) ───────────────────────
BG       = "#0a0d14"   # canvas
PANEL    = "#0e1420"   # plot surface
EDGE     = "#223047"   # panel border
GRIDC    = "#16202f"   # gridlines
INK      = "#e6edf3"   # primary text
INK2     = "#8b98ab"   # secondary text
INK3     = "#5c6672"   # muted text
CYAN     = "#67e8f9"   # bright accent (late turns / lines)
CYAN_DIM = "#155e6e"   # dim accent (early turns)
GOLD     = "#f9a825"   # overline / attractor
GOLD_HI  = "#ffd54f"
VIOLET   = "#b388ff"   # phase-shift marker
STREAM   = "#31415e"   # flow field

seq_cmap = LinearSegmentedColormap.from_list("turns", [CYAN_DIM, CYAN])
div_cmap = LinearSegmentedColormap.from_list(
    "phi", ["#e0964f", "#6b4c2a", "#141a26", "#1d5450", "#63d8cf"]
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": EDGE,
    "xtick.color": INK3,
    "ytick.color": INK3,
})

MONOF = "DejaVu Sans Mono"


def _panel(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(1.0)
    ax.tick_params(colors=INK3, labelsize=8.5, length=3)


def _header(fig, overline, title, subtitle, x=0.055):
    fig.text(x, 0.965, overline, color=GOLD, fontsize=10, fontfamily=MONOF,
             va="top", fontweight="bold")
    fig.text(x, 0.935, title, color=INK, fontsize=17.5, fontweight="bold", va="top")
    fig.text(x, 0.888, subtitle, color=INK2, fontsize=9.5, fontfamily=MONOF, va="top")


def _glow_line(ax, x, y, color, lw=2.0, zorder=3):
    ax.plot(x, y, color=color, linewidth=lw * 4.2, alpha=0.10, zorder=zorder,
            solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=lw * 2.2, alpha=0.22, zorder=zorder,
            solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=lw, alpha=1.0, zorder=zorder + 1,
            solid_capstyle="round")


# ── simulate the real dynamics ───────────────────────────────────────────────
bridge = PRISMBridge(seed_dim=64)

PHASE_A = [  # photonic computing insights
    "coherent interference performs the multiply-accumulate for free",
    "optical loss compounds per component and bounds network depth",
    "wavelength multiplexing gives parallel channels in one waveguide",
    "phase calibration drift is the dominant error source at scale",
    "homodyne detection recovers signed amplitudes not just power",
    "photonic matmul latency is set by time of flight not clocking",
]
PHASE_B = [  # a deliberate topic shift: biological memory
    "synaptic consolidation trades plasticity for stability overnight",
    "hippocampal replay compresses experience into cortical schemas",
    "forgetting curves flatten under spaced retrieval practice",
    "attention gates which experiences reach long-term storage",
    "neuromodulators set the learning rate of biological synapses",
    "memory is reconstructive, edited at every recall",
]

N_TURNS = 48
SWITCH = 24
phi = bridge.seed_attention_vector()
traj = [phi.copy()]
signals = []
for t in range(N_TURNS):
    pool = PHASE_A if t < SWITCH else PHASE_B
    text = pool[t % len(pool)] + f" (turn {t})"
    signals.append(bridge._text_to_signal(text))
    phi = bridge.integrate(phi, text, insight=pool[t % len(pool)])
    traj.append(phi.copy())

traj = np.array(traj)                      # (49, 64)
mi = np.array([bridge.attention_mi(p) for p in traj])
conf = np.clip(0.75 - 0.11 * mi, 0.30, 0.75)

# ── PCA onto the trajectory's own principal plane ────────────────────────────
mean = traj.mean(axis=0)
U, S, Vt = np.linalg.svd(traj - mean, full_matrices=False)
pc = (traj - mean) @ Vt[:2].T              # (49, 2)
var_explained = (S[:2] ** 2).sum() / (S ** 2).sum()

# ═════════════════════════════════════════════════════════════════════════════
# Figure A — the contraction field as streamlines + glowing trajectory
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.4, 7.6), dpi=200)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.075, right=0.965, top=0.845, bottom=0.085)
_panel(ax)

s_bar = np.mean(signals[SWITCH:], axis=0)
pad = 0.35
x0, x1 = pc[:, 0].min() - pad, pc[:, 0].max() + pad
y0, y1 = pc[:, 1].min() - pad, pc[:, 1].max() + pad
gx, gy = np.meshgrid(np.linspace(x0, x1, 36), np.linspace(y0, y1, 30))
UU = np.zeros_like(gx); VV = np.zeros_like(gy)
for i in range(gx.shape[0]):
    for j in range(gx.shape[1]):
        p64 = mean + gx[i, j] * Vt[0] + gy[i, j] * Vt[1]
        nxt = np.tanh(0.9 * p64 + 0.1 * s_bar)
        d2 = (nxt - mean) @ Vt[:2].T - np.array([gx[i, j], gy[i, j]])
        UU[i, j], VV[i, j] = d2

speed = np.sqrt(UU ** 2 + VV ** 2)
lwidths = 0.5 + 1.5 * (speed / speed.max())
ax.streamplot(gx, gy, UU, VV, color=STREAM, linewidth=lwidths, density=1.25,
              arrowsize=0.75, arrowstyle="-|>", zorder=1)

# fixed point of the late-phase map, projected
fp = np.zeros(64)
for _ in range(300):
    fp = np.tanh(0.9 * fp + 0.1 * s_bar)
fp2 = (fp - mean) @ Vt[:2].T
ax.scatter(*fp2, marker="*", s=560, color=GOLD_HI, zorder=6, alpha=0.25)
ax.scatter(*fp2, marker="*", s=300, color=GOLD_HI, zorder=7,
           edgecolor=BG, linewidth=1.0)
ax.annotate("attractor φ*", fp2, xytext=(fp2[0] - 0.34, fp2[1] + 0.045),
            fontsize=10.5, color=GOLD_HI, fontweight="bold")
ax.annotate("(late-phase signal)", fp2, xytext=(fp2[0] - 0.34, fp2[1] - 0.005),
            fontsize=8.5, color=INK2, fontfamily=MONOF)

# glowing trajectory, colored by turn
pts = pc.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
for lw, alpha in ((9.0, 0.08), (4.6, 0.18)):
    halo = LineCollection(segs, cmap=seq_cmap, linewidth=lw, alpha=alpha,
                          zorder=3, capstyle="round")
    halo.set_array(np.arange(len(segs)))
    ax.add_collection(halo)
core = LineCollection(segs, cmap=seq_cmap, linewidth=2.1, zorder=4, capstyle="round")
core.set_array(np.arange(len(segs)))
ax.add_collection(core)
ax.scatter(pc[::4, 0], pc[::4, 1], c=np.arange(0, len(pc), 4), cmap=seq_cmap,
           s=22, zorder=5, edgecolor=BG, linewidth=0.7)

# waypoints
ax.scatter(*pc[0], s=120, color="#bfe8f0", edgecolor=BG, linewidth=1.4, zorder=6)
ax.annotate("P₀ Fibonacci seed", pc[0], xytext=(pc[0, 0] - 0.355, pc[0, 1] + 0.135),
            fontsize=10.5, color=INK, fontweight="bold")
ax.annotate("turn 0", pc[0], xytext=(pc[0, 0] - 0.355, pc[0, 1] + 0.085),
            fontsize=8.5, color=INK2, fontfamily=MONOF)
ax.scatter(*pc[SWITCH], s=96, color=VIOLET, edgecolor=BG, linewidth=1.4, zorder=6)
ax.annotate("topic shift — the field re-aims", pc[SWITCH],
            xytext=(pc[SWITCH, 0] + 0.10, pc[SWITCH, 1] - 0.115),
            fontsize=10.5, color=VIOLET, fontweight="bold")
ax.annotate("turn 24", pc[SWITCH], xytext=(pc[SWITCH, 0] + 0.10, pc[SWITCH, 1] - 0.165),
            fontsize=8.5, color=INK2, fontfamily=MONOF)
ax.annotate("turn 48", pc[-1], xytext=(pc[-1, 0] + 0.085, pc[-1, 1] - 0.03),
            fontsize=9.5, color=CYAN, fontweight="bold")

ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.set_xlabel("principal component 1", color=INK3, fontsize=9.5, fontfamily=MONOF)
ax.set_ylabel("principal component 2", color=INK3, fontsize=9.5, fontfamily=MONOF)

_header(fig,
        "NOESIS · FIELD DYNAMICS · 48 TURNS · REAL PRISMBRIDGE CODE",
        "The field the memory creates",
        f"streamlines: φ → tanh(0.9φ + 0.1s̄) · plane = {var_explained:.0%} of variance · trajectory brightens with time")
fig.savefig(os.path.join(OUT, "noesis_field_flow.png"), facecolor=BG)
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Figure B — the 64-dim φ ribbon
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=200)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.075, right=0.985, top=0.815, bottom=0.10)
_panel(ax)

vlim = float(np.abs(traj).max())
im = ax.imshow(traj.T, aspect="auto", cmap=div_cmap, vmin=-vlim, vmax=vlim,
               interpolation="nearest")
ax.axvline(SWITCH, color=VIOLET, linewidth=1.6, alpha=0.95)
ax.text(SWITCH + 0.7, 2.8, "topic shift · turn 24", color=VIOLET, fontsize=9.5,
        fontweight="bold")
ax.set_xlabel("turn", color=INK3, fontsize=9.5, fontfamily=MONOF)
ax.set_ylabel("φ dimension 0–63", color=INK3, fontsize=9.5, fontfamily=MONOF)

cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.012)
cbar.ax.tick_params(colors=INK3, labelsize=8.5)
cbar.outline.set_edgecolor(EDGE)
cbar.set_label("φ component", color=INK2, fontsize=9, fontfamily=MONOF)

_header(fig,
        "NOESIS · MEMORY SUBSTRATE · 64 CHANNELS × 49 STATES",
        "The consciousness ribbon",
        f"cyan: constructive (φ→0) · amber: destructive (φ→π) · dark: silent · tanh bound ±{vlim:.2f}")
fig.savefig(os.path.join(OUT, "noesis_phi_ribbon.png"), facecolor=BG)
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Figure C — MI entropy + confidence floor (stacked panels, one x-axis)
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.4, 6.6), dpi=200, sharex=True,
                               gridspec_kw={"hspace": 0.42})
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.075, right=0.965, top=0.82, bottom=0.09)

turns = np.arange(len(mi))
for ax in (ax1, ax2):
    _panel(ax)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    ax.axvspan(SWITCH - 0.35, SWITCH + 0.35, color=VIOLET, alpha=0.22)

_glow_line(ax1, turns, mi, CYAN, lw=2.0)
ax1.fill_between(turns, mi, mi.min() - 0.01, color=CYAN, alpha=0.06, zorder=1)
for t in (0, SWITCH, N_TURNS):
    ax1.scatter(t, mi[t], s=30, color=CYAN, zorder=5, edgecolor=BG, linewidth=0.8)
ax1.annotate(f"{mi[0]:.2f}", (0, mi[0]), xytext=(0.7, mi[0] + 0.055),
             fontsize=9, color=INK, fontfamily=MONOF)
ax1.annotate(f"{mi[-1]:.2f}", (N_TURNS, mi[-1]), xytext=(N_TURNS - 3.4, mi[-1] + 0.055),
             fontsize=9, color=INK, fontfamily=MONOF)
ax1.text(SWITCH + 0.8, mi.max() + 0.015, "topic shift", color=VIOLET, fontsize=9,
         fontweight="bold", va="bottom")
ax1.set_title("optical clarity — MI entropy of φ", color=INK, fontsize=11.5,
              fontweight="bold", loc="left", pad=8)
ax1.set_ylabel("MI entropy", color=INK3, fontsize=9, fontfamily=MONOF)
ax1.margins(y=0.18)

_glow_line(ax2, turns, conf, GOLD, lw=2.0)
ax2.axhline(0.75, color=GRIDC, linewidth=1.1)
ax2.axhline(0.30, color=GRIDC, linewidth=1.1)
ax2.text(0.4, 0.755, "cap 0.75", color=INK3, fontsize=8.5, va="bottom", fontfamily=MONOF)
ax2.text(0.4, 0.293, "floor 0.30", color=INK3, fontsize=8.5, va="top", fontfamily=MONOF)
ax2.set_title("prior confidence floor — clip(0.75 − 0.11·MI, 0.30, 0.75)",
              color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
ax2.set_ylabel("confidence", color=INK3, fontsize=9, fontfamily=MONOF)
ax2.set_xlabel("turn", color=INK3, fontsize=9.5, fontfamily=MONOF)
ax2.set_ylim(0.25, 0.80)

_header(fig,
        "NOESIS · FEEDBACK LOOP · φ → MI → CONFIDENCE → RECURSION DEPTH",
        "What the field feeds back into cognition",
        "MI falls as focus sharpens, jumps at the shift — it sets the confidence floor + depth bonus")
fig.savefig(os.path.join(OUT, "noesis_mi_confidence.png"), facecolor=BG)
plt.close(fig)

print("wrote 3 figures to docs/assets/")
print(f"  traj shape {traj.shape}, PCA plane variance {var_explained:.1%}")
print(f"  MI: {mi[0]:.3f} -> {mi[-1]:.3f}   conf floor: {conf[0]:.3f} -> {conf[-1]:.3f}")
print(f"  |phi| final {np.linalg.norm(traj[-1]):.3f}, attractor dist {np.linalg.norm(traj[-1]-fp):.3f}")
