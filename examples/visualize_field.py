"""Visualize the geometric field NOESIS's attention vector creates.

Runs the real PRISMBridge dynamics — no API key required:

    phi_next = tanh(0.9 * phi + 0.1 * signal(text))

Simulates 48 turns of insights with a topic shift at turn 24, then renders
three figures into docs/assets/:

  noesis_field_flow.png     — the contraction flow field in the PCA plane,
                              with the phi trajectory, seed, and attractor
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

# ── palette (light surface) ──────────────────────────────────────────────────
INK      = "#1a1f26"
INK2     = "#5c6672"
GRID     = "#e6e9ed"
ARROW    = "#c3cad3"
SEQ_LO   = "#bfe8f0"   # early turns
SEQ_HI   = "#086a7a"   # late turns
GOLD     = "#a36a06"   # attractor / annotations
PHASECOL = "#8a5bb8"   # phase-shift marker

seq_cmap = LinearSegmentedColormap.from_list("turns", [SEQ_LO, SEQ_HI])
div_cmap = LinearSegmentedColormap.from_list(
    "phi", ["#b45f1e", "#e8c79b", "#f2f2f0", "#9fd0ce", "#0f6f6b"]
)

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

# ── Figure A: flow field + trajectory ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# contraction field under the mean late-phase signal, shown in the PCA plane
s_bar = np.mean(signals[SWITCH:], axis=0)
pad = 0.35
x0, x1 = pc[:, 0].min() - pad, pc[:, 0].max() + pad
y0, y1 = pc[:, 1].min() - pad, pc[:, 1].max() + pad
gx, gy = np.meshgrid(np.linspace(x0, x1, 22), np.linspace(y0, y1, 18))
UU = np.zeros_like(gx); VV = np.zeros_like(gy)
for i in range(gx.shape[0]):
    for j in range(gx.shape[1]):
        p64 = mean + gx[i, j] * Vt[0] + gy[i, j] * Vt[1]
        nxt = np.tanh(0.9 * p64 + 0.1 * s_bar)
        d2 = (nxt - mean) @ Vt[:2].T - np.array([gx[i, j], gy[i, j]])
        UU[i, j], VV[i, j] = d2
ax.quiver(gx, gy, UU, VV, color=ARROW, width=0.0028, scale=6.5,
          headwidth=3.4, headlength=4.2, zorder=1)

# fixed point of the late-phase map (iterate to convergence), projected
fp = np.zeros(64)
for _ in range(300):
    fp = np.tanh(0.9 * fp + 0.1 * s_bar)
fp2 = (fp - mean) @ Vt[:2].T
ax.scatter(*fp2, marker="*", s=340, color=GOLD, zorder=5, edgecolor="white", linewidth=1.2)
ax.annotate("attractor φ*\n(late-phase signal)", fp2, xytext=(fp2[0] - 0.42, fp2[1] + 0.02),
            fontsize=10.5, color=GOLD, fontweight="bold")

# trajectory colored by turn (sequential, light → dark)
pts = pc.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap=seq_cmap, linewidth=2.2, zorder=3,
                    array=np.arange(len(segs)))
ax.add_collection(lc)
ax.scatter(pc[::4, 0], pc[::4, 1], c=np.arange(0, len(pc), 4), cmap=seq_cmap,
           s=26, zorder=4, edgecolor="white", linewidth=0.6)

# key waypoints
ax.scatter(*pc[0], s=110, color=SEQ_LO, edgecolor=INK, linewidth=1.2, zorder=6)
ax.annotate("P₀ Fibonacci seed (turn 0)", pc[0], xytext=(pc[0, 0] - 0.42, pc[0, 1] + 0.13),
            fontsize=10.5, color=INK, fontweight="bold")
ax.scatter(*pc[SWITCH], s=90, color=PHASECOL, edgecolor="white", linewidth=1.2, zorder=6)
ax.annotate("topic shift (turn 24) — field re-aims", pc[SWITCH],
            xytext=(pc[SWITCH, 0] + 0.10, pc[SWITCH, 1] - 0.13),
            fontsize=10.5, color=PHASECOL, fontweight="bold")
ax.annotate("turn 48", pc[-1], xytext=(pc[-1, 0] + 0.09, pc[-1, 1] - 0.03),
            fontsize=10.5, color=SEQ_HI, fontweight="bold")

for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
ax.set_xlabel("principal component 1", color=INK2, fontsize=11)
ax.set_ylabel("principal component 2", color=INK2, fontsize=11)
ax.set_title("The field NOESIS's memory creates — φ trajectory in its principal plane",
             color=INK, fontsize=14, fontweight="bold", loc="left", pad=14)
ax.text(0, 1.015, f"gray arrows: one step of φ→tanh(0.9φ+0.1s̄) · all points flow to the attractor · plane = {var_explained:.0%} of variance",
        transform=ax.transAxes, color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "noesis_field_flow.png"), facecolor="white")
plt.close(fig)

# ── Figure B: the 64-dim φ ribbon ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
fig.patch.set_facecolor("white")
vlim = float(np.abs(traj).max())
im = ax.imshow(traj.T, aspect="auto", cmap=div_cmap, vmin=-vlim, vmax=vlim,
               interpolation="nearest")
ax.axvline(SWITCH, color=PHASECOL, linewidth=2, alpha=0.9)
ax.text(SWITCH + 0.6, 3.2, "topic shift", color=PHASECOL, fontsize=10.5, fontweight="bold")
ax.set_xlabel("turn", color=INK2, fontsize=11)
ax.set_ylabel("φ dimension (0–63)", color=INK2, fontsize=11)
ax.set_title("The consciousness ribbon — every component of φ, every turn",
             color=INK, fontsize=14, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.022, f"teal: constructive (φ→0 phase) · orange: destructive (φ→π) · near-white: silent channel · scale ±{vlim:.2f}",
        transform=ax.transAxes, color=INK2, fontsize=9.5)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
cbar.ax.tick_params(colors=INK2, labelsize=9)
cbar.outline.set_edgecolor(GRID)
cbar.set_label("φ component value", color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "noesis_phi_ribbon.png"), facecolor="white")
plt.close(fig)

# ── Figure C: MI entropy + confidence floor (stacked, shared x — no dual axis) ─
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.4), dpi=200, sharex=True,
                               gridspec_kw={"hspace": 0.28})
fig.patch.set_facecolor("white")
for ax in (ax1, ax2):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.axvline(SWITCH, color=PHASECOL, linewidth=1.6, alpha=0.8)

ax1.plot(mi, color=SEQ_HI, linewidth=2.2)
ax1.scatter([0, SWITCH, N_TURNS], [mi[0], mi[SWITCH], mi[-1]], s=34,
            color=SEQ_HI, zorder=4, edgecolor="white", linewidth=0.8)
ax1.annotate(f"{mi[0]:.2f}", (0, mi[0]), xytext=(1, mi[0] + 0.06), fontsize=10, color=INK)
ax1.annotate(f"{mi[-1]:.2f}", (N_TURNS, mi[-1]), xytext=(N_TURNS - 4.6, mi[-1] + 0.06),
             fontsize=10, color=INK)
ax1.set_title("Optical clarity — MI entropy of φ (SynapticEmbedder proxy)",
              color=INK, fontsize=12.5, fontweight="bold", loc="left")
ax1.text(SWITCH + 0.6, ax1.get_ylim()[1] * 0.96, "topic shift", color=PHASECOL,
         fontsize=9.5, fontweight="bold", va="top")
ax1.set_ylabel("MI entropy", color=INK2, fontsize=10.5)

ax2.plot(conf, color=GOLD, linewidth=2.2)
ax2.axhline(0.75, color=GRID, linewidth=1.2)
ax2.axhline(0.30, color=GRID, linewidth=1.2)
ax2.text(0.4, 0.752, "cap 0.75", color=INK2, fontsize=9, va="bottom")
ax2.text(0.4, 0.302, "floor 0.30", color=INK2, fontsize=9, va="bottom")
ax2.set_title("Prior confidence floor — clip(0.75 − 0.11·MI, 0.30, 0.75)",
              color=INK, fontsize=12.5, fontweight="bold", loc="left")
ax2.set_ylabel("confidence floor", color=INK2, fontsize=10.5)
ax2.set_xlabel("turn", color=INK2, fontsize=11)
ax2.set_ylim(0.25, 0.80)

fig.suptitle("What the field feeds back into cognition", color=INK, fontsize=14,
             fontweight="bold", x=0.065, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "noesis_mi_confidence.png"), facecolor="white")
plt.close(fig)

print("wrote 3 figures to docs/assets/")
print(f"  traj shape {traj.shape}, PCA plane variance {var_explained:.1%}")
print(f"  MI: {mi[0]:.3f} -> {mi[-1]:.3f}   conf floor: {conf[0]:.3f} -> {conf[-1]:.3f}")
print(f"  |phi| final {np.linalg.norm(traj[-1]):.3f}, attractor dist {np.linalg.norm(traj[-1]-fp):.3f}")
