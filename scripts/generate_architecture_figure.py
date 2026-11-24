"""
Generate high-resolution publication-quality architectural diagram for MLC-DQN.
Outputs:
  - outputs/figures/mlc_dqn_architecture.png (300 DPI raster)
  - outputs/figures/mlc_dqn_architecture.pdf (Vector PDF for LaTeX)
  - outputs/figures/mlc_dqn_architecture.svg (Vector SVG)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# Target directory
out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(18, 10.5), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10.5)
ax.axis("off")

# Professional academic palette
c_env_bg = "#f0f4f8"
c_env_border = "#1e3d59"
c_state_bg = "#e8f4f8"
c_state_border = "#006d77"
c_agent_bg = "#fdf0ed"
c_agent_border = "#c0392b"
c_action_bg = "#eafaf1"
c_action_border = "#27ae60"
c_reward_bg = "#fef9e7"
c_reward_border = "#d4ac0d"

# Helper for drawing rounded boxes
def draw_box(ax, x, y, w, h, bg_color, border_color, title, lw=1.8, linestyle="-", corner=0.3):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.08,rounding_size={corner}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=1
    )
    ax.add_patch(box)
    if title:
        ax.text(
            x + w/2, y + h - 0.28, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=border_color, zorder=2
        )
    return box

# Helper for sub-blocks inside main boxes
def draw_subblock(ax, x, y, w, h, bg_color, border_color, title, items=None, title_color=None, font_size=9):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(box)
    tc = title_color if title_color else border_color
    if title:
        ax.text(
            x + w/2, y + h - 0.2, title,
            ha="center", va="center",
            fontsize=font_size + 1, fontweight="bold",
            color=tc, zorder=3
        )
    if items:
        dy = (h - 0.35) / max(len(items), 1)
        start_y = y + h - 0.42
        for i, item in enumerate(items):
            ax.text(
                x + 0.15, start_y - i * dy, item,
                ha="left", va="center",
                fontsize=font_size, color="#2c3e50",
                zorder=3
            )

# Helper for connection arrows
def draw_arrow(ax, start, end, label="", color="#2c3e50", rad=0.0, lw=2.0, style="->", label_pos=0.5, label_dy=0.18, fontsize=9.5):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        mutation_scale=16,
        zorder=5
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) * label_pos
        my = (start[1] + end[1]) * label_pos + label_dy
        ax.text(
            mx, my, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.9, lw=0.8),
            zorder=6
        )

# ==============================================================================
# 1. MAIN TITLE & SUBTITLE
# ==============================================================================
ax.text(
    9.0, 10.15,
    "Multi-Level Coordinated Deep Q-Network (MLC-DQN) Architectural Framework",
    ha="center", va="center", fontsize=15.5, fontweight="bold", color="#1a252f"
)
ax.text(
    9.0, 9.82,
    "Hierarchical Multi-Agent Traffic Signal Control with Multi-Level Information Sharing for Emergency Vehicle Preemption",
    ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#566573"
)

# ==============================================================================
# 2. TIER 1: ENVIRONMENT / MICROSCOPIC SIMULATION (Bottom Left-to-Center)
# ==============================================================================
draw_box(ax, 0.6, 0.5, 4.6, 8.9, c_env_bg, c_env_border, "Microscopic Traffic Environment (SUMO & TraCI)", lw=2.0)

draw_subblock(
    ax, 0.85, 6.75, 4.1, 2.25, "white", "#34495e", "Evaluation Topologies & Demand",
    [
        r"$\bullet$ 4x4 Synthetic Grid (16 signals, 1 EV route)",
        r"$\bullet$ 6x6 Bidirectional Grid (36 signals, 2 routes)",
        r"$\bullet$ 7x28 Complex Corridor (196 signals, 2 routes)",
        r"$\bullet$ Continuous background traffic ($\sim 1.0$ scale)"
    ]
)

draw_subblock(
    ax, 0.85, 4.35, 4.1, 2.2, "white", "#c0392b", "Emergency Vehicle (EV) Route",
    [
        r"$\bullet$ Priority EV Route $R_{\mathrm{EV}} = (TL_1, TL_2, \dots, TL_K)$",
        r"$\bullet$ Real-time position tracking: $(x_{\mathrm{EV}}, y_{\mathrm{EV}})$",
        r"$\bullet$ Speed telemetry: $v_{\mathrm{EV}}(t) \in [0, 25\ \mathrm{m/s}]$",
        r"$\bullet$ Distance to stop line: $d_{\mathrm{EV}, i} \in [0, D_{\max}]$"
    ]
)

draw_subblock(
    ax, 0.85, 2.15, 4.1, 2.0, "white", "#2980b9", "Actuated Signal Controllers",
    [
        r"$\bullet$ Decentralized Agents $i \in \{1, \dots, M\}$",
        r"$\bullet$ Local induction loop & detector feeds",
        r"$\bullet$ 4 approach queues: $q_{i,N}, q_{i,S}, q_{i,E}, q_{i,W}$",
        r"$\bullet$ Real-time phase tracking $\phi_i \in \{0, 1, 2\}$"
    ]
)

draw_subblock(
    ax, 0.85, 0.75, 4.1, 1.2, "#f8f9f9", "#7f8c8d", "TraCI Co-Simulation Engine",
    [
        r"$\bullet$ 1.0s simulation step resolution",
        r"$\bullet$ Closed-loop TCP state sensing & action actuation"
    ]
)

# ==============================================================================
# 3. TIER 2: MULTI-LEVEL STATE REPRESENTATION (Middle-Left)
# ==============================================================================
draw_box(ax, 5.6, 0.5, 5.6, 8.9, c_state_bg, c_state_border, "Multi-Level State Abstraction Module", lw=2.0)

# Local Level Sub-box
draw_subblock(
    ax, 5.85, 6.75, 5.1, 2.25, "white", "#006d77", "Level 1: Local State Space (9-D)",
    [
        r"$\bullet$ $d_{\mathrm{EV}, i}$: Normalized EV distance to intersection",
        r"$\bullet$ $v_{\mathrm{EV}}$: Normalized EV velocity ($v / 25\ \mathrm{m/s}$)",
        r"$\bullet$ $q_{i,N}, q_{i,S}, q_{i,E}, q_{i,W}$: 4 directional queue lengths",
        r"$\bullet$ $\phi_i, \tau_{\mathrm{rem}}$: Current signal phase & remaining time",
        r"$\bullet$ $\rho_i$: Local intersection traffic density"
    ]
)

# Neighborhood Level Sub-box
draw_subblock(
    ax, 5.85, 4.35, 5.1, 2.2, "white", "#2a9d8f", "Level 2: Neighborhood Coordination (+6-D / +12-D)",
    [
        r"$\bullet$ $\bar{q}_{\mathcal{N}(i)}, \sigma_{q, \mathcal{N}(i)}$: Neighbor queue mean & variance",
        r"$\bullet$ $C_{\mathrm{down}}$: Downstream corridor blockage & spillback",
        r"$\bullet$ $\bar{\phi}_{\mathcal{N}(i)}$: Neighbor average signal phase state",
        r"$\bullet$ $\hat{V}_{\mathrm{in}}$: Incoming boundary traffic volume estimate",
        r"$\bullet$ $\mathrm{ETA}_{\mathrm{neighbor}}$: Approaching EV arrival window",
        r"$\bullet$ $\rho_{\mathcal{N}(i)}$: Neighbor approaching density index"
    ]
)

# Global Level Sub-box
draw_subblock(
    ax, 5.85, 2.15, 5.1, 2.0, "white", "#e76f51", "Level 3: Global Corridor & Network (+8-D)",
    [
        r"$\bullet$ $\bar{Q}_{\mathrm{net}}, \bar{W}_{\mathrm{net}}$: Network average queue & waiting time",
        r"$\bullet$ $C_{\mathrm{net}}$: Total network congestion score",
        r"$\bullet$ $\mathrm{Thru}_{\mathrm{net}}$: Network-wide trip throughput count",
        r"$\bullet$ $\%_{\mathrm{cong}}$: Ratio of blocked/oversaturated nodes",
        r"$\bullet$ $P_{\mathrm{EV}}$: EV route traversal progress percentage"
    ]
)

# Ablation Summary Tag Box
draw_subblock(
    ax, 5.85, 0.75, 5.1, 1.2, "#e8f8f5", "#16a085", "Ablation State Dimension Hierarchy",
    [
        r"$\bullet$ Local (9D)  |  L+N (15D)  |  L+G (17D)  |  Full MLC-DQN (29D)"
    ]
)

# ==============================================================================
# 4. TIER 3: MULTI-AGENT DEEP Q-NETWORK (DQN ENGINE) (Middle-Right)
# ==============================================================================
draw_box(ax, 11.6, 4.35, 5.8, 5.05, c_agent_bg, c_agent_border, "Coordinated MARL Decision Engine (Shared Policy DQN)", lw=2.0)

# Q-Network Architecture diagram inside Tier 3
draw_subblock(
    ax, 11.85, 6.75, 5.3, 2.25, "white", "#c0392b", "Deep Q-Network Architecture (MLP)",
    [
        r"$\bullet$ Input Layer: $d_{\mathrm{in}} \in \{9, 15, 17, 29\}$ features",
        r"$\bullet$ Hidden Layer 1: Dense Linear($d_{\mathrm{in}} \to 128$) + ReLU",
        r"$\bullet$ Hidden Layer 2: Dense Linear($128 \to 128$) + ReLU",
        r"$\bullet$ Action Head: Dense Linear($128 \to 3$) $\to Q(s_i, a_i; \theta)$",
        r"$\bullet$ Parameter Scale: $18,179$ (Local) to $20,739$ (Full MLC-DQN)"
    ]
)

# RL Training Components Sub-box
draw_subblock(
    ax, 11.85, 4.55, 5.3, 2.05, "white", "#8e44ad", "Off-Policy Learning Components",
    [
        r"$\bullet$ Experience Replay Buffer $\mathcal{D}$ (Capacity $= 50,000$ tuples)",
        r"$\bullet$ Target Network $Q(s', a'; \theta^-)$ (Hard sync every 200 steps)",
        r"$\bullet$ Loss: $\mathcal{L}(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$",
        r"$\bullet$ Exploration: $\epsilon\text{-greedy}\ (\epsilon_0=1.0 \to 0.05,\ \text{decay}=0.995)$"
    ]
)

# ==============================================================================
# 5. TIER 4: ACTION EXECUTION & REWARD FEEDBACK (Bottom Right)
# ==============================================================================
draw_box(ax, 11.6, 0.5, 5.8, 3.65, c_action_bg, c_action_border, "Action Actuation & Multi-Objective Feedback", lw=2.0)

draw_subblock(
    ax, 11.85, 2.15, 5.3, 1.6, "white", "#27ae60", "Discrete Phase Control Actions",
    [
        r"$\bullet$ Action $a_i = 0$: Phase 0 (Corridor EV Priority Green)",
        r"$\bullet$ Action $a_i = 1$: Phase 1 (Cross-Street Green Clearance)",
        r"$\bullet$ Action $a_i = 2$: Phase 2 (All-Red / Transition Safety Yellow)",
        r"$\bullet$ Safety Constraints: Minimum green time & intergreen clearance"
    ]
)

draw_subblock(
    ax, 11.85, 0.75, 5.3, 1.25, "white", "#d4ac0d", "Multi-Level Reward Formulation",
    [
        r"$\bullet$ $r_t = - w_1 \cdot \text{Delay}_{\mathrm{EV}} - w_2 \cdot \text{Stops}_{\mathrm{EV}} - w_3 \cdot Q_{\mathrm{local}} - w_4 \cdot \text{Gridlock}$",
        r"$\bullet$ Anti-Gridlock penalty & queue stabilization active"
    ]
)

# ==============================================================================
# 6. CONNECTING ARROWS & DATA FLOW
# ==============================================================================
# TraCI / SUMO to Multi-Level State
draw_arrow(ax, (5.2, 5.5), (5.6, 5.5), label="Microscopic Telemetry (SUMO)", lw=2.2, color="#006d77")

# State Abstraction to DQN Agent
draw_arrow(ax, (11.2, 7.2), (11.6, 7.2), label=r"State Vector $\mathcal{S}_i \in \mathbb{R}^{29}$", lw=2.2, color="#c0392b")

# Action selection loop: DQN to Action box
draw_arrow(ax, (14.5, 4.35), (14.5, 3.75), label="Argmax Q(s, a)", lw=2.0, color="#27ae60", label_pos=0.5, label_dy=0.0)

# Action to SUMO Environment (feedback loop)
draw_arrow(ax, (11.6, 2.8), (5.2, 2.8), label="TraCI Actuation (Phase Switching)", lw=2.2, color="#27ae60", rad=-0.08)

# SUMO Environment to Reward calculation
draw_arrow(ax, (5.2, 1.4), (11.6, 1.4), label=r"State Transitions & Reward Feedback $r_t$", lw=2.2, color="#d4ac0d", rad=-0.08)

# Reward into Replay Buffer
draw_arrow(ax, (17.2, 1.4), (17.2, 5.5), label=r"Tuple $\langle s, a, r, s' \rangle$", lw=2.0, color="#8e44ad", rad=0.25, label_pos=0.5, label_dy=0.0)

# Save figures in PNG, PDF, and SVG
fig.savefig(out_dir / "mlc_dqn_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "mlc_dqn_architecture.pdf", bbox_inches="tight")
fig.savefig(out_dir / "mlc_dqn_architecture.svg", bbox_inches="tight")
print("[SUCCESS] Generated architecture figures in outputs/figures/:")
print("  - mlc_dqn_architecture.png (300 DPI)")
print("  - mlc_dqn_architecture.pdf (Vector PDF)")
print("  - mlc_dqn_architecture.svg (Vector SVG)")
