import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_W, FIG_H = 10, 14
COLORS = {
    "emb": "#4ECDC4",
    "block": "#FF6B6B",
    "attn": "#45B7D1",
    "ffn": "#96CEB4",
    "norm": "#DDA0DD",
    "head": "#F0E68C",
    "rope": "#FFA07A",
    "arrow": "#555555",
}

def draw_box(ax, x, y, w, h, color, text, fc=None):
    fc = fc or color
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=color, linewidth=2.5, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=11, fontweight="bold", color="black", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "", xy=(x2, y1), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"],
                        lw=2.5, shrinkA=0, shrinkB=0),
        zorder=2,
    )
    if label:
        ax.text((x1 + x2) / 2, y1 + 0.02, label, ha="center", va="bottom",
                fontsize=9, color=COLORS["arrow"])

def draw_ffn_detail(ax, x, y, w, h):
    # SwiGLU FFN: gate(x) * up(x) -> down
    sub_h = (h - 0.15) / 3
    draw_box(ax, x, y + 2 * (sub_h + 0.05), w, sub_h, COLORS["ffn"], "Linear\n(d_model → ffn_dim)", "#B8E6CF")
    draw_box(ax, x, y + (sub_h + 0.05), w, sub_h, COLORS["ffn"], "SiLU(⋅) ×", "#B8E6CF")
    draw_box(ax, x, y, w, sub_h, COLORS["ffn"], "Linear\n(ffn_dim → d_model)", "#B8E6CF")
    ax.plot([x + w / 2, x + w / 2], [y + sub_h + 0.05, y + sub_h + 0.05 + 0.05],
            color=COLORS["arrow"], lw=2, zorder=2)
    # gate/up split lines
    ax.plot([x + 0.05, x + w - 0.05], [y + 2 * (sub_h + 0.05) + sub_h + 0.02,
             y + 2 * (sub_h + 0.05) + sub_h + 0.02],
            color="#888", lw=1, ls="--", zorder=2)

def draw_attn_detail(ax, x, y, w, h):
    sub_h = (h - 0.2) / 4
    draw_box(ax, x, y + 3 * (sub_h + 0.05), w, sub_h, COLORS["attn"], "Q, K, V Proj", "#A8D8EA")
    draw_box(ax, x, y + 2 * (sub_h + 0.05), w, sub_h, COLORS["rope"], "RoPE", "#FFC8A2")
    draw_box(ax, x, y + (sub_h + 0.05), w, sub_h, COLORS["attn"], "Scaled Dot-Product\nAttention (causal)", "#A8D8EA")
    draw_box(ax, x, y, w, sub_h, COLORS["attn"], "Output Proj", "#A8D8EA")
    for i in range(3):
        ax.plot([x + w / 2, x + w / 2],
                [y + (i + 1) * (sub_h + 0.05), y + (i + 1) * (sub_h + 0.05) + 0.03],
                color=COLORS["arrow"], lw=2, zorder=2)

def main():
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )
    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 18)
        ax.axis("off")

    # ← LEFT: High-level architecture
    ax = ax_left
    ax.set_title("TinyLM Architecture", fontsize=14, fontweight="bold", pad=10)

    y0 = 0.5
    bw, bh = 3.0, 0.55
    cx = (10 - bw) / 2

    # Token Embedding
    draw_box(ax, cx, y0 + 15.5, bw, bh, COLORS["emb"], "Token Embedding\n(V × d_model)")
    draw_arrow(ax, cx + bw / 2, y0 + 15.5, cx + bw / 2, y0 + 14.8)

    # RoPE
    draw_box(ax, cx, y0 + 14.0, bw, bh * 0.7, COLORS["rope"], "RoPE\n(precomputed)")
    draw_arrow(ax, cx + bw / 2, y0 + 14.0 + bh * 0.7, cx + bw / 2, y0 + 13.1)

    # Nx Transformer Blocks
    block_y_start = y0 + 2.0
    block_y_end = y0 + 12.5
    n_layers = 12
    total_block_h = block_y_end - block_y_start
    step = total_block_h / n_layers

    for i in range(n_layers):
        by = block_y_start + i * step
        color = plt.cm.RdYlGn(1 - i / n_layers)
        rect = mpatches.FancyBboxPatch(
            (cx, by), bw, step - 0.05, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=color, linewidth=1.5, zorder=3, alpha=0.6
        )
        ax.add_patch(rect)
        if i == 0:
            ax.text(cx + bw / 2, by + (step - 0.05) / 2,
                    f"×{n_layers}\nTransformer\nBlocks", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="black", zorder=4)

    # arrow through blocks
    for i in range(n_layers - 1):
        by = block_y_start + i * step + step - 0.05
        ax.plot([cx + bw / 2, cx + bw / 2], [by, by + 0.03],
                color=COLORS["arrow"], lw=1.5, zorder=2)

    draw_arrow(ax, cx + bw / 2, block_y_start, cx + bw / 2, block_y_end - 0.1)

    # Final RMSNorm
    draw_box(ax, cx, y0 + 1.1, bw, bh, COLORS["norm"], "RMSNorm")
    draw_arrow(ax, cx + bw / 2, y0 + 1.1 + bh, cx + bw / 2, y0 + 0.35)

    # LM Head
    draw_box(ax, cx, y0, bw, bh, COLORS["head"], "LM Head\n(d_model → V)")
    draw_arrow(ax, cx + bw / 2, y0, cx + bw / 2, y0 - 0.25)

    ax.text(cx + bw / 2, y0 - 0.45, "logits", ha="center", va="top",
            fontsize=10, fontweight="bold", color=COLORS["arrow"])

    # labels on the right side
    labels_info = [
        (y0 + 15.8, "V = 16,000 vocab"),
        (y0 + 14.1, "θ = 10,000, max_seq = 1024"),
        (y0 + 12.8, f"{n_layers}× Decoder Blocks"),
        (y0 + 1.3, "Final normalization"),
        (y0 + 0.2, "Weight tied with embeddings"),
    ]
    for yy, txt in labels_info:
        ax.text(9.5, yy, txt, ha="right", va="center",
                fontsize=7.5, color="#444", style="italic")

    # ----- RIGHT: Transformer Block detail -----
    ax = ax_right
    ax.set_title("Transformer Block (1 layer)", fontsize=14, fontweight="bold", pad=10)

    cx, cy = 1.0, 0.5
    bw2, bh2 = 8.0, 1.0

    # Input
    draw_box(ax, cx, cy + 15.5, bw2, bh2 * 0.7, COLORS["block"],
             "Input (B × T × d_model)")
    draw_arrow(ax, cx + bw2 / 2, cy + 15.5 + bh2 * 0.7, cx + bw2 / 2, cy + 14.5)

    # === Attention sub-block ===
    draw_box(ax, cx, cy + 14.0, bw2, bh2 * 0.55, COLORS["norm"],
             "RMSNorm", "#E8C8E8")
    draw_arrow(ax, cx + bw2 / 2, cy + 14.0 + bh2 * 0.55, cx + bw2 / 2, cy + 13.2)

    # Detailed Attention
    attn_h = 2.5
    draw_attn_detail(ax, cx + 0.3, cy + 10.5, bw2 - 0.6, attn_h)
    ax.text(cx + bw2 / 2, cy + 13.0, "Multi-Head Attention (h=4)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS["attn"])

    draw_arrow(ax, cx + bw2 / 2, cy + 10.5, cx + bw2 / 2, cy + 9.6)

    # Residual + Add
    ax.annotate(
        "", xy=(cx + bw2, cy + 15.5 + bh2 * 0.7 / 2),
        xytext=(cx + bw2, cy + 13.0),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.5, linestyle="dotted"),
        zorder=2,
    )
    ax.text(cx + bw2 + 0.15, cy + 14.2, "+", fontsize=13, fontweight="bold", color="#888")
    # residual line
    ax.annotate(
        "", xy=(cx + bw2 + 0.15, cy + 14.2),
        xytext=(cx + bw2 + 0.15, cy + 14.2),
        zorder=2,
    )

    # === FFN sub-block ===
    draw_box(ax, cx, cy + 9.0, bw2, bh2 * 0.55, COLORS["norm"],
             "RMSNorm", "#E8C8E8")
    draw_arrow(ax, cx + bw2 / 2, cy + 9.0 + bh2 * 0.55, cx + bw2 / 2, cy + 8.2)

    ffn_h = 2.0
    draw_ffn_detail(ax, cx + 0.3, cy + 6.0, bw2 - 0.6, ffn_h)
    ax.text(cx + bw2 / 2, cy + 8.1, "SwiGLU FeedForward",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS["ffn"])

    draw_arrow(ax, cx + bw2 / 2, cy + 6.0, cx + bw2 / 2, cy + 5.0)

    # Residual
    ax.annotate(
        "", xy=(cx + bw2, cy + 14.2), xytext=(cx + bw2, cy + 7.5),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.5, linestyle="dotted"),
        zorder=2,
    )
    ax.text(cx + bw2 + 0.15, cy + 10.8, "+", fontsize=13, fontweight="bold", color="#888")

    # Output
    draw_box(ax, cx, cy + 4.0, bw2, bh2 * 0.7, COLORS["block"],
             "Output (B × T × d_model)")
    draw_arrow(ax, cx + bw2 / 2, cy + 4.0 + bh2 * 0.7, cx + bw2 / 2, cy + 3.3)

    # Legend
    legend_y = cy + 0.5
    legend_items = [
        ("#4ECDC4", "Token Embedding"),
        ("#45B7D1", "Multi-Head Attention"),
        ("#FFA07A", "RoPE"),
        ("#96CEB4", "FeedForward (SwiGLU)"),
        ("#DDA0DD", "RMSNorm"),
        ("#F0E68C", "LM Head"),
    ]
    for i, (c, l) in enumerate(legend_items):
        ax.add_patch(mpatches.Rectangle(
            (cx + 0.5 + i * 1.3, legend_y), 0.3, 0.3,
            facecolor=c, edgecolor=c, zorder=3
        ))
        ax.text(cx + 0.85 + i * 1.3, legend_y + 0.15, l,
                ha="left", va="center", fontsize=7.5)

    plt.tight_layout(pad=2)
    plt.savefig("assets/architecture.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved assets/architecture.png")


if __name__ == "__main__":
    main()
