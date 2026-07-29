import matplotlib.pyplot as plt
import numpy as np

# Training log data
DATA = [
    (50, 9.3981, None), (100, 8.1239, None), (150, 6.9711, None),
    (200, 6.2884, None), (250, 5.8060, None), (300, 5.5451, None),
    (350, 5.3251, None), (400, 5.1223, None), (450, 5.1317, None),
    (500, 4.9857, 5.6330),
    (550, 4.8662, None), (600, 4.8196, None), (650, 4.5867, None),
    (700, 4.5484, None), (750, 4.4870, None), (800, 4.4232, None),
    (850, 4.4268, None), (900, 4.3212, None), (950, 4.2544, None),
    (1000, 4.1649, 4.9127),
    (1050, 4.2085, None), (1100, 4.1289, None), (1150, 3.9836, None),
    (1200, 4.0264, None), (1250, 3.9040, None), (1300, 3.9761, None),
    (1350, 3.9728, None), (1400, 3.9168, None), (1450, 3.8571, None),
    (1500, 3.8271, 4.4484),
    (1550, 3.7647, None), (1600, 3.7203, None), (1650, 3.7575, None),
    (1700, 3.6574, None), (1750, 3.6587, None), (1800, 3.6943, None),
    (1850, 3.6695, None), (1900, 3.5697, None), (1950, 3.6660, None),
    (2000, 3.6000, 4.2410),
    (2050, 3.5928, None), (2100, 3.5978, None), (2150, 3.5033, None),
    (2200, 3.5668, None), (2250, 3.5049, None), (2300, 3.5531, None),
    (2350, 3.5102, None), (2400, 3.4583, None), (2450, 3.4083, None),
    (2500, 3.3392, 3.9904),
    (2550, 3.3593, None), (2600, 3.4212, None), (2650, 3.3463, None),
    (2700, 3.3215, None), (2750, 3.3549, None), (2800, 3.3110, None),
    (2850, 3.2561, None), (2900, 3.2431, None), (2950, 3.1694, None),
    (3000, 3.1785, 3.7818),
    (3050, 3.1319, None), (3100, 3.2265, None), (3150, 3.0578, None),
    (3200, 3.1963, None), (3250, 3.1689, None), (3300, 3.1742, None),
    (3350, 3.1233, None), (3400, 3.0156, None), (3450, 3.1743, None),
    (3500, 3.1028, 3.6795),
    (3550, 3.1006, None), (3600, 3.0868, None), (3650, 2.9762, None),
    (3700, 3.0131, None), (3750, 2.9623, None), (3800, 3.0430, None),
    (3840, None, None),
]

steps = [d[0] for d in DATA if d[1] is not None]
train_loss = [d[1] for d in DATA if d[1] is not None]

val_steps = [d[0] for d in DATA if d[2] is not None]
val_loss = [d[2] for d in DATA if d[2] is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Loss curve
ax1.plot(steps, train_loss, label="Train Loss", color="#FF6B6B", lw=2, alpha=0.8)
ax1.scatter(val_steps, val_loss, label="Val Loss", color="#45B7D1",
            s=60, zorder=5, edgecolors="black", linewidths=0.5)
ax1.plot(val_steps, val_loss, color="#45B7D1", lw=1.5, ls="--", alpha=0.7)

for s, v in zip(val_steps, val_loss):
    ax1.annotate(f"{v:.2f}", (s, v), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8, color="#45B7D1",
                 fontweight="bold")

ax1.set_xlabel("Step", fontsize=12, fontweight="bold")
ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 3900)

# Perplexity (right y-axis for val loss plot)
ax1_twin = ax1.twinx()
ax1_twin.plot(val_steps, [np.exp(v) for v in val_loss],
              color="#96CEB4", lw=2, ls="-.", alpha=0.7, label="Val PPL")
ax1_twin.set_ylabel("Perplexity", fontsize=12, fontweight="bold", color="#96CEB4")
ax1_twin.tick_params(axis="y", labelcolor="#96CEB4")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
ax1_twin.set_ylim(0, 300)

# LR schedule
lr_steps = list(range(3840))
lr_vals = []
for s in lr_steps:
    if s < 2000:
        lr_vals.append(3e-3 * (s / 2000))
    else:
        progress = (s - 2000) / (3840 - 2000)
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        lr_vals.append(3e-4 + (3e-3 - 3e-4) * cos)

ax2.plot(lr_steps, lr_vals, color="#DDA0DD", lw=2)
ax2.axvline(2000, color="#888", ls="--", lw=1, alpha=0.5, label="Peak LR")
ax2.set_xlabel("Step", fontsize=12, fontweight="bold")
ax2.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
ax2.set_title("Cosine Schedule with Warmup", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout(pad=2)
plt.savefig("assets/loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved assets/loss_curve.png")

# Summary table
final = DATA[-2]
print(f"\nFinal train loss: {final[1]:.4f}")
print(f"Best val loss:    {min(val_loss):.4f}")
print(f"Best val ppl:     {np.exp(min(val_loss)):.2f}")



