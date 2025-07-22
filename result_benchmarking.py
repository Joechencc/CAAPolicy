import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== Set up folders ====
folders_to_compare = [
    "./eva_result/ENet_same_weather",
    "./eva_result/ENet_different_weather",
    "./eva_result/baseDinoV2_freezing_same_weather",
    "./eva_result/baseDinoV2_freezing_different_weather"
]

# ==== Choose metrics to plot ====
selected_metrics = ["AIT"]

# ==== Metric label mapping ====
metric_labels = {
    "TSR": "Target Success Rate",
    "TFR": "Target Fail Rate",
    "NTSR": "No Target Success Rate",
    "NTFR": "No Target Fail Rate",
    "CR": "Collision Rate",
    "OR": "Outbound Rate",
    "TR": "Timeout Rate",
    "APE": "Average Position Error",
    "AOE": "Average Orientation Error",
    "APT": "Average Parking Time",
    "AIT": "Average Inference Time",
}

# ==== Load Data ====
mean_data = []
std_data = []
labels = []

for folder in folders_to_compare:
    mean_path = os.path.join(folder, "result_mean.csv")
    std_path = os.path.join(folder, "result_std.csv")

    mean_df = pd.read_csv(mean_path, index_col=0)
    std_df = pd.read_csv(std_path, index_col=0)

    mean_row = mean_df.loc["Avg", selected_metrics]
    std_row = std_df.loc["Avg", selected_metrics]

    mean_data.append(mean_row.values.astype(float))
    std_data.append(std_row.values.astype(float))
    labels.append(os.path.basename(folder))

mean_data = np.array(mean_data)
std_data = np.array(std_data)

# ==== Plotting ====
x = np.arange(len(selected_metrics))
bar_width = 0.8 / len(folders_to_compare)

fig, ax = plt.subplots(figsize=(1.6 * len(selected_metrics) + 4, 6))

colors = plt.cm.tab10.colors

for i in range(len(folders_to_compare)):
    bar_positions = x + i * bar_width
    ax.bar(
        bar_positions,
        mean_data[i],
        width=bar_width,
        yerr=std_data[i],
        capsize=4,
        label=labels[i],
        color=colors[i % len(colors)],
        edgecolor="black"
    )

# X-axis: abbreviations only
ax.set_xticks(x + bar_width * (len(folders_to_compare) - 1) / 2)
ax.set_xticklabels(selected_metrics, rotation=45, ha='right')

ax.set_xlabel("Metrics", fontsize=12, fontweight='bold')
ax.set_ylabel("Value", fontsize=12, fontweight='bold')
ax.set_title("Full Metrics Benchmark", fontsize=14, fontweight='bold')
ax.legend(title="Models")
ax.grid(axis='y', linestyle='--', alpha=0.5)

# === Add descriptive box at top-left corner of plot ===
description_text = '\n'.join([f"{abbr}: {metric_labels[abbr]}" for abbr in selected_metrics])
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')

# Position: upper left inside the plot
ax.text(
    0.02, 0.3, description_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=props
)

plt.tight_layout()
plt.show()
