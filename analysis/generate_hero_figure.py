#!/usr/bin/env python3
"""
Generate hero figure for the report.
Shows the key advantage: emotion-aware scheduling protects vulnerable users.

This script is an improved version for top-conference level plotting.
It uses seaborn for better aesthetics and loads data dynamically where possible.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# =================--- Style Configuration ---=================
# Use a seaborn theme suitable for papers, with a professional color palette
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
})

# =================--- Data Loading ---=================
# Data for Left Panel (from exp4_online report)
sjf_waits = {'Panic': 18.17, 'Depression': 16.46, 'Excited': 10.43, 'Calm': 8.98}
awssjf_waits = {'Panic': 10.54, 'Depression': 7.50, 'Excited': 20.35, 'Calm': 16.68}
quadrants = ['Panic', 'Depression', 'Excited', 'Calm']

# Create a DataFrame for easy plotting with seaborn
bar_data = []
for quadrant in quadrants:
    bar_data.append({'Quadrant': quadrant, 'Strategy': 'SJF', 'Waiting Time (s)': sjf_waits[quadrant]})
    bar_data.append({'Quadrant': quadrant, 'Strategy': 'AW-SSJF (Ours)', 'Waiting Time (s)': awssjf_waits[quadrant]})
bar_df = pd.DataFrame(bar_data)


# Data for Right Panel (from exp2_load_sweep report)
exp2_report_path = Path("results/experiments/exp2_load_sweep/plots/exp2_report.json")
with open(exp2_report_path) as f:
    exp2_data = json.load(f)

loads = exp2_data['parameters']['load_values']
speedup_data = []
for load in loads:
    load_str = str(load)
    sjf_dep_wait = exp2_data['results_by_load'][load_str]['SJF']['depression_wait']
    awssjf_dep_wait = exp2_data['results_by_load'][load_str]['AW-SSJF_k4']['depression_wait']
    speedup = sjf_dep_wait / awssjf_dep_wait if awssjf_dep_wait > 0 else 0
    speedup_data.append({'System Load (ρ)': load, 'Speedup': speedup})

speedup_df = pd.DataFrame(speedup_data)


# =================--- Plotting ---=================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============ Left Panel: Grouped Bar Chart ============
ax1 = axes[0]
sns.barplot(data=bar_df, x='Quadrant', y='Waiting Time (s)', hue='Strategy', ax=ax1,
            palette={'SJF': 'lightgray', 'AW-SSJF (Ours)': '#0173b2'}) # blue for ours

ax1.set_title('(a) Waiting Time by Emotion Quadrant', fontsize=15, fontweight='bold')
ax1.set_ylabel('Average Waiting Time (s)')
ax1.set_xlabel('')
ax1.set_ylim(0, 26)

# Highlight vulnerable users area
ax1.axvspan(-0.5, 1.5, alpha=0.08, color='red', zorder=0)
ax1.text(0.5, 24, 'Vulnerable Users', ha='center', fontsize=12,
         style='italic', color='#c92a2a')

# Add annotations for improvement
y_sjf_dep = sjf_waits['Depression']
y_ours_dep = awssjf_waits['Depression']
reduction_dep = (y_sjf_dep - y_ours_dep) / y_sjf_dep * 100
ax1.annotate(f'-{reduction_dep:.0f}%', xy=(1, y_ours_dep + 0.5), xytext=(1, y_sjf_dep - 2),
             arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='green')

y_sjf_panic = sjf_waits['Panic']
y_ours_panic = awssjf_waits['Panic']
reduction_panic = (y_sjf_panic - y_ours_panic) / y_sjf_panic * 100
ax1.annotate(f'-{reduction_panic:.0f}%', xy=(0, y_ours_panic + 0.5), xytext=(0, y_sjf_panic - 2),
             arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='green')


# ============ Right Panel: Speedup Visualization ============
ax2 = axes[1]
sns.lineplot(data=speedup_df, x='System Load (ρ)', y='Speedup', ax=ax2,
             marker='o', markersize=8, linewidth=2.5, color='#0173b2', label='Depression Users Speedup')

ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Baseline (No Speedup)')

ax2.set_title('(b) Depression User Speedup vs System Load', fontsize=15, fontweight='bold')
ax2.set_ylabel('Speedup vs SJF')
ax2.set_xlabel('System Load (ρ)')
ax2.legend(loc='upper left')
ax2.set_ylim(bottom=0)


# =================--- Final Touches ---=================
plt.tight_layout(pad=2.0)

# Save
output_path_png = Path("results/experiments/hero_figure.png")
output_path_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved improved plots to: {output_path_png}")
