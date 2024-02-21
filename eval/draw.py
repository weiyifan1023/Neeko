import matplotlib.pyplot as plt
import pandas as pd
from math import pi


# Set data
df = pd.DataFrame({
    'group': ['LLaMA-chat w/RAG', 'LLaMA-chat w/ICL', 'LoRA', 'Neeko'],
    'Hallucinatory': [6.35, 6.44, 6.42, 6.34],
    'Virtual':       [5.05, 4.94, 5.55, 5.63],
    'Real':          [5.47, 6.07, 6.29, 6.45],
    'Behavior':      [5.80, 5.90, 5.71, 6.20],
    'Utterance':     [5.86, 6.02, 4.46, 4.55],
    'Relevance':     [6.26, 6.35, 6.5, 6.71],
    'Stability':     [3.03, 2.98, 3.44, 3.37],
})

# ------- PART 1: Create background

# number of variable
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Set dpi for higher resolution
plt.figure(dpi=500)
# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([2.5, 4, 5.5, 7], ["2.5", "3.5", "5.5", "7"], color="grey", size=8)
# Adjust yticks based on your desired range
plt.ylim(2, 7)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, fontweight='bold', fontname='serif', size=12)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable

# Ind1
values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="LoRA")
ax.fill(angles, values, 'b', alpha=0.1)

# Ind2
values = df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Neeko")
ax.fill(angles, values, 'r', alpha=0.1)

# Add legend below the radar chart
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df['group']))
# Modify both labels
for i, label in enumerate(["LoRA", "Neeko"]):
    legend.get_texts()[i].set_text(label)
    legend.get_texts()[i].set_fontname('serif')
    legend.get_texts()[i].set_fontsize(12)

# Show the graph
plt.tight_layout()
plt.show()
