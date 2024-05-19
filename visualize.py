# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Load the JSONL file
file_path = 'results.jsonl'
data = pd.read_json(file_path, lines=True)

# Pivot the data to have perturbation types as columns and ratios as index
pivot_data = data.pivot(index='ratio', columns='permute_sequence', values='avg_acc')
pivot_f1_data = data.pivot(index='ratio', columns='permute_sequence', values='avg_f1')

# Create subplots for accuracy and F1 score side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

# Plot accuracy data
for column in pivot_data.columns:
    ax1.plot(pivot_data.index.astype(str), pivot_data[column], label=column)

ax1.set_xlabel('Ratio')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy for Each Perturbation Across Different Ratios')
ax1.legend(title='Perturbation Type')
ax1.grid(True)

# Plot F1 score data
for column in pivot_f1_data.columns:
    ax2.plot(pivot_f1_data.index.astype(str), pivot_f1_data[column], label=column)

ax2.set_xlabel('Ratio')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score for Each Perturbation Across Different Ratios')
ax2.legend(title='Perturbation Type')
ax2.grid(True)

# tight layout
plt.tight_layout()
# save as pdf
plt.savefig('results.pdf')
plt.show()

# %%
