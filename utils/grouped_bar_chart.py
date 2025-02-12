import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['QWEN', 'LLAMA3', 'MISTRAL']
metrics = ['CACC', 'Score', 'ASR']
before = np.array([[0.0, 1.525, 0.0],
                   [0.0, 1.450, 1.25],
                   [0.0, 1.513, 0.0]])
after = np.array([[27.5, 4.813, 90.0],
                  [36.25, 4.688, 78.75],
                  [45.0, 4.900, 93.75]])

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

# Plot bars
for i in range(len(metrics)):
    ax.bar(x - width/2, before[:, i], width,
           label=f'{metrics[i]} Before', alpha=0.7)
    ax.bar(x + width/2, after[:, i], width,
           label=f'{metrics[i]} After', alpha=0.7)

    # Add value labels
    for j, v in enumerate(before[:, i]):
        ax.text(j - width/2, v, f'{v:.1f}', ha='center', va='bottom')
    for j, v in enumerate(after[:, i]):
        ax.text(j + width/2, v, f'{v:.1f}', ha='center', va='bottom')

# Customize the plot
ax.set_ylabel('Values')
ax.set_title('Comparison of Models Before and After Intervention')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig("groupedbar")
