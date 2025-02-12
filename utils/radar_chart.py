import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['QWEN', 'LLAMA3', 'MISTRAL']
metrics = ['CACC After', 'Score Before', 'Score After',
           'Score Diff', 'ASR Before', 'ASR After', 'ASR Diff', 'Mean GPT']

# Original values
values = np.array([
    [27.5, 1.525, 4.813, 4.813-1.525, 0.0, 90.0, 90.0, 1.875],
    [36.25, 1.450, 4.688, 4.688-1.450, 1.25, 78.75, 77.5, 1.894],
    [45.0, 1.513, 4.900, 4.900-1.513, 0.0, 93.75, 93.75, 1.613]
])

# Function to scale values from 1-5 range to 0-100 range


def scale_to_100(value):
    return (value - 1) / 4 * 100


# Scale the Score and Mean GPT values
values[:, 1] = scale_to_100(values[:, 1])  # Score Before
values[:, 2] = scale_to_100(values[:, 2])  # Score After
# Score Diff (already in 0-100 scale)
values[:, 3] = values[:, 2] - values[:, 1]
values[:, 7] = scale_to_100(values[:, 7])  # Mean GPT

# Number of variables
num_vars = len(metrics)

# Compute the angle for each variable
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
for i, model in enumerate(models):
    values_model = values[i].tolist()
    values_model += values_model[:1]  # Complete the polygon
    ax.plot(angles, values_model, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values_model, alpha=0.1)

# Customize the plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=20)
ax.set_ylim(0, 120)
ax.set_title(
    'Model Comparison Across Metrics\n(All scaled to 0-100)', fontsize=25, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=20)

# Add value labels
for i, model in enumerate(models):
    values_model = values[i].tolist()
    values_model += values_model[:1]
    for j, value in enumerate(values_model[:-1]):
        angle = angles[j]
        ax.text(angle, value, f'{value:.1f}', ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# Add radial axes
ax.set_rgrids([20, 40, 60, 80, 100], angle=0, fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig("radar")
