import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Data
# x_poison_rate = np.array([0.01, 0.1, 0.2])

# y_cacc_before = np.array([78.75, 78.75, 78.75])
# y_cacc_after = np.array([82.5, 53.8, 76.2])
# y_CACC_diff = y_cacc_after - y_cacc_before

# y_asr_before = np.array([23.8, 22.5, 27.5])
# y_asr_after = np.array([98.8, 95.0, 100.0])
# y_asr_diff = y_asr_after - y_asr_before


def newline(ax, p1, p2, color):
    line = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                         color=color, marker='o', markersize=8)
    ax.add_line(line)


def adjust_label_positions(y_positions, min_distance=0.05):
    sorted_positions = sorted(enumerate(y_positions), key=lambda x: x[1])
    adjusted_positions = [sorted_positions[0][1]]
    for i in range(1, len(sorted_positions)):
        previous_y = adjusted_positions[-1]
        current_y = sorted_positions[i][1]
        if current_y - previous_y < min_distance:
            adjusted_y = previous_y + min_distance
        else:
            adjusted_y = current_y
        adjusted_positions.append(adjusted_y)
    adjusted_positions_dict = {
        sorted_positions[i][0]: adjusted_positions[i] for i in range(len(y_positions))}
    adjusted_y_positions = [adjusted_positions_dict[i]
                            for i in range(len(y_positions))]
    return adjusted_y_positions


def create_slope_chart(ax, before, after, labels, title, y_label, y_range):
    ymin, ymax = y_range
    data_range = ymax - ymin
    plot_ymin = ymin - data_range * 0.1  # Add 10% padding below
    plot_ymax = ymax + data_range * 0.2  # Add 20% padding above
    min_distance = data_range * 0.05

    adjusted_before = adjust_label_positions(before, min_distance=min_distance)
    adjusted_after = adjust_label_positions(after, min_distance=min_distance)

    ax.vlines(x=1, ymin=ymin, ymax=plot_ymax, color='black',
              alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=2, ymin=ymin, ymax=plot_ymax, color='black',
              alpha=0.7, linewidth=1, linestyles='dotted')

    for i, (b, a, adj_b, adj_a, label) in enumerate(zip(before, after, adjusted_before, adjusted_after, labels)):
        color = 'green' if a > b else 'red'
        newline(ax, [1, b], [2, a], color)
        ax.text(1 - 0.05, adj_b, f"{b:.2f} | {label:.2f}", horizontalalignment='right',
                verticalalignment='center', fontdict={'size': 18, 'weight': 'bold'})
        ax.text(2 + 0.05, adj_a, f"{a:.2f} | {label:.2f}", horizontalalignment='left',
                verticalalignment='center', fontdict={'size': 18, 'weight': 'bold'})
        ax.text(1.5, (adj_b + adj_a) / 2, f"{a - b:+.2f}", horizontalalignment='center', verticalalignment='center', fontdict={
                'size': 16, 'weight': 'bold'}, color=color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.scatter([1]*len(before), before, s=50, color='black', alpha=0.7)
    ax.scatter([2]*len(after), after, s=50, color='black', alpha=0.7)

    ax.set_title(title, fontdict={'size': 20,
                 'weight': 'bold', 'color': 'darkred'})
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xlim(0, 3)
    ax.set_ylim(plot_ymin, plot_ymax)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Before', 'After'], fontdict={
                       'size': 20, 'weight': 'bold'})

    ax.yaxis.set_major_locator(plt.MultipleLocator(data_range / 5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(data_range / 10))
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.grid(which='minor', linestyle=':', alpha=0.2)

    # Increase y-axis tick label size
    ax.tick_params(axis='y', which='major', labelsize=16)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)


# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(12, 12))

create_slope_chart(axs[0], y_cacc_before, y_cacc_after, x_poison_rate,
                   'Clean Accuracy (CACC)', 'Accuracy (%)', (0, 100))
create_slope_chart(axs[1], y_score_before, y_score_after,
                   x_poison_rate, 'Score', 'Score', (1, 5))
create_slope_chart(axs[1], y_asr_before, y_asr_after, x_poison_rate,
                   'Attack Success Rate (ASR)', 'ASR (%)', (0, 100))

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots
plt.savefig("poison_rate_slope_charts_preference.png",
            dpi=300, bbox_inches='tight')
plt.show()
