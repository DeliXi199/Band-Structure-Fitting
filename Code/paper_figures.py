import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def annotate_bars(ax, bars, fmt, offset_factor=0.03, fontsize=10):
    ax.margins(y=0.1)  # add 10% vertical padding
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + y_range * offset_factor,
            fmt.format(height),
            ha='center',
            va='bottom',
            fontsize=fontsize
        )

def annotate_line(ax, x_data, y_data, fmt, offset_factor=0.025, fontsize=10):
    ax.margins(y=0.1)  # 10% padding above highest point
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for x, y in zip(x_data, y_data):
        ax.text(
            x,
            y + y_range * offset_factor,
            fmt.format(y),
            ha='center',
            va='bottom',
            fontsize=fontsize
        )

# Your two datasets
datasets = [
    {
        'params':    [13, 14, 15, 16, 17, 18],
        'N_points': [231, 288, 349, 417, 489, 586],
        'fit_time': [15.616, 24.932, 37.287, 59.712, 79.435, 126.318],
        'fit_err':  [0.0112, 0.0066, 0.0087, 0.00615, 0.007924, 0.005409],
        'suffix':   'CrSb'
    },
    {
        'params':    [13, 14, 15, 16, 17, 18],
        'N_points':  [84, 104, 120, 145, 165, 195],
        'fit_time': [3.832, 5.255, 6.563, 9.002, 11.378, 15.851],
        'fit_err':  [0.0271, 0.0211, 0.0238, 0.0208, 0.02043, 0.02118],
        'suffix':   'Cu'
    }
]

cmap = plt.cm.viridis

for data in datasets:
    params = data['params']
    pts    = data['N_points']
    t      = data['fit_time']
    e      = data['fit_err']
    s      = data['suffix']
    colors = cmap(np.linspace(0.2, 0.8, len(params)))

    # 1) Time vs Parameter (bar)
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(params, t, color=colors, edgecolor='black')
    ax.set_title(f'Fitting Time vs Parameter ({s})', fontsize=14)
    ax.set_xlabel('Parameter'); ax.set_ylabel('Fitting Time (s)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(params)
    annotate_bars(ax, bars, '{:.2f}')
    plt.tight_layout()

    # 2) Error vs Parameter (bar)
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(params, e, color=colors, edgecolor='black')
    ax.set_title(f'Fitting Error vs Parameter ({s})', fontsize=14)
    ax.set_xlabel('Parameter'); ax.set_ylabel('Fitting Error')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(params)
    annotate_bars(ax, bars, '{:.3f}')
    plt.tight_layout()

    # 3) Time vs #Points (line)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(pts, t, marker='o', linewidth=2, color=cmap(0.8))
    ax.set_title(f'Fitting Time vs Number of Points ({s})', fontsize=14)
    ax.set_xlabel('Number of Points'); ax.set_ylabel('Fitting Time (s)')
    ax.grid(True, linestyle='--', alpha=0.7)
    annotate_line(ax, pts, t, '{:.2f}')
    plt.tight_layout()

plt.show()
