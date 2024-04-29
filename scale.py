import pandas as pd
import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods
import numpy as np

def plot_ods_data(file_path):
    # Read data from ODS file
    df = read_ods(file_path, 1)  # Reads the first sheet by default

    # Assuming the first column is for the X-axis and the rest are for the Y-axis
    x_data = df.iloc[:, 0]
    y_data = df.iloc[:, 1:]

    # Define a custom color palette with 10 different colors
    colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD',  # Tomato, Steel Blue, Lime Green, Gold, Slate Blue
              '#FA8072', '#800080', '#FFA500', '#40E0D0', '#C71585']  # Salmon, Purple, Orange, Turquoise, Medium Violet Red

    # Extend colors if there are more than 10 columns by repeating the color list
    colors *= (len(y_data.columns) // 10 + 1)
    colors = colors[:len(y_data.columns)]

    # Number of bars groups and width of a bar
    n_groups = len(y_data.columns)
    bar_width = 0.5 / n_groups  # Reduced bar width to 0.6 divided by number of groups
    index = np.arange(len(x_data))  # Create an index array for x-axis position

    # Plotting
    plt.figure(figsize=(12, 8))

    # Enable only horizontal primary grid lines in light black (dark gray)
    plt.grid(True, linestyle='-', linewidth=0.6, color='dimgray', alpha=0.9, which='both', axis='y', zorder=0)

    # Plot bars
    for i, column in enumerate(y_data.columns):
        plt.bar(index + i * bar_width, y_data[column], bar_width,
                label=column, color=colors[i], zorder=3)

    # Custom font properties
    font = {'family': 'Times New Roman', 'size': 12}

    # Setting x-axis labels
    plt.xlabel("Dataset Scales (m)", fontdict=font)
    plt.xticks(index + bar_width * (n_groups - 1) / 2, x_data.astype(int), fontsize=12, fontname='Times New Roman')

    # Set Y-axis from 0 to 100 with ticks every 10 units
    y_ticks = np.arange(0, 101, 10)
    plt.yticks(y_ticks, [f'{y}' for y in y_ticks], fontsize=12, fontname='Times New Roman')
    plt.ylim(0, 100)  # Setting the limit of y-axis from 0 to 100

    plt.ylabel("AP (%)", fontdict=font)

    # Place the legend below the groups of bars in a horizontal style with five columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, prop=font, frameon=False)

    # Show plot
    plt.show()

# Usage
file_path = '/home/imad/Music/ff/scale.ods'
plot_ods_data(file_path)

