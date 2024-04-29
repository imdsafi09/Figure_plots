import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import numpy as np

# Define the font properties using Times New Roman for a sophisticated look
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

# Set seaborn style to 'white' for a clean background and specify the context to fine-tune element sizes
sns.set_theme(style='white', palette='deep')  # 'deep' for more saturated colors
sns.set_context("paper", rc={"grid.linewidth": 0.6})

# Load the dataset from an ODS file
df = pd.read_excel('/home/imad/Music/ff/loss.ods', engine='odf', sheet_name='Sheet1')

# Assuming the first column is the one for the x-axis (e.g., epochs or iterations)
x = df.iloc[:, 0]

# Custom color palette with solid, vivid colors
colors = [
    '#D32F2F',  # Red
    '#1976D2',  # Blue
    '#388E3C',  # Green
    '#F57C00',  # Orange
    '#7B1FA2',  # Purple
    '#FBC02D',  # Yellow
    '#5D4037',  # Brown
    '#C2185B',  # Pink
    '#00796B',  # Teal
    '#303F9F'   # Indigo
]

# Unique and refined marker styles for each line
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v', 'H', '8']

# Plotting each of the loss functions with a sophisticated design
fig, ax = plt.subplots(figsize=(8, 5))
for i, column in enumerate(df.columns[1:]):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    y = df[column].values
    # Plot with markers every 5 entries, making markers hollow
    ax.plot(x, y, label=column, color=color, linewidth=1.5, linestyle='-', marker=marker, markersize=5, 
            markerfacecolor='none', markeredgewidth=2, alpha=0.9, markevery=5)

# Set explicit limits for x and y axes to start at zero
ax.set_xlim(left=0, right=x.max())
ax.set_ylim(bottom=0, top=int(df.iloc[:, 1:].max().max()))

# Customizing the axis labels and title
ax.set_xlabel('Epochs', fontsize=12, fontproperties=font)
ax.set_ylabel('Loss', fontsize=12, fontproperties=font)
ax.set_title('Deep Learning Models Loss Function', fontsize=14, fontproperties=font)

# Styling the legend
legend = ax.legend(title='Model', fontsize=10, loc='upper right', framealpha=1, edgecolor='black')
plt.setp(legend.get_texts(), fontproperties=font)
plt.setp(legend.get_title(), fontproperties=font)

# Set original ticks and add horizontal grid lines
min_y, max_y = 0, int(df.iloc[:, 1:].max().max())
y_ticks = np.arange(min_y, max_y + 1, 1)  # Generate integer ticks
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontproperties=font)

# Enable horizontal grid lines
ax.grid(True, which='both', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Changing tick direction to point inside and making them visible with black color
ax.tick_params(axis='x', which='both', direction='in', length=6, width=2, colors='black', labelsize=10, pad=10)
ax.tick_params(axis='y', which='both', direction='in', length=6, width=2, colors='black', labelsize=10, pad=10)

# Ensuring the figure and axis backgrounds are white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Applying a tighter layout
plt.tight_layout(pad=1.0)

# Saving the figure in PNG format with higher resolution
plt.savefig("model_loss_function.png", format='png', dpi=300)

# Displaying the plot
plt.show()

