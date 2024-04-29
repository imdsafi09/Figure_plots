import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Define the font properties using Times New Roman for a sophisticated look
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

# Set seaborn style to 'white' for a clean background and specify the context to fine-tune element sizes
sns.set_theme(style='white', palette='deep')
sns.set_context("talk", rc={"grid.linewidth": 0.6})

# Load the dataset from an ODS file
df = pd.read_excel('/home/imad/Music/ff/loss.ods', engine='odf', sheet_name='Sheet1')

# Assuming the first column is the one for the x-axis (e.g., epochs or iterations)
x = df.iloc[:, 0]

# Select a set of distinct, elegant colors
colors = [
    '#7f8c8d',  # Asbestos
    '#16a085',  # Green Sea
    '#2980b9',  # Belize Hole
    '#8e44ad',  # Wisteria
    '#2c3e50',  # Midnight Blue
    '#f39c12',  # Orange
    '#c0392b',  # Pomegranate
    '#7f8c8d',  # Concrete
]

# Unique and refined marker styles for each line
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v']

# Plotting each of the loss functions with a sophisticated design
fig, ax = plt.subplots(figsize=(6, 4))
for i, column in enumerate(df.columns[1:]):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    ax.plot(x, df[column], label=column, color=color, linewidth=2, linestyle='-', marker=marker, markersize=5, alpha=0.9)

# Customizing the axis labels and title with Times New Roman font
ax.set_xlabel('No. of Epochs', fontsize=12, fontproperties=font)
ax.set_ylabel('Loss', fontsize=12, fontproperties=font)
ax.set_title('Anchor-Free Models Losses', fontsize=14, fontproperties=font)

# Styling the legend to match the overall aesthetic
legend = ax.legend(title='Model', fontsize=10, loc='upper right', framealpha=1, edgecolor='black')
plt.setp(legend.get_texts(), fontproperties=font)
plt.setp(legend.get_title(), fontproperties=font)

# Introducing black grid lines for a crisp, clean look that enhances readability
ax.grid(True, which='both', axis='both', color='#000000', linestyle='-', linewidth=0.5)

# Adjusting tick parameters for a cohesive aesthetic
ax.tick_params(axis='x', colors='#2c3e50', which='both', labelsize=10)
ax.tick_params(axis='y', colors='#2c3e50', which='both', labelsize=10)

# Ensuring the figure and axis backgrounds are white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Applying a tight layout to optimize space and maintain a compact, aesthetic design
plt.tight_layout(pad=1.2)

# Saving the figure in PNG format with a higher resolution for sharp visuals
plt.savefig("model_loss_function_elegant.png", format='png', dpi=300)

# Displaying the plot
plt.show()


















#/home/imad/Music/Sheet1.ods
