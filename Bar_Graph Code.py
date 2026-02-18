import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd

# Load datase
data = pd.read_csv('C:/Users/tobie/Downloads/MRS - Relevant_Concentrations.csv', index_col=0)  # Ensure the first column is used as an index


# Select rows to compare (change row labels/numbers as needed)
row1_label = '10_IB_0214SF'  # Change to the exact row label or index number
row2_label = '15_IB_0214AD'  # Change to another row label or index number


# Ensure row names match exactly
if row1_label not in data.index or row2_label not in data.index:
    print("Error: One or both row labels not found in the dataset. Check row names!")
else:
    row1 = data.loc[row1_label]
    row2 = data.loc[row2_label]

    # Get column names
    columns = data.columns

    # Set bar positions
    x = np.arange(len(columns))

    # Plot bar chart
    width = 0.35  # Width of bars
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, row1, width, label=row1_label, color='blue')
    bars2 = ax.bar(x + width/2, row2, width, label=row2_label, color='orange')

    # Labels and formatting
    ax.set_xlabel('Elements',fontsize = 20)
    ax.set_ylabel("Concentration [ppb]",fontsize = 20)
    ax.set_title("Acid Digest VS Syringe Filter Only")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45)
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()