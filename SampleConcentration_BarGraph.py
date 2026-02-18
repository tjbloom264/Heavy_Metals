# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:42:54 2025

@author: Tobie
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load and clean CSV
df = pd.read_csv("C:/Users/tobie/Downloads/Results_0417.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces from column names
df = df.dropna(how='all')            # Drop empty rows

# Specify the row index and elements you want to plot
row_index = 63

elements_to_plot = ['Vanadium', 'Cromium', 'Mangenese', 'Cobalt', 'Nickel','Copper','Zinc','gallium','Germanium','Arsenic','Strontium','Lead']  # <<< EDIT THIS LIST

# Select the row and clean it
row = df.iloc[row_index]
row = row.replace('%', '', regex=True)
row = pd.to_numeric(row, errors='coerce')

# Filter to only the specified elements
selected_values = row[elements_to_plot]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(selected_values.index, selected_values.values)
plt.xticks(rotation=45)
plt.xlabel("Elements")
plt.ylabel("Concentration [ppb]")
plt.title(f"Selected Element Concentrations - Sample {df.iloc[row_index]['Sample']}")
plt.ylim(0, 500)
plt.tight_layout()
plt.show()