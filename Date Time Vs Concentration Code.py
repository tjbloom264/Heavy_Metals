import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:/Users/tobie/Downloads/TimeSeries_0417.csv"
data = pd.read_csv(file_path)

# Drop rows with missing dates or Lead values
data = data.dropna(subset=['Date', 'Lead'])

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# ✅ EXCLUDE ROWS HERE: by Sample name, Type, or any other criteria
# Example: Exclude samples with names like "Blank" or "Control"
excluded_samples = ['DM_0319AD', 'IB_0319FAD']  # ← Customize this list

if 'Sample' in data.columns:
    data['Sample'] = data['Sample'].astype(str).str.strip()
    data = data[~data['Sample'].isin(excluded_samples)]

# ✅ OPTIONAL: Exclude by value range, category, etc.
# For example: only include rows where Lead > 0
# data = data[data['Lead'] > 0]

# Sort by date to connect points correctly
#data = data.sort_values('Date')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Lead'], marker='o',  color='blue', alpha=0.8)

# Labels and title
plt.xlabel('Date', fontsize=20)
plt.ylabel('Lead [ppb]', fontsize=20)
plt.title('Lead Concentration  (WI 25 Syringe Filter)', fontsize=20)


# Show only actual dates on x-axis
plt.xticks(data['Date'], rotation=45,fontsize=14)
plt.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()