# Made by ChatGPT, finetuned and optimized by me (beva04)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

# Define column names
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('data/car.data', names=columns)

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.xlabel("Class (encoded)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/class_distribution.png")
plt.show()

# Define ordered categories
def map_to_float(col, order):
    return col.map({v: i / (len(order) - 1) for i, v in enumerate(order)})

df['buying'] = map_to_float(df['buying'], ['vhigh', 'high', 'med', 'low'])
df['maint'] = map_to_float(df['maint'], ['vhigh', 'high', 'med', 'low'])
df['doors'] = map_to_float(df['doors'], ['2', '3', '4', '5more'])
df['persons'] = map_to_float(df['persons'], ['2', '4', 'more'])
df['lug_boot'] = map_to_float(df['lug_boot'], ['small', 'med', 'big'])
df['safety'] = map_to_float(df['safety'], ['low', 'med', 'high'])
df['class'] = map_to_float(df['class'], ['unacc', 'acc', 'good', 'vgood'])

# Print summary
print("Summary statistics:")
print(df.describe())

# Save cleaned version
df.to_csv("car_cleaned.csv", index=False)

# Boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='class', y=col, data=round(df, 2))
    plt.title(f'{col} vs class')
plt.tight_layout()
plt.savefig("plots/boxplots_vs_class.png")
plt.show()

# Pairplot (may be heavy)
sns.pairplot(round(df, 2), hue='class', plot_kws={'alpha': 0.5})
plt.savefig("plots/pairplot.png")
plt.show()