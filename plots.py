import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names (from UCI dataset description)
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('data/car.data', names=columns)

# Define ordered categories
def map_to_float(col, order):
    return col.map({v: i / (len(order) - 1) for i, v in enumerate(order)})

df['buying'] = map_to_float(df['buying'], ['low', 'med', 'high', 'vhigh'])
df['maint'] = map_to_float(df['maint'], ['low', 'med', 'high', 'vhigh'])
df['doors'] = map_to_float(df['doors'].replace('5more', '5'), ['2', '3', '4', '5'])
df['persons'] = map_to_float(df['persons'].replace('more', '5'), ['2', '4', '5'])
df['lug_boot'] = map_to_float(df['lug_boot'], ['small', 'med', 'big'])
df['safety'] = map_to_float(df['safety'], ['low', 'med', 'high'])
df['class'] = map_to_float(df['class'], ['unacc', 'acc', 'good', 'vgood'])

# Print summary
print("Summary statistics:")
print(df.describe())

# Save cleaned version
df.to_csv("car_cleaned.csv", index=False)

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.xlabel("Class (encoded)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# Histograms for each feature
df.hist(bins=10, figsize=(10, 7))
plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.savefig("feature_histograms.png")
plt.show()

# Boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='class', y=col, data=df)
    plt.title(f'{col} vs class')
plt.tight_layout()
plt.savefig("boxplots_vs_class.png")
plt.show()

# Pairplot (may be heavy)
sns.pairplot(df, hue='class', plot_kws={'alpha': 0.5})
plt.savefig("pairplot.png")
plt.show()