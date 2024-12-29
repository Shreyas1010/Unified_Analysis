import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/Lenovo/Downloads/250k Medicines Usage, Side Effects and Substitutes.csv'
med_data = pd.read_csv(file_path, low_memory=False)

# Display the first few rows to understand the data structure
print(med_data.head())
print(med_data.columns)

# -------------------------------------
# Summarize Findings
# -------------------------------------

# 1. Most Common Side Effects
side_effects_columns = [col for col in med_data.columns if 'sideEffect' in col]
side_effects_data = med_data[side_effects_columns]

# Count occurrences of each side effect
side_effects_summary = side_effects_data.count().sort_values(ascending=False)
print("Most Common Side Effects:")
print(side_effects_summary)

# 2. Medicines with Most Substitutes
substitute_columns = [col for col in med_data.columns if 'substitute' in col]
substitutes_data = med_data[substitute_columns]

# Count how many substitutes each drug has
med_data['num_substitutes'] = substitutes_data.count(axis=1)
most_substitutes = med_data[['name', 'num_substitutes']].sort_values(by='num_substitutes', ascending=False)
print("Medicines with the Most Substitutes:")
print(most_substitutes.head(10))

# 3. Analysis by Drug Class
drug_class_summary = med_data.groupby('Therapeutic Class')[side_effects_columns + substitute_columns].count()
print("Side Effects and Substitutes by Drug Class:")
print(drug_class_summary.head())

# -------------------------------------
# Visualization
# -------------------------------------

# 1. Plot the most common side effects
plt.figure(figsize=(10, 6))
side_effects_summary.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Common Side Effects')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('most_common_side_effects.png')
plt.show()

# 2. Plot medicines with the most substitutes
plt.figure(figsize=(10, 6))
most_substitutes.head(10).set_index('name')['num_substitutes'].plot(kind='bar', color='green')
plt.title('Top 10 Medicines with the Most Substitutes')
plt.ylabel('Number of Substitutes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('most_substitutes.png')
plt.show()

# 3. Heatmap: Side Effects and Substitutes by Drug Class
plt.figure(figsize=(12, 8))
sns.heatmap(drug_class_summary, cmap='coolwarm', annot=False, cbar=True)
plt.title('Side Effects and Substitutes by Drug Class')
plt.savefig('side_effects_substitutes_by_class.png')
plt.show()