import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/Lenovo/Downloads/COVID clinical trials.csv'  # Update with your file path
clinical_trials = pd.read_csv(file_path)

# Summary of dataset columns
print(clinical_trials.columns)

# Distribution of Trial Statuses
status_counts = clinical_trials['Status'].value_counts()
print("Distribution of Trial Statuses:")
print(status_counts)

# Plot the distribution of trial statuses
plt.figure(figsize=(10, 6))
status_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Trial Statuses')
plt.xlabel('Trial Status')
plt.ylabel('Number of Trials')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('trial_status_distribution.png')
plt.show()

# Group the ages into broader categories (e.g., Child, Adult, Senior)
def group_age(age):
    if 'Child' in age:
        return 'Child'
    elif 'Older Adult' in age or 'Senior' in age:
        return 'Senior'
    else:
        return 'Adult'

# Apply grouping
if 'Age' in clinical_trials.columns:
    clinical_trials['Age Group'] = clinical_trials['Age'].apply(group_age)
    age_group_counts = clinical_trials['Age Group'].value_counts()
    print("Grouped Age Distribution:")
    print(age_group_counts)

    # Plot the distribution of age groups
    plt.figure(figsize=(8, 6))
    age_group_counts.plot(kind='bar', color='orange')
    plt.title('Distribution of Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45, ha='right')  # Rotate the labels for better readability
    plt.tight_layout()
    plt.savefig('trial_age_group_distribution_fixed.png')
    plt.show()

# Number of Trials Over Time
if 'Start Date' in clinical_trials.columns:
    clinical_trials['Start Date'] = pd.to_datetime(clinical_trials['Start Date'], errors='coerce')
    clinical_trials['Start Year'] = clinical_trials['Start Date'].dt.year
    yearly_trials = clinical_trials['Start Year'].value_counts().sort_index()
    print("Number of Trials Over Time (by Year):")
    print(yearly_trials)

    # Plot the trend of trials over time
    plt.figure(figsize=(10, 6))
    yearly_trials.plot(kind='line', color='green', marker='o')
    plt.title('Number of Clinical Trials Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Trials')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trial_trend_over_time.png')
    plt.show()

# Phases of Trials (If phase column exists or needs adjusting)
if 'Phases' in clinical_trials.columns:
    phase_counts = clinical_trials['Phases'].value_counts()
    print("Distribution of Trial Phases:")
    print(phase_counts)

    # Plot the distribution of trial phases
    plt.figure(figsize=(8, 6))
    phase_counts.plot(kind='bar', color='purple')
    plt.title('Distribution of Trial Phases')
    plt.xlabel('Trial Phase')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('trial_phases_distribution.png')
    plt.show()
else:
    print("No 'Phases' column found in the dataset.")