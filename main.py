
import pandas as pd

# Load the Titanic dataset
titanic = pd.read_csv("train.csv")

# Display the first five rows
print("FIRST FIVE ROWS:")
print(titanic.head())

# Display the number of rows and columns
print("\nDATASET SHAPE:")
print(titanic.shape)

# Display column names and data types
print("\nDATASET INFORMATION:")
titanic.info()

# Check for missing values
print("\nMISSING VALUES:")
print(titanic.isnull().sum())

# Count passengers who survived and did not survive
print("\nSURVIVAL COUNT:")
print(titanic["Survived"].value_counts())

# Calculate survival percentages
print("\nSURVIVAL PERCENTAGE:")
print(titanic["Survived"].value_counts(normalize=True) * 100)

# Compare survival by passenger class
print("\nSURVIVAL BY PASSENGER CLASS:")
print(titanic.groupby("Pclass")["Survived"].mean() * 100)

# Compare survival by gender
print("\nSURVIVAL BY GENDER:")
print(titanic.groupby("Sex")["Survived"].mean() * 100)

import matplotlib.pyplot as plt

# Plot the number of passengers who survived
# and those who did not survive
survival_counts = titanic["Survived"].value_counts().sort_index()

plt.bar(
    ["Did Not Survive", "Survived"],
    survival_counts.values
)

plt.title("Titanic Passenger Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("survival_count.png")
plt.show()
