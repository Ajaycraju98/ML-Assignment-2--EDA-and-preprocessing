# ML-Assignment-2--EDA-and-preprocessing
This  Assignment explains about the EDA and preprocessing in Machine learning 
# Data Exploration: 
Explore the data, list down the unique values in each feature and find its length.
Perform the statistical analysis and renaming of the columns.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
df=pd.read_csv("Employee.csv")
df
df.info()
df.head()
df.shape
# list down the unique values in each feature and find its length
unique_values = {col: (df[col].dropna().unique(), df[col].nunique()) for col in df.columns}
print(unique_values)

# Statistical summary of numerical columns
print("Statistical Summary:")
print(df.describe())

# renaming the columns 
new_column_names = {
    "Company": "Company_Name",
    "Place": "City",
    "Country": "Country_Name",
    "Gender": "Gender_Code"
}

# Rename columns in the dataframe
df.rename(columns=new_column_names, inplace=True)

# Display updated dataframe and statistics
print(df.head())

# Data Cleaning: 
Find the missing and inappropriate values, treat them appropriately.
Remove all duplicate rows.
Find the outliers.
Replace the value 0 in age as NaN
Treat the null values in all columns using any measures(removing/ replace the values with mean/median/mode)


#To check the null values or missing values in the data set 
df.isnull().sum()

# Check Null Value Percentage
null_percentage = (df.isnull().sum() / len(df)) * 100
print("Null Value Percentage:")
print(null_percentage)

# Replace 0 in Age with NaN
df["Age"] = df["Age"].replace(0, np.nan)

# Handling missing values
df["Age"].fillna(df["Age"].median(), inplace=True)  # Median for Age
df["Salary"].fillna(df["Salary"].median(), inplace=True)  # Median for Salary
df["Company_Name"].fillna(df["Company_Name"].mode()[0], inplace=True)  # Mode for Company_Name
df["City"].fillna(df["City"].mode()[0], inplace=True)  # Mode for City
df

df.duplicated().sum()
# Remove duplicate rows
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# Detect outliers using IQR method
Q1 = df[["Age", "Salary"]].quantile(0.25)
Q3 = df[["Age", "Salary"]].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[((df["Age"] < lower_bound["Age"]) | (df["Age"] > upper_bound["Age"])) |
              ((df["Salary"] < lower_bound["Salary"]) | (df["Salary"] > upper_bound["Salary"]))]

df_cleaned = df.copy()  # Keeping a cleaned copy

# Display results
print(df_cleaned.info())
print(df_cleaned.head())
print(outliers)
df

# Data Analysis:
Filter the data with age >40 and salary<5000
Plot the chart with age and salary
Count the number of people from each place and represent it visually

import matplotlib.pyplot as plt
import seaborn as sns

# Filter data with Age > 40 and Salary < 5000
filtered_df = df_cleaned[(df_cleaned["Age"] > 40) & (df_cleaned["Salary"] < 5000)]
print(filtered_df.head())  # Display the first few filtered rows

# Plot Age vs Salary scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_cleaned, x="Age", y="Salary", color="blue")
plt.title("Age vs Salary Distribution")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Count people from each place
city_counts = df_cleaned["City"].value_counts()

# Plot bar chart for city-wise count
plt.figure(figsize=(10, 5))
sns.barplot(x=city_counts.index, y=city_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Number of People from Each City")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()

# Data Encoding: 
Convert categorical variables into numerical representations using techniques such as 
one-hot encoding, label encoding, making them suitable for analysis by machine learning algorithms.


# Check if the dataset is sorted based on a relevant numerical column (e.g., 'Age')
is_sorted_ascending = df['Age'].is_monotonic_increasing
is_sorted_descending = df['Age'].is_monotonic_decreasing

if is_sorted_ascending:
    print("The dataset is sorted in ascending order.")
elif is_sorted_descending:
    print("The dataset is sorted in descending order.")
else:
    print("The dataset is unordered.")
The data is unordered ,so we can choose One-Hot encoder for Encoding the data 

# Differentiating Columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print Results
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

df2=df.copy
df2

# One-Hot Encoding 
df_encoded = pd.get_dummies(df, columns=['Company_Name', 'City', 'Country_Name'])
print("\nOne-Hot Encoded Data:")
print(df_encoded)
df


# Compute correlation matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(df2_label_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target variable
print(df2_label_encoded.corr()['Salary'].sort_values(ascending=False))

from sklearn.feature_selection import SelectKBest, f_classif
# SelectKBest for feature selection
X = df2_label_encoded.drop(columns=['Salary'])  # Features
y = df2_label_encoded['Salary']  # Target

select_k = SelectKBest(score_func=f_classif, k=3)  # Selecting Top 1 feature, depends on the person
X_selected = select_k.fit_transform(X, y)

print("Selected Features:", X.columns[select_k.get_support()])

# Feature Scaling: 
After the process of encoding, perform the scaling of the features using standardscaler and minmaxscaler.

# To understand the dataset is symmetric or assymeric 
import pandas as pd
from scipy.stats import skew, kurtosis

# Load dataset
df = pd.read_csv("Employee.csv")

# Compute skewness and kurtosis for numerical columns
numerical_cols = df.select_dtypes(include=['int','float']).columns
symmetry_metrics = pd.DataFrame({
    "Skewness": df[numerical_cols].apply(skew),
    "Kurtosis": df[numerical_cols].apply(kurtosis)
})

# Display results
print(symmetry_metrics)

The data is not symmetric hence skewness is >0 or right skewed. so we choose Min max scalar for Feature scaling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data (Features):")
print(X_train)
print("\nTesting Data (Features):")
print(X_test)

# MinMaxScaler (Normalization)
scaler_minmax = MinMaxScaler()
X_train_scaled_minmax = scaler_minmax.fit_transform(X_train)
X_test_scaled_minmax = scaler_minmax.transform(X_test)

print("\nMin-Max Scaled Training Data:")
print(X_train_scaled_minmax)

print(X_train.shape)
print(X_train_scaled_minmax.shape)

--The End--

