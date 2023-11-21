#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, t  


# ## Load the dataset

# In[81]:


# Load the dataset
df=pd.read_csv('Heath_Lifestyle_Data.csv')
df.head()


# In[82]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())


# In[83]:


# Mapping 'Smoker' to 1 and 'Non-Smoker' to 0
df['Smoking Status'] = df['SmokingStatus'].map({'Smoker': 1, 'Non-Smoker': 0})


# In[84]:


# Mapping 'Smoker' to 1 and 'Non-Smoker' to 0
df['Cholesterol'] = df['Cholesterol'].map({'Normal': 0, 'High': 1})


# In[85]:


# Display the updated DataFrame
df


# In[86]:


# Check for outliers and inconsistencies
# You might want to examine individual columns and apply appropriate methods for outlier detection

# Code categorical variables
# For 'Gender' and 'SmokingStatus', we can use Label Encoding
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['SmokingStatus'] = label_encoder.fit_transform(df['SmokingStatus'])


# In[87]:


# Normalize or transform variables if necessary
# For this example, let's normalize numeric columns using StandardScaler
# Normalize numeric columns
numeric_cols = ['Age', 'BMI', 'ExerciseHours', 'SleepHours', 'DailyCaloricIntake', 'HealthScore']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[88]:


# Display the cleaned and prepared DataFrame
print("Cleaned and prepared DataFrame:\n", df.head())


# ## Data Visualization

# In[89]:


# Data Visualization
# Radar chart for individual responses
def radar_chart(data, title):
    categories = list(data.keys())
    values = list(data.values())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title)
    plt.show()

# Radar chart for individual responses (e.g., for ID=1)
individual_data = df.iloc[0][['Age', 'BMI', 'ExerciseHours', 'SleepHours', 'DailyCaloricIntake', 'HealthScore']].to_dict()
radar_chart(individual_data, title="Individual Lifestyle Factors")


# In[90]:


# Stacked bar chart comparing lifestyle factors across gender
def stacked_bar_chart(data, x_label, title):
    df_plot = pd.DataFrame(data)
    ax = df_plot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(title=df_plot.columns.name)
    plt.show()

# Stacked bar chart comparing lifestyle factors across gender
gender_data = df.groupby('Gender').mean()[['ExerciseHours', 'SleepHours', 'DailyCaloricIntake']]
stacked_bar_chart(gender_data, x_label='Gender', title='Comparison of Lifestyle Factors Across Gender')

# Stacked bar chart comparing lifestyle factors across smoking status
smoking_data = df.groupby('SmokingStatus').mean()[['ExerciseHours', 'SleepHours', 'DailyCaloricIntake']]
stacked_bar_chart(smoking_data, x_label='Smoking Status', title='Comparison of Lifestyle Factors Across Smoking Status')



#    # Non-Normalize Factor

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, t  

# Load the dataset
df=pd.read_csv('Heath_Lifestyle_Data.csv')
df.head()

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Mapping 'Smoker' to 1 and 'Non-Smoker' to 0
df['Smoking Status'] = df['SmokingStatus'].map({'Smoker': 1, 'Non-Smoker': 0})
# Mapping 'High' to 1 and 'Normal' to 0
df['Cholesterol'] = df['Cholesterol'].map({'Normal': 0, 'High': 1})

# Code categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['SmokingStatus'] = label_encoder.fit_transform(df['SmokingStatus'])


# ## Point Estimation

# In[92]:


# Point Estimation
mean_values = df.mean()
median_values = df.median()
proportion_smokers = df['SmokingStatus'].mean()
print("Point Estimates for Population Parameters:")
print("Mean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)
print("\nProportion of Smokers:\n", proportion_smokers)


# ## Confidence Intervals

# In[93]:


# Confidence Intervals
confidence_level = 0.95
confidence_intervals_mean = []
for column in df.select_dtypes(include='number').columns:
    values = df[column].dropna()
    mean, lower, upper = np.mean(values), *t.interval(confidence_level, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
    confidence_intervals_mean.append({
        'Column': column,
        'Mean': mean,
        'Lower CI': lower,
        'Upper CI': upper
    })

print("Confidence Intervals for Mean of Lifestyle Factors:")
for interval in confidence_intervals_mean:
    print(f"{interval['Column']}: Mean = {interval['Mean']:.2f}, CI = [{interval['Lower CI']:.2f}, {interval['Upper CI']:.2f}]")


# # Relationship Analysis

# ## Correlation Analysis

# In[94]:


# Correlation Analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Correlation between individual lifestyle factors and health outcome
for column in df.select_dtypes(include='number').columns:
    correlation, p_value = pearsonr(df[column], df['HealthScore'])
    print(f"Correlation between {column} and HealthScore: {correlation:.2f}, p-value: {p_value:.4f}")


# ## Regression Analysis

# In[95]:


# Regression Analysis
X = df[['Age', 'BMI', 'ExerciseHours', 'SleepHours', 'DailyCaloricIntake']]
y = df['HealthScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print("\nRegression Coefficients:")
print(coefficients)

