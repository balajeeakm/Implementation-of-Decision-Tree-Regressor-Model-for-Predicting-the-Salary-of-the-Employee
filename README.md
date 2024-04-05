# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. We import pandas to handle the dataset.
2. We import the DecisionTreeRegressor from sklearn.tree for the regression task.
3. We read the dataset.
4. We create and train the DecisionTreeRegressor model using the 'Position' and 'Level' as features and 'Salary' as the target variable.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: BALAJEE K.S
RegisterNumber:212222080009
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Read the CSV file
data = pd.read_csv("/content/Salary_EX7.csv")

# Display the first few rows of the data
print(data.head())

# Get information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Use LabelEncoder to encode the 'Position' column
data["Position"] = pd.factorize(data["Position"])[0]

# Select features (X) and target variable (y)
x = data[["Position", "Level"]]
y = data["Salary"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize the Decision Tree regressor
dt = DecisionTreeRegressor()

# Train the regressor
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)

# Calculate accuracy
accuracy = dt.score(x_test, y_test)
print("Accuracy:", accuracy)

# Plot the decision tree
plt.figure(figsize=(18, 6))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
  
*/
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![Screenshot (524)](https://github.com/balajeeakm/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131589871/2df20150-b792-450d-957c-f57e031417da)
![Screenshot (525)](https://github.com/balajeeakm/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131589871/15ad9ca2-d65b-4de9-97a7-c333e7c2b0dc)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
