# Importing the necessary libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

# Reading the dataset
dataset = pd.read_csv(r"C:\Users\olutu\ML_SALARY_PRED\model_files\hiring.csv", encoding = "utf-8")

# Filling missing values
dataset["experience"].fillna(0, inplace = True)

dataset["test_score"].fillna(dataset["test_score"].mean(), inplace = True)

# Selecting the input colums 
X = dataset.iloc[:, :3]

# function to convert words to integer
def convert_to_int(word):
  word_dict = {"one":1, "two":2, "three":3, "four":4, 
               "five":5, "six":6, "seven":7, "eight":8, 
               "nine":9, "ten":10, "eleven":11, "twelve":12,
               "zero":0, 0:0}
  return word_dict[word]

# Mapping experience variable to integers
X["experience"] = X["experience"].apply(lambda x: convert_to_int(x))

# Selecting the output column 
y = dataset.iloc[:, -1]

# Outlining the model and fittin it
regressor = LinearRegression()

regressor.fit(X, y)

# Saving the model
pickle.dump(regressor, open("model_sal_pred", "wb"))

# loading and testing the model
model = pickle.load(open("model_sal_pred", "rb"))
print(model.predict([[2, 9, 6]]))

