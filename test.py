import pandas as pd

df = pd.read_csv("D:\delivery-prediction-mlops\data\Food_Delivery_Times.csv")
print(df.head())
print(df.info())
print(df.describe())