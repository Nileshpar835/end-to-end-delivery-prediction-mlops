import pandas as pd

from zenml import step

@step 
def ingest_data():
    df = pd.read_csv("D:\delivery-prediction-mlops\data\Food_Delivery_Times.csv")
    return df
