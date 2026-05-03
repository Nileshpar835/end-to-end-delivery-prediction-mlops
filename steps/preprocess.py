from zenml import step
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

@step
def preprocess_data(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:

    df = df.drop(columns=["Order_ID"])

    df.fillna({
        "Weather": "Unknown",
        "Traffic_Level": "Unknown",
        "Time_of_Day": "Unknown",
        "Courier_Experience_yrs": df["Courier_Experience_yrs"].mean()
    }, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    # 🔥 IMPORTANT FIX
    df = df.astype(float)

    X = df.drop("Delivery_Time_min", axis=1)
    y = df["Delivery_Time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test