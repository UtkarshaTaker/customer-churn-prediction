import pandas as pd
from sklearn.preprocessing import LabelEncoder


import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

data = load_data(r"C:\Users\taker\customer-churn-prediction\data\Telco-Customer-Churn.csv")
print(data.head())


def preprocess_data(df):

    # Remove customerID column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df