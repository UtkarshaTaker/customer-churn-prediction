import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


def split_data(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    model = LGBMClassifier()

    model.fit(X_train, y_train)

    return model
#print("Model training completed!")