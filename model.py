"""
Crime Safety Detection — Model Training
Trains a Random Forest Classifier to predict crime_type from contextual features.

Note: as explored during EDA, city/age/time/gender/race/state show little to no
real relationship with crime_type in this dataset (crosstab confirms crime types
are nearly uniformly distributed across cities). Accuracy is expected to be close
to the random baseline (~10% for 10 classes). This script is kept for transparency
and to demonstrate the full workflow, not because it yields a strong model.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from graph_chart import load_data

FEATURE_COLUMNS = ["city", "age_bracket", "time_period", "victim_gender", "victim_race", "state"]


def prepare_features(df):
    y = df["crime_type"]
    X = df[FEATURE_COLUMNS]
    X = pd.get_dummies(X, columns=FEATURE_COLUMNS, drop_first=True)
    return X, y


def build_pipeline():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[("model", model)])
    return pipeline


def train_and_evaluate(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    return pipeline, acc


def save_model(pipeline, columns, path="crime_model.pkl"):
    joblib.dump({"pipeline": pipeline, "columns": list(columns)}, path)
    print(f"\nModel saved to {path}")


def main():
    df = load_data()
    X, y = prepare_features(df)
    pipeline = build_pipeline()
    pipeline, acc = train_and_evaluate(X, y, pipeline)
    save_model(pipeline, X.columns)


if __name__ == "__main__":
    main()
