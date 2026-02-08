import json
import os

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Product_Store_Sales_Total"


def main():
    hf_dataset_repo = os.environ["HF_DATASET_REPO"]

    # Load prepared splits from HF Dataset Hub
    train_df = load_dataset(hf_dataset_repo, data_files="processed/train_prepared.csv")[
        "train"
    ].to_pandas()
    test_df = load_dataset(hf_dataset_repo, data_files="processed/test_prepared.csv")[
        "train"
    ].to_pandas()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Feature groups (must match your prepared data columns)
    numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age"]
    categorical_features = [
        "Product_Sugar_Content",
        "Product_Type",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(random_state=42)

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate quickly here too (store metrics for next step)
    preds = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/superkart_best_model.joblib")

    with open("artifacts/train_results.json", "w") as f:
        json.dump(
            {
                "rmse": rmse,
                "best_params": search.best_params_,
            },
            f,
            indent=2,
        )

    print("✅ Training complete. RMSE:", rmse)
    print("✅ Saved model to artifacts/superkart_best_model.joblib")


if __name__ == "__main__":
    main()
