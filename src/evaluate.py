import json
import os

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "Product_Store_Sales_Total"


def main():
    # load test split from HF dataset
    hf_dataset_repo = os.environ["HF_DATASET_REPO"]
    test_df = load_dataset(hf_dataset_repo, data_files="processed/test_prepared.csv")[
        "train"
    ].to_pandas()

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    model = joblib.load("artifacts/superkart_best_model.joblib")
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
