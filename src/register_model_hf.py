import os
from huggingface_hub import HfApi


def main():
    token = os.environ["HF_TOKEN"]
    model_repo = os.environ["HF_MODEL_REPO"]  # e.g. username/superkart-sales-rf

    api = HfApi(token=token)
    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj="artifacts/superkart_best_model.joblib",
        path_in_repo="superkart_best_model.joblib",
        repo_id=model_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="artifacts/metrics.json",
        path_in_repo="metrics.json",
        repo_id=model_repo,
        repo_type="model",
    )

    print(f"✅ Model registered: https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
