import os

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ["HF_TOKEN"]
    space_repo = os.environ["HF_SPACE_REPO"]  # e.g. username/superkart-sales-space

    api = HfApi(token=token)
    api.upload_folder(
        folder_path="deployment",
        repo_id=space_repo,
        repo_type="space",
    )
    print(f"✅ Space deployed: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
