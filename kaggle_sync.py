"""Download the trained model from the Kaggle notebook output using the Kaggle API token."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"
KERNEL = "olaelshiekh/cnn-realdata-covid19"
WANTED_FILES = ("best_model.keras", "class_names.json")


def get_token() -> str | None:
    load_dotenv(BASE_DIR / ".env")
    return os.getenv("KAGGLE_API_TOKEN")


def ensure_model() -> Path:
    """Return the local model path, downloading it from Kaggle first if it is missing."""
    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        return MODEL_PATH

    if not get_token():
        raise RuntimeError(
            "KAGGLE_API_TOKEN is not set. Add it to DL_Deployment/.env "
            "(KAGGLE_API_TOKEN=<your token>) so the model can be downloaded."
        )

    from kaggle import KaggleApi  # imported late: it authenticates on import

    api = KaggleApi()
    api.authenticate()
    MODEL_DIR.mkdir(exist_ok=True)
    for name in WANTED_FILES:
        api.kernels_output(KERNEL, path=str(MODEL_DIR), file_pattern=f"^{name}$", force=True, quiet=False)

    for log in MODEL_DIR.glob("*.log"):  # the API always also fetches the notebook log
        log.unlink()

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Download finished but {MODEL_PATH.name} was not found in {MODEL_DIR}.")
    return MODEL_PATH


if __name__ == "__main__":
    print(f"Model ready: {ensure_model()}")
