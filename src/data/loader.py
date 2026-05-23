import json
import pandas as pd
from pathlib import Path


def download_dataset(output_dir: Path) -> Path:
    import kagglehub

    path_str = kagglehub.dataset_download(
        "antonkozyriev/game-recommendations-on-steam",
        output_dir=str(output_dir),
    )
    return Path(path_str)


def load_games(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def load_users(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def load_recommendations(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def load_metadata(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "app_id": obj.get("app_id"),
                "tags": obj.get("tags", []),
                "genres": obj.get("genres", []),
                "type": obj.get("type"),
                "early_access": obj.get("early_access"),
            })
    return pd.DataFrame(records)
