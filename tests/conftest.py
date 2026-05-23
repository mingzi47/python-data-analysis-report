import pytest
from pathlib import Path


@pytest.fixture
def sample_games_csv(tmp_path):
    """Create a sample games.csv for testing."""
    import pandas as pd

    df = pd.DataFrame({
        "app_id": [730, 570, 440],
        "title": ["CS:GO", "Dota 2", "Team Fortress 2"],
        "date_release": ["2012-08-21", "2013-07-09", "2007-10-10"],
        "rating": [0.87, 0.82, 0.93],
        "positive_ratio": [0.85, 0.80, 0.92],
        "user_score": [8, 7, 9],
        "price_original": [14.99, 0.0, 19.99],
        "price_final": [14.99, 0.0, 19.99],
        "discount": [0.0, 0.0, 0.0],
        "owners": ["1000000-2000000", "500000-1000000", "500000-1000000"],
        "steam_deck": [1, 0, 1],
    })
    path = tmp_path / "games.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_users_csv(tmp_path):
    """Create a sample users.csv for testing."""
    import pandas as pd

    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "products": [42, 15, 0],
        "reviews": [10, 5, 0],
    })
    path = tmp_path / "users.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_recommendations_csv(tmp_path):
    """Create a sample recommendations.csv for testing."""
    import pandas as pd

    df = pd.DataFrame({
        "app_id": [730, 570, 440, 730, 570],
        "user_id": [1, 1, 2, 3, 3],
        "is_recommended": [1, 1, 0, 1, 0],
        "hours": [1234.5, 567.8, 0.0, 42.0, 999.9],
        "date": ["2018-03-15", "2019-06-01", "2020-01-10", "2021-11-20", "2022-05-05"],
        "helpful": [12, 3, 0, 1, 0],
        "funny": [1, 0, 0, 2, 0],
    })
    path = tmp_path / "recommendations.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_metadata_json(tmp_path):
    """Create a sample games_metadata.json for testing."""
    import json

    records = [
        {"app_id": 730, "description": "FPS game", "tags": ["FPS", "Multiplayer"], "genres": ["Action", "Free to Play"], "type": "game", "early_access": 0},
        {"app_id": 570, "description": "MOBA game", "tags": ["MOBA", "Strategy"], "genres": ["Strategy", "Free to Play"], "type": "game", "early_access": 0},
        {"app_id": 440, "description": "Classic FPS", "tags": ["FPS", "Classic"], "genres": ["Action"], "type": "game", "early_access": 0},
    ]
    path = tmp_path / "games_metadata.json"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path
