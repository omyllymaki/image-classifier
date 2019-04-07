import pickle
from typing import Any


def save_pickle_file(data: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_file(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
