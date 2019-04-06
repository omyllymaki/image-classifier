import os
import pickle
from typing import List, Any


def get_sub_folder_names(dir_path: str) -> List[str]:
    sub_folder_names = [name for name in os.listdir(dir_path)
                        if os.path.isdir(os.path.join(dir_path, name))]
    return sub_folder_names


def get_file_names(dir_path: str, file_extension: str) -> List[str]:
    file_names = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                  if name.endswith(file_extension)]
    return file_names


def save_pickle_file(data: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_file(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
