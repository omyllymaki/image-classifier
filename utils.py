import os
from typing import List


def get_file_paths(dir_path: str, file_extension: str) -> List[str]:
    file_names = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                  if name.endswith(file_extension)]
    return file_names