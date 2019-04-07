import os
from typing import List, Dict

import numpy as np
from PIL import Image


class DataLoader:

    def get_labeled_data(self,
                         path: str):
        file_paths_by_label = self.get_file_names_from_sub_folders(path, 'jpg')
        image_data = self.load_images_from_file_paths(file_paths_by_label)
        return image_data

    def get_file_names_from_sub_folders(self, path: str, file_extension: str) -> Dict[str, List[str]]:
        file_names = {}
        sub_folder_names = self.get_sub_folder_names(path)
        for folder_name in sub_folder_names:
            folder_path = os.path.join(path, folder_name)
            file_names[folder_name] = self.get_file_names(folder_path, file_extension)
        return file_names

    @staticmethod
    def load_images_from_file_paths(file_paths_by_label: Dict[str, List[str]]):
        data = []
        for label, file_paths in file_paths_by_label.items():
            for file_path in file_paths:
                image = Image.open(file_path)
                n_dimensions = len(np.array(image).shape)
                if n_dimensions == 3:
                    data.append({'x': image.copy(), 'y': label})
                image.close()
        return data

    @staticmethod
    def get_sub_folder_names(dir_path: str) -> List[str]:
        sub_folder_names = [name for name in os.listdir(dir_path)
                            if os.path.isdir(os.path.join(dir_path, name))]
        return sub_folder_names

    @staticmethod
    def get_file_names(dir_path: str, file_extension: str) -> List[str]:
        file_names = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                      if name.endswith(file_extension)]
        return file_names
