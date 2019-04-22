import logging
import os
from typing import List, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DataLoader:

    def get_labeled_image_data(self,
                               path: str,
                               file_extension: str = 'jpg') -> List[dict]:
        file_paths_by_label = self._get_file_names_from_sub_folders(path, file_extension)
        image_data = self._load_images_from_file_paths_by_label(file_paths_by_label)
        return image_data

    def get_image_data(self,
                       dir_path: str,
                       file_extension: str = 'jpg') -> list:
        file_paths = self._get_file_paths(dir_path, file_extension)
        images = self._load_images_from_file_paths(file_paths)
        return images

    def _get_file_names_from_sub_folders(self, path: str, file_extension: str) -> Dict[str, List[str]]:
        file_names = {}
        sub_folder_names = self._get_sub_folder_names(path)
        for folder_name in sub_folder_names:
            folder_path = os.path.join(path, folder_name)
            file_names[folder_name] = self._get_file_paths(folder_path, file_extension)
        return file_names

    @staticmethod
    def _load_images_from_file_paths_by_label(file_paths_by_label: Dict[str, List[str]]):
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
    def _load_images_from_file_paths(file_paths: List[str]):
        n_files = len(file_paths)
        images = []
        for index, file_path in enumerate(file_paths, 1):
            image = Image.open(file_path)
            n_dimensions = len(np.array(image).shape)
            if n_dimensions == 3:
                images.append(image.copy())
            image.close()
            logger.info(f'{index}/{n_files} images loaded')
        return images

    @staticmethod
    def _get_sub_folder_names(dir_path: str) -> List[str]:
        sub_folder_names = [name for name in os.listdir(dir_path)
                            if os.path.isdir(os.path.join(dir_path, name))]
        return sub_folder_names

    @staticmethod
    def _get_file_paths(dir_path: str, file_extension: str) -> List[str]:
        file_names = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                      if name.endswith(file_extension)]
        return file_names
