import logging
import os
from typing import List

import pandas as pd

from data_loaders.image_loader_labels_from_folders import ImageLoaderFromFolders

logger = logging.getLogger(__name__)


class ImageLoaderFromCSV(ImageLoaderFromFolders):

    def load_images_with_labels(self,
                                labels_file_path: str,
                                file_extension: str = 'jpg',
                                split_labels_by: str = ',',
                                is_header=False) -> List[dict]:
        if is_header:
            header = 'infer'
        else:
            header = None
        df = pd.read_csv(labels_file_path, header=header, engine='python')

        data = []
        for i, row in df.iterrows():
            labels = [label.strip() for label in row.iloc[1].split(split_labels_by)]
            image_path = row.iloc[0]
            image = self._open_image(image_path)
            if self._is_image_valid(image):
                data.append({'x': image.copy(), 'y': labels})

        return data
