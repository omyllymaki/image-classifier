from abc import abstractmethod
from typing import List, Dict, Union

from PIL.Image import Image


class ImageLoaderInterface:

    @abstractmethod
    def load_images(self, *args) -> List[Image]:
        """Loads list of images"""
        raise NotImplementedError

    @abstractmethod
    def load_images_with_labels(self, *args) -> List[Dict[str, Union[List[str], Image]]]:
        """Loads list of dicts where dict includes list of labels (key 'y') and image (key 'x')
        [{'x': image, 'y': ['Label1', 'Label2', ...]}, ...]
        """
        raise NotImplementedError
