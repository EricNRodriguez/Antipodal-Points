import imageio
import numpy as np
import glob


class DepthImageImporter:

    def __init__(self, directory):
        self._directory = directory
        self._images = []

    def import_images(self):

        file_paths = "{0}/*.png".format(self._directory)
        image_fps = glob.glob(file_paths)
        imported_images = [imageio.imread(img_fp) for img_fp in image_fps]




        return imported_images

    def import_point_clouds(self):
        pass
