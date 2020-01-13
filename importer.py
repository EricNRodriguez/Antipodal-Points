import imageio
import numpy as np
import glob


class DepthImageImporter:

    def __init__(self, directory):
        self._directory = directory
        self._images = []

    def import_images(self):

        file_paths_png = "{0}/*.png".format(self._directory)
        image_fps = glob.glob(file_paths_png)
        imported_images = [imageio.imread(img_fp) for img_fp in image_fps]

        file_paths_npy = "{0}/*.npy".format(self._directory)
        depth_fps = glob.glob(file_paths_npy)
        imported_depth_images = [np.load(depth_fp)[:, :, 0] for depth_fp in depth_fps]

        return imported_images, imported_depth_images
