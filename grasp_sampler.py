import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ckdtree
import matplotlib.pyplot as plt

class AntipodalGraspSampler:

    def __init__(self, gripper_max_width_px, depth_grad_threshold, static_friction_coef):
        self._gripper_max_width_px = gripper_max_width_px
        self._depth_grad_threshold = depth_grad_threshold
        self._static_friction_coef = static_friction_coef

    def _find_edge_pixels_index(self, depth_image):

        # gradient of depth image
        dx, dy = np.gradient(depth_image)
        edge_pixels = np.power(dx, 2) + np.power(dy,2) > self._depth_grad_threshold

        return edge_pixels.nonzero()

    def _close_pairs_ckdtree(self, pixels, max_d):
        tree = ckdtree.cKDTree(pixels)
        pairs = tree.query_pairs(max_d)
        return np.array(list(pairs))

    def sample_antipodal_points(self, depth_image, sample_number):

        # for visualisation
        plt.imshow(depth_image)
        plt.show()

        edge_pixels = self._find_edge_pixels_index(depth_image)
        edge_pixels = np.column_stack((edge_pixels[0], edge_pixels[1]))

        # distances = sd.squareform(sd.pdist(edge_pixels))
        # valid_indices = np.where(distances < self._gripper_max_width_px)
        # print(valid_indices)
        # contact_points_a = edge_pixels[valid_indices[:,0],:]
        # contact_points_b = edge_pixels[valid_indices[:,1],:]

        # depth_image_threshold = depth_image.copy()
        # depth_image_threshold[np.where(~edge_pixels)] = 0
        #
        # distances = sd.squareform(sd.pdist(edge_pixels))
        # valid_indices = np.where(distances < self._gripper_max_width_px)
        #
        # valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        # print(edge_pixels.shape)
        #
        # contact_points_a = edge_pixels[valid_indices[:,0],:]
        # contact_points_b = edge_pixels[valid_indices[:,1],:]
        # plt.imshow(depth_image)
        # for x,y in edge_pixels:
        #     plt.scatter(y,x)
        #
        # plt.show()

        # edge_pixels = depth_image[edge_pixels_index]

        # print(edge_pixels_index[1])
        #
        # plt.imshow(edge_pixels)
        # plt.show()



        # # laplace works much better!
        # depth_image_gradient_2 = ndimage.laplace(depth_image_crop)
        # depth_image_gradient_22 = depth_image_gradient_2 > 0.004
        # plt.imshow(depth_image_gradient_22)
        # # plt.contour(depth_image_crop)
        plt.imshow(depth_image)
        q = self._close_pairs_ckdtree(edge_pixels, 2)
        #
        # plt.plot(X[:, 1], X[:, 0], '.r')
        # plt.plot(X[p, 1].T, X[p, 0].T, '-b')
        # plt.figure()
        # plt.plot(X[:, 1], X[:, 0], '.r')
        # plt.plot(X[q, 1].T, X[q, 0].T, '-b')
        plt.scatter(edge_pixels[q, 1], edge_pixels[q, 0], s=0.5, color='red')

        plt.show()





