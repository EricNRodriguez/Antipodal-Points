import numpy as np
from scipy.spatial import ckdtree
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import math


class AntipodalGraspSampler:

    def __init__(self, gripper_max_width_px, depth_grad_threshold, static_friction_coef):
        self._gripper_max_width_px = gripper_max_width_px
        self._depth_grad_threshold = depth_grad_threshold
        self._static_friction_coef = static_friction_coef

    def _find_edge_pixels_index(self, grad):
        # gradient of depth image
        dx, dy = grad
        edge_pixels = np.power(dx, 2) + np.power(dy, 2) > self._depth_grad_threshold

        return edge_pixels.nonzero()

    def _surface_normals_at_edges(self, depth_img, edge_pixels, edge_pixel_pair_indexes, grad):
        dx,dy = grad

        edge_pair_surface_normals = []
        for a_i, b_i in edge_pixel_pair_indexes:
            a = edge_pixels[a_i]
            n_a = [dx[a[0]][a[1]], dy[a[0]][a[1]]]
            if n_a[0] == 0 and n_a[1] == 0:
                n_a = [2.2250738585072014e-308, 2.2250738585072014e-308]

            b = edge_pixels[b_i]
            n_b = [dx[b[0]][b[1]], dy[b[0]][b[1]]]
            if n_b[0] == 0 and n_b[1] == 0:
                n_b = [2.2250738585072014e-308, 2.2250738585072014e-308]

            edge_pair_surface_normals.append([n_a, n_b])

        return np.asarray(edge_pair_surface_normals)

    def _close_pairs_ckdtree(self, pixels, max_d):
        tree = ckdtree.cKDTree(pixels)
        pairs = tree.query_pairs(max_d)
        return np.array(list(pairs))

    def sample_antipodal_points(self, depth_image, sample_number):

        # for visualisation
        plt.imshow(depth_image)
        plt.show()

        grad = np.gradient(depth_image)
        edge_pixels = self._find_edge_pixels_index(grad)
        # numpy zip
        edge_pixels = np.column_stack((edge_pixels[0], edge_pixels[1]))
        edge_pixel_pair_indexes = self._close_pairs_ckdtree(edge_pixels, self._gripper_max_width_px)

        edge_surface_normals = self._surface_normals_at_edges(depth_image, edge_pixels, edge_pixel_pair_indexes, grad)

        edge_indexes_a = edge_pixel_pair_indexes[:, 0]
        edge_surface_normals_a = edge_surface_normals[:, 0]
        edge_surface_normals_a = (edge_surface_normals_a.T / np.linalg.norm(edge_surface_normals_a, axis=1)).T

        edge_indexes_b = edge_pixel_pair_indexes[:, 1]
        edge_surface_normals_b = edge_surface_normals[:, 1]
        edge_surface_normals_b = (edge_surface_normals_b.T / np.linalg.norm(edge_surface_normals_b, axis=1)).T


        edge_points_grasp_axis = edge_pixels[edge_indexes_a] - edge_pixels[edge_indexes_b]
        edge_points_grasp_axis_unit = (edge_points_grasp_axis.T / np.linalg.norm(edge_points_grasp_axis, axis=1)).T

        dp_a = np.sum(edge_surface_normals_a * edge_points_grasp_axis_unit, axis=1)
        # accounting for floating point error
        dp_a[dp_a > 1] = 1
        dp_a[dp_a < -1] = -1
        alpha = np.arccos(dp_a)

        dp_b = np.sum(edge_surface_normals_b * (-edge_points_grasp_axis_unit), axis=1)
        dp_b[dp_b > 1] = 1
        dp_b[dp_b < -1] = -1
        beta = np.arccos(dp_b)

        min_static_angle = np.arctan(self._static_friction_coef)
        antipodal_edge_pair_indices = np.where((alpha < min_static_angle) & (beta < min_static_angle))[0]
        color = ['red', 'green', 'blue']
        c = 0
        for i in antipodal_edge_pair_indices[:sample_number]:
            plt.imshow(depth_image)
            a_i = edge_indexes_a[i]
            b_i = edge_indexes_b[i]
            # plt.scatter(edge_pixels[a_i][1], edge_pixels[a_i][0], s=2, color='red')
            # plt.scatter(edge_pixels[b_i][1], edge_pixels[b_i][0], s=2, color='blue')
            plt.plot([edge_pixels[a_i][1], edge_pixels[b_i][1]], [edge_pixels[a_i][0], edge_pixels[b_i][0]],
                     linewidth=0.5, color=color[c % len(color)])
            c += 1

        plt.show()






