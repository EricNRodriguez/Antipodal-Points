from importer import DepthImageImporter
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def main():
    importer = DepthImageImporter("./data")
    images = importer.import_images()
    img = images[1][:,:,0]

    y, x = img.shape
    cropx = 200
    cropy = 200
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    img_crop = img[starty:starty+cropy,startx:startx+cropx]

    plt.imshow(img_crop)
    plt.show()

    depth_image = np.load('./data/depth_3.npy')[:,:,0]
    depth_image_crop = depth_image[starty:starty+cropy,startx:startx+cropx]
    plt.imshow(depth_image_crop)
    plt.show()

    # laplace works much better!
    depth_image_gradient_2 = ndimage.laplace(depth_image_crop)
    depth_image_gradient_22 = depth_image_gradient_2 > 0.004
    plt.imshow(depth_image_gradient_22)

    # we grab the indexes of the ones
    epsilon_y, epsilon_x = np.where(depth_image_gradient_22)

    antipodal_points = []
    d = 0
    while len(antipodal_points) < 2:
        # we chose one index randomly
        i = np.random.randint(len(epsilon_x))  # select one of the coordinates that match
        antipodal_points.append([epsilon_x[i], epsilon_y[i]])
        if d % 2 == 0 and d != 0:
            plt.scatter(antipodal_points[-1][0], antipodal_points[-1][1])
            plt.scatter(antipodal_points[-2][0], antipodal_points[-2][1])
            plt.show()
            plt.imshow(depth_image_gradient_22)
        d+=1
    plt.show()



    return

if __name__ == '__main__':
    main()