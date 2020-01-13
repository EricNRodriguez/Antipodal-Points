from importer import DepthImageImporter
from grasp_sampler import AntipodalGraspSampler


def crop_image(img):
    y_max = int(img.shape[0] - img.shape[0]/4)
    return img[0:y_max]


def main():

    importer = DepthImageImporter("./data")
    colour_images, depth_images = importer.import_images()

    sampler = AntipodalGraspSampler(70, 0.000095, 0.4)
    for i, d in zip(colour_images[:5], depth_images[:5]):
        i = crop_image(i)
        d = crop_image(d)
        sampler.sample_antipodal_points(d, 10)

    return


if __name__ == '__main__':
    main()
