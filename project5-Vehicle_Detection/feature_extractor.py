import numpy as np
import cv2
from skimage.feature import hog


class Extractor:
    def __init__(self, color_space, spatial_size, hist_bins, hog_orient, hog_channel, hog_pix_per_cell, hog_cell_per_block):
        self.color_space=color_space
        self.spatial_size=spatial_size
        self.hist_bins=hist_bins
        self.hog_orient=hog_orient
        self.hog_channel=hog_channel
        self.hog_pix_per_cell=hog_pix_per_cell
        self.hog_cells_per_block=hog_cell_per_block

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, visualize=False, feature_vector=True):

        if visualize:
            features, hog_image = hog(img, orientations=self.hog_orient,
                                      pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                                      cells_per_block=(self.hog_cells_per_block, self.hog_cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize, feature_vector=feature_vector)
            return features, hog_image
        else:
            features = hog(img, orientations=self.hog_orient,
                           pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                           cells_per_block=(self.hog_cells_per_block, self.hog_cells_per_block),
                           transform_sqrt=True,
                           visualise=visualize, feature_vector=feature_vector)
            return features

    def bin_spatial(self, img):
        return cv2.resize(img, self.spatial_size).ravel()

    def color_hist(self, img, bins_range=(0, 256)):
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins, range=bins_range)
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def change_color_space(self, img):
        color_space = {
            'HSV':cv2.COLOR_RGB2HSV,
            'LUV':cv2.COLOR_RGB2LUV,
            'HLS':cv2.COLOR_RGB2HLS,
            'YUV':cv2.COLOR_RGB2YUV,
            'YCrCb':cv2.COLOR_RGB2YCrCb,
        }.get(self.color_space, None)

        if color_space is None:
            return img.copy()
        else:
            return cv2.cvtColor(img, color_space)

    def from_image(self, img):

        img  = self.change_color_space(img)
        spatial_features = self.bin_spatial(img)
        hist_features = self.color_hist(img)

        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.extend(self.get_hog_features(img[:,:,channel], visualize=False))
        else:
            hog_features = self.get_hog_features(img[:,:, self.hog_channel], visualize=False)

        return np.concatenate((spatial_features, hist_features, hog_features))