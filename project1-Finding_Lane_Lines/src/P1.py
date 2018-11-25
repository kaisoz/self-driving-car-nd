from helpers import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip

class LaneFinder:

    def __init__(self, gaussian_blur_kernel_size = 3, canny_low_threshold = 91, canny_high_threshold = 182,
                 hough_rho = 2, hough_theta = np.pi/180, hough_threshold = 15, hough_min_line_length = 30, hough_max_line_gap = 10):

        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold =hough_threshold
        self.hough_min_line_length  = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap


    def __process(self, image, thickness=2, extrapolate_lines=False, accumulate_values=False):

        gray = grayscale(image)
        blurry = gaussian_blur(gray, self.gaussian_blur_kernel_size)
        edges = canny(blurry, self.canny_low_threshold, self.canny_high_threshold)

        apex = ((image.shape[1]/1.95) , (image.shape[0]/1.69))
        vertices = np.array([[(0, image.shape[0]), apex, apex, (image.shape[1], image.shape[0])]], dtype=np.int32)
        processed = region_of_interest(edges, vertices)

        lines = hough_lines(processed, self.hough_rho, self.hough_theta, self.hough_threshold, self.hough_min_line_length, self.hough_max_line_gap)

        lines_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        draw_lines(lines_img, lines, thickness=thickness, accumulate_values=accumulate_values, extrapolate=extrapolate_lines)
        return weighted_img(lines_img, image)

    def __processVideoFrame(self, image):
        return self.__process(image, thickness=10, extrapolate_lines=True, accumulate_values=True)


    def processImage(self, image_path, out_path=None, suffix ="processed"):
        image = mpimg.imread(image_path)
        processed = self.__process(image)

        image_name = os.path.basename(image_path)
        if out_path is None:
            out_path = os.path.dirname(image_path)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        out_path = "%s/%s_%s.jpg" % (out_path, image_name[:image_name.rfind(".")], suffix)

        print("-- Processing image %s to %s..." % (image_path, out_path), end='')
        plt.imsave(out_path, processed, format="jpg")
        print("Done")

    def processImagesDirectory(self, directory, images_suffix="processed"):

        out_path = ("%s/%s" % (directory, images_suffix))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print("* Processing %s directory to %s" % (directory, out_path))

        for entry in os.listdir(directory):
            path = "%s/%s" % (directory, entry)
            if os.path.isdir(path) is False:
                self.processImage(path, out_path, images_suffix)

    def processVideo(self, video_input, video_output):

        print("* Processing video %s to %s" % (video_input, video_output))
        input_video_clip = VideoFileClip(video_input)
        clip = input_video_clip.fl_image(self.__processVideoFrame)
        clip.write_videofile(video_output, audio=False)
        print("* Done")


if __name__ == "__main__":

    finder = LaneFinder()

    # Single image
    finder.processImage("../test_images/solidWhiteCurve.jpg", "../test_images/processed", suffix="single")

    # Image directory
    finder.processImagesDirectory("../test_images", images_suffix="processed")

    # Video1
    finder.processVideo("../test_videos/solidWhiteRight.mp4", "../test_videos/solidWhiteRight_processed.mp4")

    # Video2
    finder.processVideo("../test_videos/solidYellowLeft.mp4", "../test_videos/solidYellowLeft_processed.mp4")

