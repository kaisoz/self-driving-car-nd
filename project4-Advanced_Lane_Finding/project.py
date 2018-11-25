import pickle
import argparse
import glob
import matplotlib.image as mpimpg

import matplotlib.pyplot as plt
from image_utils import ImageUtils, AreaOfInterest
import numpy as np
import cv2

from collections import deque
from moviepy.editor import VideoFileClip
import lane_finder



# Source and destination points for perspective transform
src_points = AreaOfInterest(bottom_left = [180, 660],
                            top_left = [550,470],
                            bottom_right = [1170,660],
                            top_right = [760,470])

dst_points = AreaOfInterest(bottom_left =  [src_points.bottom_left[0] + 120, 720],
                            top_left =     [src_points.top_left[0] - 250, 0],
                            bottom_right = [src_points.bottom_right[0] - 150, 720],
                            top_right =    [src_points.top_right[0] + 256,0])

# Global road
road  = None
utils = None


class Road:

    ym_per_pix = 30/720   # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    def __init__(self, img_shape, max_num_lanes=20, max_num_frames_dropped=5):
        self.left_lane, self.right_lane  = Lane(max_num_lanes), Lane(max_num_lanes)
        self.max_num_num_frames_dropped = max_num_frames_dropped
        self.current_lanes_diff = 0
        self.num_frames_dropped = 0
        self.force_raw_discovery = False
        self.img_shape = img_shape
        self.ypoints = np.linspace(0, img_shape[0]-1, img_shape[0])

    def reset_lanes(self):
        self.force_raw_discovery = True
        self.right_lane.reset()
        self.left_lane.reset()

    def lane_distance_is_correct(self, l_detect, r_detect):
        mean_difference = np.mean(r_detect.lane - l_detect.lane)

        if self.current_lanes_diff != 0 and                     \
            (mean_difference < 0.7*self.current_lanes_diff or   \
            mean_difference > 1.3*self.current_lanes_diff):
                return False

        return True

    def calculate_average_curvature(self):
        return (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature)/2

    def calculate_center_offset(self):
        lane_center = (self.right_lane.detected_lanes[-1][-1] + self.left_lane.detected_lanes[-1][-1]) / 2
        center_offset_pixels = abs(self.img_shape[1]/2 - lane_center)
        return (Road.xm_per_pix * center_offset_pixels)

    def add_new_lanes_discovery(self, l_discovery, l_radius, r_discovery, r_radius):
        self.left_lane.add_lane_points(l_discovery, l_radius)
        self.right_lane.add_lane_points(r_discovery, r_radius)
        self.force_raw_discovery = False
        self.num_frames_dropped = 0

    def calculate_current_lane_distance(self):
        mean_difference = np.mean(self.right_lane.avg_lane - self.left_lane.avg_lane)
        if self.current_lanes_diff == 0:
            self.current_lanes_diff = mean_difference
        else:
            self.current_lanes_diff = 0.9*self.current_lanes_diff + 0.1*mean_difference

    def lanes_not_found(self):
        self.num_frames_dropped += 1
        if self.num_frames_dropped == self.max_num_num_frames_dropped:
            self.num_frames_dropped = 0
            self.force_raw_discovery = True


class Lane():
    def __init__(self, max_num_lanes=20):

        self.radius_of_curvature = None
        self.avg_lane = None
        self.last_fit = None
        self.detected_lanes = deque(maxlen=max_num_lanes)
        self.current_prediction = None

    @staticmethod
    def compute_rad_curv(discovery, ypoints):
        fit_cr = np.polyfit(ypoints * Road.ym_per_pix, discovery.lane * Road.xm_per_pix, 2)
        y_eval = np.max(ypoints)
        return ((1 + (2 * fit_cr[0] * y_eval * Road.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    def calculate_average_line(self):
        num_lanes = len(self.detected_lanes)
        averaged_line = np.zeros_like(self.detected_lanes[0])
        for i in range(num_lanes):
            averaged_line += self.detected_lanes[i]
        return averaged_line/num_lanes

    def add_lane_points(self, discovery, radius):
        self.last_fit = discovery.fit
        self.detected_lanes.append(discovery.lane)
        self.avg_lane = self.calculate_average_line()
        self.radius_of_curvature = radius

    def reset(self):
        self.detected_lanes.clear()
        self.radius_of_curvature = None
        self.last_fit = None


def undistort(img):
    gauss = utils.gaussian_blur(img)
    return utils.undistort(gauss)


def apply_thresholds(img):
    sxbinary = utils.sobelx_binary(img, kernel_size=7, thresh_min=50, thresh_max=180)
    sbinary =  utils.schannel_binary(img, thresh_min=170, thresh_max=255)
    lbinary =  utils.lchannel_binary(img, thresh_min=120, thresh_max=255)

    combined = np.zeros_like(sbinary)
    combined[(sxbinary == 1) | ((sbinary == 1) & (lbinary == 1))] = 1

    return combined


def mask_and_wrap(img):
    masked = utils.mask_source_area_of_interes(img)
    return utils.wrap_image(masked)


def discover_lane_points(img):
    ret = False
    l_discovery, r_discovery = None, None

    peak_points = lane_finder.find_peak_points(img)
    if peak_points is not None:
        ret, l_discovery, r_discovery= lane_finder.find_lane_points(img, peak_points, road.ypoints)

    return ret, l_discovery, r_discovery


def compute_lanes_curves(l_discovery, r_discovery):
    l_radius = Lane.compute_rad_curv(l_discovery, road.ypoints)
    r_radius = Lane.compute_rad_curv(r_discovery, road.ypoints)

    return l_radius, r_radius


def discover_lane_points_from_previous_data(img):
    return lane_finder.find_points_from_previous_lanes(img, road.left_lane, road.right_lane, road.ypoints)


def process_image(img, overlay_information=True):

    global road

    undist = undistort(img)
    thresholded = apply_thresholds(undist)
    warped = mask_and_wrap(thresholded)

    if road is None:
        road = Road(warped.shape)
        ret, l_discovery, r_discovery = discover_lane_points(warped)
        if ret is True:
            l_radius, r_radius = compute_lanes_curves(l_discovery, r_discovery)
            road.add_new_lanes_discovery(l_discovery, l_radius, r_discovery, r_radius)
        else:
            road.force_raw_discovery = True
    else:
        if road.force_raw_discovery is False:
            found, l_discovery, r_discovery = discover_lane_points_from_previous_data(warped)
        else:
            road.force_raw_discovery = False
            found, l_discovery, r_discovery = discover_lane_points(warped)

        if found is True and road.lane_distance_is_correct(l_discovery, r_discovery):
            l_radius, r_radius = compute_lanes_curves(l_discovery, r_discovery)
            road.add_new_lanes_discovery(l_discovery, l_radius, r_discovery, r_radius)
            road.calculate_current_lane_distance()
        else:
            road.lanes_not_found()

    result = utils.project_lanes_on_image(undist, warped.shape, road.left_lane.avg_lane, road.right_lane.avg_lane, road.ypoints)

    if overlay_information:

        curvature_left_str = "Left radius of curvature: %.2f m" % road.left_lane.radius_of_curvature
        curvature_right_str = "Right radius of curvature: %.2f m" % road.right_lane.radius_of_curvature
        center_offset_str= "Center offset: %.2f m" % road.calculate_center_offset()

        cv2.putText(result, curvature_left_str,  (100, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
        cv2.putText(result, curvature_right_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
        cv2.putText(result, center_offset_str,   (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)

    return result


def show_processed_image(original, processed):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(original)
    ax1.set_title("Original")
    ax2.imshow(processed)
    ax2.set_title("Processed")
    plt.show()


def load_jpg_images_from(path):
    images = glob.glob("{}/*.jpg".format(path))
    def load_image(images):
        for image_path in images:
            yield mpimpg.imread(image_path)
    return load_image(images)


def processVideo(video_input, video_output):
    print("* Processing video %s to %s" % (video_input, video_output))
    input_video_clip = VideoFileClip(video_input)
    clip = input_video_clip.fl_image(process_image)
    clip.write_videofile(video_output, audio=False)
    print("* Done")

if __name__ == "__main__":

    utils = ImageUtils()

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true", dest="process_video",default=False)
    parser.add_argument("--input", action="store", dest="src_video", default="project_video.mp4")
    args = parser.parse_args()

    utils.compute_perspective_transform_for_points(src_points, dst_points)
    load_image = load_jpg_images_from("camera_cal")

    print("* Calibrating camera...")
    utils.calibrate_camera(load_image, draw_corners=False)
    print("* Done")

    if args.process_video:
        src_video = args.src_video
        dst_video = "{}_processed.mp4".format(src_video.split('.')[-2])
        processVideo(src_video, dst_video)
    else:
        load_image = load_jpg_images_from("test_images")
        for img in load_image:
            processed = process_image(img, overlay_information=False)
            show_processed_image(img, processed)
            road.reset_lanes()
