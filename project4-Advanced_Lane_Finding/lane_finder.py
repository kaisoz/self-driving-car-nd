import numpy as np
from scipy.signal import find_peaks_cwt
from collections import namedtuple

LaneDiscovery = namedtuple("LaneDiscovery", ['fit', 'lane'])


def find_points_from_previous_lanes(image, left_lane, right_lane, ypoints):

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_lane.last_fit[0] * (nonzeroy ** 2) + left_lane.last_fit[1] * nonzeroy + left_lane.last_fit[2] - margin)) & (
        nonzerox < (left_lane.last_fit[0] * (nonzeroy ** 2) + left_lane.last_fit[1] * nonzeroy + left_lane.last_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_lane.last_fit[0] * (nonzeroy ** 2) + right_lane.last_fit[1] * nonzeroy + right_lane.last_fit[2] - margin)) & (
        nonzerox < (right_lane.last_fit[0] * (nonzeroy ** 2) + right_lane.last_fit[1] * nonzeroy + right_lane.last_fit[2] + margin)))

    l_x = nonzerox[left_lane_inds]
    l_y = nonzeroy[left_lane_inds]

    r_x = nonzerox[right_lane_inds]
    r_y = nonzeroy[right_lane_inds]

    left_predict = predict_lane(l_x, l_y, ypoints)
    right_predict = predict_lane(r_x, r_y, ypoints)
    ret = True if left_predict is not None and right_predict is not None else False

    return ret, left_predict, right_predict


def find_lane_points(img, base_pts, ypoints, num_bands = 10, window_width = 0.2):

    height = img.shape[0]
    band_height = int(1./num_bands * height)
    band_width = int(window_width * img.shape[1])

    left_x, left_y, right_x, right_y = [], [], [], []
    base_left, base_right = base_pts

    for i in reversed(range(num_bands)):

        y_min = i*band_height
        y_max = (i+1)*band_height

        x_left_min = base_left-band_width//2
        x_left_max = base_left+band_width//2

        x_right_min = base_right-band_width//2
        x_right_max = base_right+band_width//2

        w_left = img[y_min:y_max, x_left_min:x_left_max]
        w_right = img[y_min:y_max, x_right_min:x_right_max]

        left_y_pt, left_x_pt = np.nonzero(w_left)
        right_y_pt, right_x_pt = np.nonzero(w_right)

        left_x.extend(left_x_pt + base_left-band_width//2)
        left_y.extend(left_y_pt + i*band_height)
        right_x.extend(right_x_pt+ base_right-band_width//2)
        right_y.extend(right_y_pt+ i*band_height)

        s_left = np.sum(w_left, axis=0)
        s_right = np.sum(w_right, axis=0)

        if np.any(s_left > 0):
            base_left = np.argmax(s_left) + base_left-band_width//2
        if np.any(s_right > 0):
            base_right = np.argmax(s_right) + base_right-band_width//2

    left_predict = predict_lane(left_x, left_y, ypoints)
    right_predict = predict_lane(right_x, right_y, ypoints)
    ret = True if left_predict is not None and right_predict is not None else False

    return ret, left_predict, right_predict


def predict_lane(x, y, ypoints):
    if len(x) == 0 or len(y) == 0:
        return None

    fit = np.polyfit(y, x, 2)
    lane = fit[0] * ypoints ** 2 + fit[1] * ypoints + fit[2]

    return LaneDiscovery(fit=fit, lane=lane)


def find_peak_points(lanes):
    hist = np.sum(lanes[int(lanes.shape[0]*0.5):,:], axis=0)
    widths = [100]
    idx = find_peaks_cwt(hist, widths, max_distances=widths, noise_perc=50)
    if len(idx) < 2:
        return None
    return [min(idx), max(idx)]
