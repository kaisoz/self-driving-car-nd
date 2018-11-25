import cv2
import numpy as np
import matplotlib.pyplot as plt

class AreaOfInterest:

    def __init__(self, bottom_left, top_left, bottom_right, top_right):
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.top_right = top_right

    def to_np_int32_array(self):
        return np.int32([self.bottom_left, self.bottom_right, self.top_right, self.top_left])

    def to_np_float32_array(self):
        return np.float32([self.bottom_left, self.bottom_right, self.top_right, self.top_left])



class ImageUtils:

    def __init__(self):
        self.mxt = None
        self.dist = None
        self.M = None
        self.M_inv = None

    def __prepare_objectpoints(self, corners_in_rows=9, corners_in_colums=6, channels=3):
        objpoints = np.zeros((corners_in_rows * corners_in_colums, channels), np.float32)
        objpoints [:, :2] = np.mgrid[0:corners_in_colums, 0:corners_in_rows].T.reshape(-1, 2)
        return objpoints

    def __find_chessboard_corners(self, img, corners_in_rows=9, corners_in_colums=6, draw_corners=False):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (corners_in_rows, corners_in_colums), None)
        if draw_corners is True:
            img = cv2.drawChessboardCorners(img, (corners_in_rows, corners_in_colums), corners, ret)
            self.show_images({'chessboard':img})
        return ret, corners

    def compute_perspective_transform_for_points(self, src_points, dst_points):
        self.src_points = src_points
        self.M = cv2.getPerspectiveTransform(src_points.to_np_float32_array(), dst_points.to_np_float32_array())
        self.M_inv = cv2.getPerspectiveTransform(dst_points.to_np_float32_array(), src_points.to_np_float32_array())


    def calibrate_camera(self, calibration_images_loader, draw_corners=False):
        objpoints, imgpoints = [],[]
        img_shape = None
        objp = self.__prepare_objectpoints()

        for img in calibration_images_loader:
            ret, corners = self.__find_chessboard_corners(img, draw_corners=draw_corners)
            if ret is True:
                imgpoints.append(corners)
                objpoints.append(objp)
            if img_shape is None:
                img_shape = img.shape[0:2]

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def gaussian_blur(self, img, kernel_size=5):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def sobelx_binary(self, img, kernel_size, thresh_min, thresh_max):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary

    def schannel_binary(self, img, thresh_min, thresh_max):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
        return s_binary

    def lchannel_binary(self, img, thresh_min, thresh_max):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1
        return l_binary

    def wrap_image(self, img):
        img_size = (img.shape[1], img.shape[0])


        return cv2.warpPerspective(img, self.M, img_size)#, flags=cv2.INTER_LINEAR)

    def unwrap_image(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M_inv, img_size)#, flags=cv2.INTER_LINEAR)

    def mask_source_area_of_interes(self, img):
        mask = np.zeros_like(img)
        ignore_mask_color = 255
        cv2.fillPoly(mask, [self.src_points.to_np_int32_array()], ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    def project_lanes_on_image(self, img, warped_shape, left_lane, right_lane, ypoints):
        warp_zero = np.zeros(warped_shape).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_lane, ypoints]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane, ypoints])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = self.unwrap_image(color_warp)
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    def draw_lanes_on_warped_image(self, warped, left_lane, right_lane, ypoints):
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)

        left_line_window1 = np.array([np.transpose(np.vstack([left_lane, ypoints]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane, ypoints])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_lane, ypoints]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane, ypoints])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 0, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(result)
        plt.plot(left_lane, ypoints, color='red')
        plt.plot(right_lane, ypoints, color='red')
        plt.xlim(0, warped.shape[1])
        plt.ylim(warped.shape[0],0)


