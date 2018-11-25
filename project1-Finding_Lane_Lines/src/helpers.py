import numpy as np
import cv2

class SideLane:

    def __init__(self, max_lines_averaged = 160, slope_diff_threshold = 0.16):

        self.max_lines_averaged   = max_lines_averaged
        self.slope_diff_threshold = slope_diff_threshold

        self.m = self.b = 0
        self.x1 = self.x2 = self.y1 = self.y2 = 0

        self.x1_acc = self.x2_acc = self.y1_acc = self.y2_acc = 0
        self.m_acc = self.b_acc = self.lines = 0

    def accumulate(self, line):
        if self.lines == self.max_lines_averaged:
            self.reset_accumulators()

        for x1,y1,x2,y2 in line:

            m = (y2 - y1)/(x2 - x1)
            if self.m == 0 or abs(self.m - m) < self.slope_diff_threshold:
                self.x1_acc += x1
                self.y1_acc += y1
                self.x2_acc += x2
                self.y2_acc += y2
                self.m_acc  += m
                self.b_acc  += (y2 - m * x2)

                self.lines += 1
                self.average()

    def reset_accumulators(self):
        self.x1_acc = self.x2_acc = self.y1_acc = self.y2_acc = 0
        self.m_acc = self.b_acc = self.lines = 0

    def average(self):
        self.x1 = int(self.x1_acc/self.lines)
        self.y1 = int(self.y1_acc/self.lines)
        self.x2 = int(self.x2_acc/self.lines)
        self.y2 = int(self.y2_acc/self.lines)
        self.m  = self.m_acc/self.lines
        self.b  = int(self.b_acc/self.lines)

    def extrapolate(self, img):
        self.y1 = img.shape[0]
        self.x1 = int((self.y1 - self.b)/self.m)
        self.y2 = int(img.shape[0]/1.60)
        self.x2 = int((self.y2 - self.b)/self.m)



left  = SideLane()
right = SideLane()

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def line_has_positive_slope(line):
    for x1,y1,x2,y2 in line:
        return True if ((y2 - y1)/(x2 - x1))  < 0 else False

def draw_lines(img, lines, color=[255, 0, 0], thickness=2, accumulate_values= False, extrapolate=False):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).

    By default, lines are not extrapolated. If you want lines o be extrapolated, set 'extrapolate' to True

    Line points, slope and intercept can accumulated in order to calculate an the average line. This is useful with Videos
    If accumulate_values is set to False, the accumulated values are reset each time a new draw is requested

    """
    if accumulate_values is False:
        left.reset_accumulators()
        right.reset_accumulators()

    if extrapolate is False:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        for line in lines:
            left.accumulate(line) if line_has_positive_slope(line) else right.accumulate(line)

        left.extrapolate(img)
        right.extrapolate(img)

        cv2.line(img, (  left.x1,   left.y1), (  left.x2,   left.y2) , color, thickness)
        cv2.line(img, ( right.x1,  right.y1), ( right.x2,  right.y2) , color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an array of found lines
    """

    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
