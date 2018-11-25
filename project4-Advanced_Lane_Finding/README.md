# Project 4 - Advanced Lane Finding
Tomás Tormo Franco (tomas.tormo@gmail.com)

[//]: # (Image References)

[undistorted]: ./images/undistorted.png "Undistorted image"
[processed]: ./images/processed.png "Processed image"
[thresholded]: ./images/thresholded.png "Thresholded image"
[warped]: ./images/warped.png "Warped image"
[lanes]: ./images/lanes.png "Lanes"


## Camera Calibration


In order to calibrate the camera the chessboard pattern images provided with the project are used. This images contain a 9x6 corners in the horizontal and vertical directions respectively.  First, a list of "object points", which are the (x, y, z) coordinates of these chessboard corners in the real-world in 3D space are prepared. It's assumed that the chessboard is fixed on the (x, y) plane at z=0 with the top-left corner at the origin, such that the object points are the same for each calibration image. 

Then, for each calibration image, its chessboard corners are found and appended to a list of 'imgpoints' along with a copy of the object points mentioned before. This lists of 'object points' and corners are the ones used to calibrate the camera and to obtain the camera calibration and distortion coefficients. 

The code for calibrating the camera is located in the *calibrate_camera* function of the ***ImageUtils*** class (line 50).


## Pipeline (single images)

### 1. Distortion-corrected image.

The images are undistorted by using the camera calibration matrix and distortion coefficients obtained in the previous step. For this, the *undistort* function of the *ImageUtils* class (line 65), which in turn calls the *cv2.undistort* function is used.

An example of an image before and after the distortion correction procedure is shown below.

![undistorted][undistorted]




### 2. Thresholding

A several number of combinations of color and gradient thresholds were attempted but finally, a simple combination of color and gradient was found to be working pretty well.
Sobel x gradiend threshold and H and L color thresholds are applied.

An example of a thresholded image is shown below. In this image, Sobel X gradient, H color space and L color space thresholds are applied:

![thresholded][thresholded]

The code for this step is located in the *sobelx_binary*, *schannel_binary* and *lchannel_binary* functions in ***ImageUtils*** class

### 3. Perspective transform

Once the image is thresholded, the region of the image which contain the lanes is isolated and a perspective transform is performed. 

The fist step has been to manually select the source and destination points for the perspective transform. The source points enclose the region of the image which contain the lanes. After that, the are of the image outside this points is masked out and a perspective transform is applied.

The code for this steps is located in the *mask_and_wrap* function of the *project.py* file. This function takes an undistorted image as an argument and applies both the masking and the warping.

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear in the warped image.

![warped][warped]

### 4. Lane Detection

The lane detection is performed using a sliding window search algorithm. This algorithm splits the binary image in 10 rows and slides two windows (one for each lane) from the bottom to the top of the image looking for non zero pixels. 

In order to perform this algorithm, first the histogram of the lower half of the binary image is computed. Then, the peaks of the histogram are found by using the *find_peaks_cwt* function from the *scipy.signal *module. The indices obtained are the center of the windows at the lowest row. The x and y coordinates of all the nonzero pixels in these windows are accumulated into into separate lists. The columns with the maximum number of pixels in the current row will be the center points for the next row.
Found pixels are used with *np.polyfit* to fit a second order polynomial and to generate the x and y lane values for plotting. 

In following frames, a search around a margin of 100 pixels around the previously detected lanes is performed in order to speed up the detection. This is because there is a high probability that the new lanes are in roughly similar positions than the old ones.

Before being used, the lanes values shall meet some sanity conditions:

- Some pixels were detected
- Distance between lanes is greater than 0.7 times the moving average of the lanes distance 
- Distance between lanes is lower than 1.3 times the moving average of the lanes distance

If these sanity conditions are not met, the values are discarded. If more than 5 lane values are discarded in a row, a sliding window search is performed from scratch.

If the values does meet the sanity conditions,  they are accumulated into a Lane class. This class contains the values of the last 20 computed lanes and calculates the average line plotted in each frame along with the moving average of the lanes distance. 

![lanes][lanes]


![alt text][image5]

### 5. Radius of curvature and vehicle position
The radius of curvature is computed according to the formula given in the lectures. The pixel values of the lanes are scaled into meters using the scaling factors defined as follows:

	ym_per_pix = 30/720  # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meteres per pixel in x dimension

To calculate the position of the vehicle, both the image center and the mean of the lane pixels closest to the car are calculated. The difference between them is the offset from the center.


### 6. Example result

The complete pipeline is defined in the *process_image* function in lines 166-209 that performs all these steps and then draws the lanes.  In case of processing a video, this function also draws the radius and position information on to the frame.

Following, an example of the result on a test image is shown:

![processed][processed]

---

## Pipeline (video)

### Video output

The video result can be found at the following [link](https://youtu.be/--7s4ZAz3f0).


---

## Discussion


- The pipeline does not perform really well with the challenge videos. Probably, the thresholding techniques are not robust enough to perform well with darker scenarios where the lanes are not distinguished easily. A possible solution could be to do some tests with other color spaces (such RGB or HSV) and test how they perform in such scenarios. 
Some literature on the subject points to the Laplacian filter for detecting edges, which could be another field of study for lane detection.

- The source region for the perspective transform might be too big for this problem. In cases where the lanes are narrow, the road edges might be detected as lanes making the pipeline fail

- The number of lanes used to calculate the current lane (20) might be to big. This might make the pipeline fail if the lanes shape changes quickly as it won't be fast enough to adjust
