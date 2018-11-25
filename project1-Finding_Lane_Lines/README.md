# Project 1 - Finding Lane Lines on the Road


Tomás Tormo Franco (tomas.tormo@gmail.com)


[//]: # (Image References)


[image1]: ./test_images/processed/solidWhiteCurve_processed.jpg        "solidWhiteCurve_processed"
[image2]: ./test_images/processed/solidWhiteRight_processed.jpg         "solidWhiteRight_processed"
[image3]: ./test_images/processed/whiteCarLaneSwitch_processed.jpg  "whiteCarLaneSwitch_processed"
[image4]: ./test_images/processed/solidYellowLeft_processed.jpg           "solidYellowLeft_processed"
[image5]: ./test_images/processed/solidYellowCurve_processed.jpg        "solidYellowCurve_processed"
[image6]: ./test_images/processed/solidYellowCurve2_processed.jpg      "solidYellowCurve2_processed"


[image7]:  ./test_images/extrapolated/solidWhiteCurve_extrapolated.jpg        "solidWhiteCurve_extrapolated"
[image8]:  ./test_images/extrapolated/solidWhiteRight_extrapolated.jpg         "solidWhiteRight_extrapolated"
[image9]:  ./test_images/extrapolated/whiteCarLaneSwitch_extrapolated.jpg  "whiteCarLaneSwitch_extrapolated"
[image10]:  ./test_images/extrapolated/solidYellowLeft_extrapolated.jpg         "solidYellowLeft_extrapolated"
[image11]: ./test_images/extrapolated/solidYellowCurve2_extrapolated.jpg     "solidYellowCurve2_extrapolated"
[image12]: ./test_images/extrapolated/solidYellowCurve2_extrapolated.jpg     "solidYellowCurve2_extrapolated"

---

## Reflection

### 1.  Pipeline description


The processing pipeline consists of 6 steps: 

1. The images is converted to grayscale.

2. A Gaussian Blur with a kernel value of 3 is aplied in order to suppress possible noises and spurious gradients.

3. Once smoothed, a Canny Edges function is applied to detect the edges. The function is called with a low threshold of 91 and a high threshold of 182.

4. From the Canny Edges result, a region of interest is isolated.  The chosen shape has been a triangular one, since it works quite well given that the lanes start from the front sides of the car and go to the middle of the image. The region of interest has to be applied after the Canny function, otherwise the latter detects and remarks the borders of the triangular shape.

5. Once the edges are detected and isolated, the Hough Lines probabilistic function is called. In order to ease program behaviour parametrization, this function has been modified to only return an array of found lines. The function has been called with the following parameters:

	- rho = 2, 
	- theta = pi/180, 
	- threshold = 15
	- min_line_length = 30
	- max_line_gap = 10
	     
	
6. The last step is to add the segments returned by the Hough Lines to the original image

Following, all processed test images are shown. Please note that, in this images, **the lines are intentionally not averaged/extrapolated** :


![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


In order to draw a single line on the left and right lanes, the draw_lines function first divides the lines in two big groups: left lines and right lines. Lines separation is performed by its slope: positive slope lines are grouped in the left group and negative slope lines are grouped in the right group. Each group is represented in a class called **SideLane**. 

For each line, this class calculates its slope and intercept and accumulates them along with its line end values. Then, it calculates the average of all the accumulated points, slopes and intercepts to get a unique line.

Finally, this average line is extrapolated to increase its length. Having both the slope and the intercept values, the line equation is be used to get the points the line would pass at the bottom of the image (using image.shape[0] for y1) and at the top of the region of the interest.

When processing a video, some additional steps are applied in order to reduce jittering. First, only the last 6.5 seconds are averaged (160 frames). This helps to avoid possible accumulated errors which makes the lines deviate from its place. Second, in order to be accumulated, the line slope cannot be 1.6 units higher or lower than the average slope. Otherwise, the line is discarded.

Following, all processed test images are shown along with the extrapolated lines:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]


### 2. Potential shortcomings


One potential shortcoming would be what would happen when the image has brighter and darker zones, for example different asphalt colours or shadows. This would make the algorithm miss lane lines and potentionally detect incorrect lines, generating lots of clutter.

Another shortcoming would be the region of interest shape. Currently it's quite coupled to the vision area shown in the images. Roads with different shapes, where lanes does not fit in a triangular shape wouldn't be detected.


### 3. Possible improvements

A possible improvement would be to use a more complex region of interest which would fit to different road shapes other than the usual and clean ones. 
 
 Another potential improvement could be to detect the edges in all colour channels and then apply the Hough transform. This could help avoiding problems caused by the color changes or brightness.
