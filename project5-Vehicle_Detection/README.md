# Project 5 - Vehicle Detection
Tomás Tormo Franco (tomas.tormo@gmail.com)

[vehicle_nonvehicle]: ./images/vehicle_nonvehicle.png
[hog_features]: ./images/hog_features.png
[boxes]: ./images/boxes.png
[heatmap1]: ./images/heatmap1.png
[heatmap2]: ./images/heatmap2.png
[video1]: ./images/video_screenshot1.jpg
[video2]: ./images/video_screenshot2.jpg
[video3]: ./images/video_screenshot3.jpg

---

## 1. Features extraction

For this project, the features are extracted from the images provided with the project. Since the number of available images is almost the same images for both vehicle and non-vehicle classes (8792 and 8968 respectively), the dataset can be cosidered ti be well balanced.

The code for loading the images and extracting its features is located in the `extract_features_from_images_in_path` function of the `detector.py` module. To load the images, it uses the `load_image` generator located in the line XXXX of the `detector.py` module.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


![vechicle_nonvehicle][vehicle_nonvehicle]

After several combinations, the best results were obtained with the combination of the following features:

- **Color Space**
Several color spaces were tested, like YUV, RGB and YCrCb. The best results were obtained with the YCrCb color space

- **Spatial binning of color**
Images are resized down to 16x16 pixels. This resolution proved to reduce the number of features while preserving the relevant ones.

- **Color histogram**
Number of bins is set to 16. This value provides a good tradeoff between the number of features detected and the memory used.

- **Histogram of Orient Gradients**
The HOG features are extracted from all image channels. After several parameter explorations, the ones that provided the best results were the following:
	- Orientations: 9
	- Pixels per cell: (8,8)
	- Cells per block: (2,2)
	
The feature extraction functions are grouped in a class called `FeatureExtractor`. This class is initialized with all the params mentioned above. In this way, there's only a single point for parameter tuning, since once initialized, this class is reused along the project.

Following, a vehicle image of the result of the color space transformation and HOG features per channel is shown:

![Features][hog_features]


## 2. Training the classifier

Once all features from vehicle and non-vehicle images are extracted, they are stacked and normalized using an `StandardScaler`. The code for this step is located in the `train_classifier` function of the `detector.py` module.

The normalized examples along with its labels are shuffled and splitted in 90% of the examples to the training set and 10% of the examples to the test set. The code for splitting the sets is located in the `split_dataset` function of the `detector.py` module.

Finally, a Linear SVC  classifier is trained, achieving a accuracy of 98.6% 

## 3. Hog Sub-sampling Window Search

In order to search for vehicles in a image, a Hog Sub-sampling Window search is used. This method was chosen due to its efficiency over the sliding window method.

The `find_cars` function of the lectures is used as implementation. This function extracts the features using parameters set in the `FeatureExtractor` class created during the training phase.

As extra parameters, the function uses an `scaling` factor of 1.5 and a `ystart` and `ystop` of 400 and 656 respectively in order to isolate the part of the image where the cars appear.

Here are some example images with the vehicles detected:

![boxes][boxes]

The overlapping windows and false positives are filtered by creating a heatmap and applying a threshold with a value of 1. This value gives a good result removing all the overlapping windows and false positives. 

Once thresholded, the  `scipy.ndimage.measurements.label()` function is used to identify individual blobs, which are assumed to correspond to a vehicle. Finally, a bounding boxes is constructed around each detected blob. 

Following, the previous images filtered along with their heatmap are shown:

![heatmap1][heatmap1]
![heatmap2][heatmap2]

---

## 4. Video Implementation

The video processing pipeline follows the same procedure as the single images pipeline. The only difference is the way the heatmap is calculated. 

In this case, the heatmap thresholded is the sum of the heatmaps of the last 10 frames. In this way, the boxes are less wobbly and the false positives are mostly removed. The threshold value in this case is 5 which is the one that gave better results.

Following, three video screenshots taken randomly are shown:

![video1][video1]
![video2][video2]
![video3][video3]




---

## 5. Discussion

The main issue has been to properly select the parameters and the features to extract and use. This took a considerable amount of time which made me consider the neural network approach. What's more, the feature extraction took a considerable amount of memory which made my program crash sometimes while training.

The pipeline detects the cars that are coming in the other direction. Although they don't have to be considered as false positives (after all, they are vehicles), the pipeline shouldn't detect them since they are not in the correct direction. A simple solution would be to reduce the sliding window region in order to cover only the lane of our direction. Other solution would be to use images of cars in the opposite direction as examples of the non-vehicle class.
