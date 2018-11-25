# Project 2 - Traffic Sign Recognition

Tomás Tormo Franco (tomas.tormo@gmail.com)


[//]: # (Image References)

[image1]: ./images/classes_chart.png "Visualization"
[image2]: ./images/original.png  "Original"
[image3]: ./images/preprocessed.png  "Preprocessed"
[image4]: ./images/new_images.png  "New Images"



## Data Set Summary & Exploration

In order to calculate the summmary statistic of the traffic signs data set I used the numpy library, since the datasets are repesented as multidimensional arrays whose features represent the requested data.

The size of a dataset is the size of its first dimension and the number of classes is the number of different values the labels array contains. The shape of a traffic sign image is the shape of the array which contains its values.

The code for this step is contained in the second code cell of the IPython notebook. 

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


### 1. Exploratory visualization

Here is an exploratory visualization of the data set. First, a bar chart showing the number of examples per class is presented:

![alt text][image1]

As can be seen from the char, the dataset is very unbalanced, since some classes (like class 1) are much more represented than others (for example, class 0).  The IPython notebook also shows three images of each class as an example. 

The code for this visualization is contained in the third code cell of the IPython notebook. 


## Design and Test a Model Architecture

### 1. Image data preprocess

First, the images are converted to grayscale since the color is not an important feature for this problem and the less features, the easier for the model.

Second, the image features are scaled to the [0.1] range to reduce the magnitudes in order to help the optimization algorithm. 
Third, an histogram equalization is applied to each image to enhance contrasts and improve feature extraction.

Finally, the labels are one-hot encoded and the dataset is suffled.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2] ![alt text][image3]


The code for the images preprocess is contained in the fourth code cell of the IPython notebook.


### 2. Model architecture


My model is based on the LeNet architecture as it works well for this kind of problem. The model includes a couple of dropout layers just after the first two Fully connected layers in order to introduce regularization to avoid overfitting and improve generalization.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		 | 32x32x1 grayscale image   					| 
| Convolution 5x5      | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			 |										|
| Max pooling	      	 | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5      | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU			 |										|
| Max pooling	      	 | 2x2 stride,  outputs 5x5x16 				|
| FullyConnected       | input 400, output120					 	|
| RELU			 |										|
| Dropout			 | keep probability for training = 0.5			|
| FullyConnected       | input = 120, output = 84					|
| RELU			 |										|
| Dropout			 | keep probability for training = 0.5			|
| FullyConnected       | input = 84, output = 43					|


The code of the model is contained in the fifth, sixth and seventh code cells of the IPython notebook. 


### 3. Training

For weight initialization, I used a mean of 0 and a sigma of 0.1 so I assure small orders of magnitude at initialization.

To train the model, I used an Adam Optimizer which a adaptively selects a separate learning rate for each parameter. In this way, the manual parameter tuning can be avoided since it automatically figures out how much to step and adjust the learning rate. In order to fine tune the hyperparameters, I started with a small subset of the examples (20%) and made it overfit. Once the model overfitted, I started modifying the hyperparameters and introducing regularization techniques to improve generalization.

In the final model, I kept the batch size in 50 examples but increased the number of epoch up to 70, since the more epoch, the more backpropagations the model does and therefore the more it learns. This values along with the dropout regularization technique improved the performance up to the current values.

The code for training the model is located in the ninth and tenth cells of the ipython notebook. 

### 4. Approach


My final model results were:

* Training set accuracy of 99%
* Validation set accuracy of 97.1% 
* Test set accuracy of 95.4%

The architecture I chose was the LeNet-5 architecture with subtle regularization modifications. I chose this model because it's a simple well known architecture that performs well for simple shapes detection. 

It can be seen that the model is neither overfitting nor underfitting: it gets high accuracy values for both the training and validation sets and the difference between them is less than a 2%. Also, since the accuracy in the test set is also high and really close to the validation and test values, we can conclude that the model is working well.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

## Test a Model on New Images

### 1. New images

Here are five German traffic signs that I found on the web:

![alt text][image4]


The easiest images to classify are the "Yield" and the "Keep right" signs, as there are lots of examples available for them in the test set (around 2000 examples). The model has less of 500 examples for the other signs, so they are expected to be more difficult to classify. 


### 2. Results


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout Mandatory     		| Roundabout Mandatory  	| 
| Yield     						| Yield 					|
| Speed Limit (20km/h)			| Speed Limit (60km/h)		|
| Keep right	      				| Keep right				|
| Children Crossing				| Children Crossing      		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Contrary to what was expected, the majority of the signs with low number of examples have been correctly detected.

The only one that the model couldn't properly classify has been the speed limit sign. This behavior is expected since the model doesn't have enough examples of this sign to learn to differentiate it from the other speed limits signs.

The code for this results is located in the twelfth and thirteenth cells of the Ipython notebook.


### 3. Model prediction

The code for making predictions on my final model is located in the fourteenth cell of the Ipython notebook.

For the **first** image, the model is very sure that this is a "Roundabout mandatory" sign (probability of 94%), and the image does contain a stop sign. The top five soft max probabilities were:


| Probability         	|     Prediction	        				| 
|:---------------------:|:---------------------------------------------:| 
| .944         	 	 | Roundabout mandatory   			| 
| .029     			 | Priority road 					|
| .018			 | Right-of-way at the next intersection	|
| .004	      		 | Speed limit (100km/h)				|
| .001		 	 | Go straight or left      				|


In the case of the  **second** image, the model is is absolutely sure that this is a "Yield" sign, since it gives is a probability of the 100%:

| Probability         	|     Prediction	        				| 
|:---------------------:|:---------------------------------------------:| 
| 1         	 	 | Yield   						| 
| 0     		 | Priority road 				|
| 0			 | Ahead only					|
| 0	      		 | Turn left ahead				|
| 0		 	 | Speed limit (20km/h)      		|


The case of the  **third** image is similar to the second one, the model is is absolutely sure that this is a "Keep right" sign, since it gives is a probability of the 100%:

| Probability         	|     Prediction	        				| 
|:---------------------:|:---------------------------------------------:| 
| 1         	 	 | Keep right   						| 
| 0     		 | Yield 							|
| 0			 | Turn left ahead					|
| 0	      		 | No vehicles					 	|
| 0		 	 | Bumpy road      					|


The model fails predicting the **fourth** image. It is quite certain that this is a "Speed limit (60km/h)" sign but it is a "Speed limit (20km/h)" sign. At least, the model has taken it into account altough it has give it a very low probability:

| Probability         	|     Prediction	        				| 
|:---------------------:|:---------------------------------------------:| 
| .810         	 	| Speed limit (60km/h)   			| 
| .176     		 	| Speed limit (80km/h) 			|
| .008			| Bicycles crossing				|
| .002	      		| Speed limit (20km/h)			|
| .000		 	| Speed limit (120km/h)      		|

The **fifth** and last one image is being correctly predicted by the model. It gives a 99% probability to the "Children crossing" sign which is the one that the image contains:

| Probability         	|     Prediction	        				| 
|:---------------------:|:---------------------------------------------:| 
| .999         	 	| Children crossing   			| 
| .000     		 	| Dangerous curve to the right 	|
| .000			| Bicycles crossing				|
| .000	      		| Road work					|
| .000		 	| Road narrows on the right      	|
