# Project 3 - Behavioral Cloning
Tomás Tormo Franco (tomas.tormo@gmail.com)


[center_processed]: ./images/center_processed.png "Center Processed"
[center_flipped_yuv]: ./images/YUV_flipped.png "Center flipped YUV"
[left_processed]: ./images/right_processed.png "Left Processed"
[right_processed]: ./images/left_processed.png "Right Processed"
[nvidia_cnn]: ./images/cnn-nvidia.png "NVIDIA Model Architecture"
[model_mse_loss]: ./images/model_mse_loss.png "Model MSE loss"


## Overview

In this project, a trained car drives in a simulated environment by cloning the behavior as seen during training mode.  Leveraging TensorFlow and Keras, a deep learning network predicts the proper steering angle given training examples.

## Files Submitted
* `model.py` - The script used to create and train the model.
* `drive.py`   - The script to drive the car in autonomous mode.
* `model.h5`-  Trained convolutional neural network.


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Model Architecture

The model I used is the NVIDIA model depicted in their paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)
I chose this model since it's a proven model designed specifically for this problem. 

![NVIDIA Model Architecture][nvidia_cnn]



The network is divided in a normalization layer and two sets of convolutional and fully connected layers. It has about 27 million connections and 250 thousand parameters.

Having a normalization layer right after the input layer assures that the images will have the proper values also in prediction phase. 

The normalization layer is followed by a set of five convolutional layers: the first three use a 2×2 stride and a 5×5 kernel, and the last two use a 3×3 kernel size and no stride at all. The result of the convolutional layers is flattened and passed to the last set of layers, which consists of three fully connected layers of 100, 50 and 10 neurons respectively. 

In order to reduce overfitting I added a SpatialDropout2D layer after each convolutional layer. Since the images have a strong spatial correlation, the feature map activations are also strongly correlated. In this setting, the regular dropout will not regularize the activations and results in an learning rate decrease. The spatial dropout, instead, drops entire 2D feature maps instead of individual elements resulting in an effective regularization. 
For full details, please see the original paper: [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

All layers include ELU activations to introduce nonlinearity.  According to the original paper ([Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)), ELU activations speeds up learning and offers the same protection against vanishing gradients as RELU. Besides, ELUs have negative values, which allows them to push the mean activations closer to zero, improving the efficiency of gradient descent.

To train the model, I used an Adam Optimizer which a adaptively selects a separate learning rate for each parameter. In this way, the manual parameter tuning can be avoided since it automatically figures out how much to step and adjust the learning rate.

The implementation of the network in located in the function **create_model** of the file **model.py**. The final model architecture shown by the Keras model summary is as follows:

```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo (None, 31, 98, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       spatialdropout2d_1[0][0]
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo (None, 14, 47, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       spatialdropout2d_2[0][0]
____________________________________________________________________________________________________
spatialdropout2d_3 (SpatialDropo (None, 5, 22, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       spatialdropout2d_3[0][0]
____________________________________________________________________________________________________
spatialdropout2d_4 (SpatialDropo (None, 3, 20, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       spatialdropout2d_4[0][0]
____________________________________________________________________________________________________
spatialdropout2d_5 (SpatialDropo (None, 1, 18, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           spatialdropout2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
____________________________________________________________________________________________________

```

## Training Strategy

I finally only used the example data provided with the project because the simulator was working really slow on my computer and I couldn't drive the car reliably enough to get quality data.

My first approach has been to to overfit the model using an small subset of examples (around 2000), to make sure that it works properly. Once overfitted, included dropout layers and tuning parameters to improve generalization.

In order to ease the manipulation of the CSV, I created an *AdjustableCsvReader* class which is a wrapper around the python csv reader. This class reads the csv and transforms each line into a *Row* class that contains fields with the value of each field. In this way, it's much friendly to access to a csv field using a attribute name such as *img_center* than using an index.

To improve memory usage, I used a generator function to feed the model during the training phase. This generator receives a list of Rows and a batch size. For each Row, it loads and preprocess the center, left and right camera images and performs some data augmentation. For details about how I created the training data, please see the next section.

Most of the images of the dataset are from straight lanes and left turns. This made the model learn how to drive straight and turn left overall. 
In order to help the model to learn how to turn right, I flipped all center camera images. This creates right turn images in the same proportion as left turn images.
To reduce the straight driving bias, I filtered the straight lane images prior training the model. To do this, I randomly removed all Rows which contained images which steering angle were less or equal than 0.55 in absolute value. With a higher value the car tended to go to the sides, which made me think that there were not enough straight lane images. This value introduced enough straight lane images without influencing too much the model.

To get the model with the less validation loss, I used a ModelCheckpoint callback to monitor the validation loss and save the models with the best performances.
I trained the model with 32 batch size up to 20 epoch and I kept the model with the less validation loss, which happened at epoch 15 according to Keras output.

Next, the Model Mean Squared Error Loss is shown:

![Model MSE loss][model_mse_loss]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



## Creation of the Training and Validation Sets

As stated before, due the poor simulator performance I had to use the example data provided with the project. I used the three camera images and a flipped the center camera images to get right turns.

In order to create the training and validation sets, I splitted the number of rows, giving 80% for the training set, and 20% for the validation set. The implementation of the split is located in the function **split_dataset** of the **model.py**

For each image, first I cropped 25 pixels from the top to remove the horizon, and 60 pixels from the bottom, to remove the car hood. Once cropped, I resized the image to the shape expected by the NVIDIA model (66x200x3) and transformed it to the YUV color space. For the side cameras, after several tests, I decided to apply a correction of 0.25 to the steering angle. 

This preprocessing is applied to images of the training dataset, validation dataset and also to the images used for steering angle prediction in the **drive.py** file.

The following images show the result of the preprocess for each camera. Images are shown in the following order: original image, cropped image, resized image and the YUV image. The flipped image is generated directly from the YUV image of the center camera.

**Center camera**
![Center Preprocessed][center_processed]

**Center camera flipped**

![Center flipped YUV][center_flipped_yuv]

**Left camera**
![Left Preprocessed][left_processed]

**Right camera**
![Right Preprocessed][right_processed]


After the image filtering and the image generation, I got the following numbers. Please, take into account that the number of rows after filtering may differ between trainings, since the number of images filtered is chosen randomly. This also affects to the number of training/validation rows and the train/validation samples per epoch, since they are derived from the number of rows after filtering:

-----------------------------------
* Total number of rows in CSV: 8036
* Number of rows after filtering: 6435
* Number of training rows: 5148
* Number of validation rows: 1287
* Train samples per epoch: 20592
* Validation samples per epoch: 5148

-----------------------------------
