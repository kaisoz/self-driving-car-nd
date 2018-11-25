import pickle
import argparse
import math
import numpy as np

from collections import namedtuple
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Convolution2D, SpatialDropout2D
from keras.models import Sequential, load_model

from csv_reader import AdjustableCsvReader
import image_utils

Parameters = namedtuple("Parameters", [
    "csv_file", "csv_path_adjust",
    "model_checkpoint_format",
    "model_checkpoint_path",
    "history_save_path",

    "batch_size",
    "nr_epoch",
    
    "input_shape",
    "correction",
    "dropout",
])

params = Parameters(
    csv_file= "./example_data/driving_log.csv",
    csv_path_adjust="./example_data/IMG",
    history_save_path="./history/history.sav",
    model_checkpoint_format="model.{epoch:02d}-{val_loss:.2f}",
    model_checkpoint_path=".",

    batch_size=32,
    nr_epoch=20,

    input_shape=(66, 200, 3),
    correction = 0.25,
    dropout=0.2,
)

def plot_history(history):
    """Plot the Mean Squared Error Loss for both the training and validation sets 
    
    :param history: History dictionary obtained from a history object
    :return: 
    """
    print(history)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def save_history(history, path=params.history_save_path):
    """Pickle the history dictionary embedded into a History object.
    
    :param history: History object which contains the history dictionary
    :param path: Where to save the pickled dictionary
    :return: 
    """
    pickle.dump(history.history, open(path, 'wb+'))

def plot_saved_history(history_path):
    """Unpickle a history dictionary and plot it.
    
    :param history_path: Path to the pickled history dictionary
    :return: 
    """
    history = pickle.load(open(history_path, 'rb'))
    plot_history(history)


def read_csv(csv_reader):
    """Read all rows of the csv loaded by the csv_reader. Returns"""

    rows = []
    while True:
        try:
            rows.append(csv_reader.next())
        except Exception as e:
            print(str(e))
            break

    return rows

def load_and_preprocess_image(path):
    """Load an image from a given path and preprocess it."""
    image_center = image_utils.load_image(path)
    return image_utils.preprocess(image_center)

def generator(rows, batch_size=params.batch_size):
    """Generate a preprocessed batch of image examples.
    
        For each row, four images are generated:
            * Center camera image
            * Flipped center camera image
            * Left camera image
            * Right camera image
        
        Therefore, each time this generator is called, it returns (batch_size * 4) images
        All images are preprocessed before being returned the called. Images are:
            1. Cropped to remove the horizon and the hood, ending up with shape 66x320
            2. Resized to meet the input shape requirements for the NVIDIA model (66x200)
            3. Transformed to YUV color space to meet NVIDIA model requirements
    
    :param rows: List of Row objects
    :param batch_size: Size of the batch. This size will be multiplied by 4
    """

    num_samples = len(rows)
    while 1:
        shuffle(rows)
        for offset in range(0, num_samples, batch_size):
            batch_samples = rows[offset:offset + batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:

                image_center = load_and_preprocess_image(batch_sample.img_center)
                steering_center = float(batch_sample.steering)

                image_flipped = np.fliplr(image_center)
                steering_center_flipped = -steering_center

                image_left = load_and_preprocess_image(batch_sample.img_left)
                steering_left = steering_center + params.correction

                image_right = load_and_preprocess_image(batch_sample.img_right)
                steering_right = steering_center - params.correction

                images.extend(  [image_center, image_flipped, image_left, image_right])
                steering.extend([steering_center, steering_center_flipped, steering_left, steering_right])

            x_train = np.array(images)
            y_train = np.array(steering)
            yield shuffle(x_train, y_train)

def split_dataset(dataset):
    """Split the given dataset into the train and validation datasets.
        The training dataset will contain the first 80% of the examples while the validation dataset will contain the remaining 20%
    
    :param dataset: List of row objects to be splitted
    :return: train_samples, validation_samples lists
    """
    train_samples = dataset[:math.floor(len(dataset) * 0.8)]
    validation_samples = dataset[math.floor(len(dataset) * 0.8):]
    return train_samples, validation_samples

def filter_straight_driving_images(rows):
    """Randomly remove those examples which its absolute steering angle is lower than or equal to 0.55 .
    
    :param rows: Row objects to be filterd
    """
    for r in rows:
        if abs(float(r.steering)) <= 0.55 and np.random.random() > 0.75:
            rows.remove(r)

def train_from_model(model, model_checkpoint_suffix="", save_history_path=params.history_save_path):
    """Train the given model. Save the models with smaller validation loss and the complete history
    
    The model includes a ModelCheckpoint callback which saves the models with the smaller validation loss
    
    :param model: Model to be trained
    :param model_checkpoint_suffix: Suffix to add to the saved checkpointed model
    :param save_history_path: Where to save the complete History object
    """
    model_checkpoint_path="{}/{}{}.h5".format(params.model_checkpoint_path, params.model_checkpoint_format, model_checkpoint_suffix)
    csv_reader = AdjustableCsvReader(params.csv_file, params.csv_path_adjust)
    rows = read_csv(csv_reader)

    print("\n-----------------------------------")
    print("* Total number of rows in CSV: {}".format(len(rows)))
    filter_straight_driving_images(rows)

    training_rows, validation_rows =  split_dataset(rows)
    train_samples_per_epoch = len(training_rows) * 4
    validation_samples_per_epoch = len(validation_rows) * 4

    print("* Number of rows after filtering: {}".format(len(rows)))
    print("* Number of training rows: {}".format(len(training_rows)))
    print("* Number of validation rows: {}".format(len(validation_rows)))
    print("* Train samples per epoch: {}".format(train_samples_per_epoch))
    print("* Validation samples per epoch: {}".format(validation_samples_per_epoch))
    print("-----------------------------------\n")

    train_generator = generator(training_rows)
    validation_generator = generator(validation_rows)

    fit_callbacks = [
        ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    ]

    history = model.fit_generator(train_generator,
                                  callbacks = fit_callbacks,
                                  samples_per_epoch = train_samples_per_epoch,
                                  validation_data = validation_generator,
                                  nb_val_samples = validation_samples_per_epoch,
                                  nb_epoch = params.nr_epoch)

    save_history(history, save_history_path)

def create_model():
    """Create and return the NVIDIA model architecture."""

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=params.input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu", init = 'he_normal'))
    model.add(SpatialDropout2D(params.dropout))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu", init = 'he_normal'))
    model.add(SpatialDropout2D(params.dropout))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu", init = 'he_normal'))
    model.add(SpatialDropout2D(params.dropout))
    model.add(Convolution2D(64, 3, 3, activation="elu", init = 'he_normal'))
    model.add(SpatialDropout2D(params.dropout))
    model.add(Convolution2D(64, 3, 3, activation="elu", init = 'he_normal'))
    model.add(SpatialDropout2D(params.dropout))
    model.add(Flatten())
    model.add(Dense(100, activation="elu", init = 'he_normal' ))
    model.add(Dense(50,  activation="elu", init = 'he_normal' ))
    model.add(Dense(10,  activation="elu", init = 'he_normal' ))
    model.add(Dense(1,   activation="elu", init = 'he_normal' ))

    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.summary()
    return model


def train():
    """Create and train the NVIDIA model architecture."""
    print("* Full training *")
    model = create_model()
    train_from_model(model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest="history_path", action='store')
    parser.add_argument('-m', dest="model_path",   action='store')

    args = parser.parse_args()
    if args.history_path:
        plot_saved_history(args.history_path)
    elif args.model_path:
        print("* Training from model: %s *" % (args.model_path))
        model = load_model(args.model_path)
        train_from_model(model, model_checkpoint_suffix=".transfered", save_history_path="{}.transfered".format(params.history_save_path))
    else:
        train()
