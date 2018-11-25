import glob
import argparse
import cv2
import math
import numpy as np
import matplotlib.image as mpimpg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from collections import deque
from feature_extractor import Extractor

extractor = None
classifier = None
scaler = None
heatmaps = deque(maxlen=10)


def load_image(images_list):
    for img_path in images_list:
        yield mpimpg.imread(img_path)


def extract_features_from_images_in_path(path, extractor):
    features = []
    loader = load_image(glob.glob(path))
    for img in loader:
        img_features = extractor.from_image(img)
        features.append(img_features)
    return features


def split_dataset(x, y):
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=rand_state)

    return x_train, y_train, x_test, y_test


def train_classifier(extractor):

    print("* Extracting vehicle features...")
    car_features = extract_features_from_images_in_path("dataset/vehicles/*/*.png", extractor)
    print("* Done. Read {} car vehicle".format(len(car_features)))

    print("\n* Extracting non-vehicle features...")
    noncar_features = extract_features_from_images_in_path("dataset/non-vehicles/*/*.png", extractor)
    print("* Done. Read {} non-vehicle images".format(len(noncar_features)))

    # Normalize features and create labels
    x = np.vstack((car_features, noncar_features)).astype(np.float64)

    clf = LinearSVC()
    x_scaler = StandardScaler().fit(x)
    x = x_scaler.transform(x)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    print("\n* Have {} samples".format(len(y)))
    print("* Have {} features".format(x.shape[1]))

    x, y = shuffle(x, y)
    x_train, y_train, x_test, y_test = split_dataset(x, y)

    print("\n* Trainig...")
    clf.fit(x_train, y_train)
    print("* Done")

    print('\n*Test Accuracy of SVC = ', round(clf.score(x_test, y_test), 4))
    return clf, x_scaler


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, extractor, classifier, scaler, ystart, ystop, scale):

    bboxes = []
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = extractor.change_color_space(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // extractor.hog_pix_per_cell) - extractor.hog_cells_per_block + 1
    nyblocks = (ch1.shape[0] // extractor.hog_pix_per_cell) - extractor.hog_cells_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // extractor.hog_pix_per_cell) - extractor.hog_cells_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = extractor.get_hog_features(ch1, feature_vector=False)
    hog2 = extractor.get_hog_features(ch2, feature_vector=False)
    hog3 = extractor.get_hog_features(ch3, feature_vector=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * extractor.hog_pix_per_cell
            ytop = ypos * extractor.hog_pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = extractor.bin_spatial(subimg)
            hist_features = extractor.color_hist(subimg)

            # Scale features and make a prediction
            test_features = scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = classifier.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append([(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)])

    return bboxes


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


def process_image(img, return_heatmap=False, accumulate_heatmaps=True):

    threshold = 1
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    bboxes = find_cars(img, extractor, classifier, scaler,
                       ystart=400, ystop=656, scale=1.5)

    frame_heatmap = add_heat(heat, bboxes)
    if accumulate_heatmaps is True:
        threshold = 5
        heatmaps.append(frame_heatmap)
        frame_heatmap = sum(heatmaps)

    frame_heatmap = apply_threshold(frame_heatmap, threshold)
    labels = label(frame_heatmap)
    bboxes = draw_labeled_bboxes(np.copy(img), labels)

    if return_heatmap:
        frame_heatmap = np.clip(frame_heatmap, 0, 255)
        return bboxes, frame_heatmap
    else:
        return bboxes


def process_video(video_input, video_output):
    print("* Processing video %s to %s" % (video_input, video_output))
    input_video_clip = VideoFileClip(video_input)
    clip = input_video_clip.fl_image(process_image)
    clip.write_videofile(video_output, audio=False)
    print("* Done")

if __name__ == "__main__":

    extractor = Extractor(color_space = 'YCrCb',
                          spatial_size=(16,16),
                          hist_bins=16,
                          hog_orient=9,
                          hog_channel='ALL',
                          hog_pix_per_cell=8,
                          hog_cell_per_block=2)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true", dest="process_video", default=False)
    parser.add_argument("--input", action="store", dest="src_video", default="project_video.mp4")
    args = parser.parse_args()

    classifier, scaler = train_classifier(extractor)

    if args.process_video:
        src_video = args.src_video
        dst_video = "{}_detected_vehicles.mp4".format(src_video.split('.')[-2])
        process_video(src_video, dst_video)
    else:

        for i in glob.glob("test_images/*.jpg"):
            img = mpimpg.imread(i)
            draw_img, heatmap = process_image(img, return_heatmap=True, accumulate_heatmaps=False)
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
            ax1.imshow(draw_img)
            ax1.set_title("Labeled image")
            ax2.imshow(heatmap, cmap='gray')
            ax2.set_title("Heatmap")
            plt.show()
