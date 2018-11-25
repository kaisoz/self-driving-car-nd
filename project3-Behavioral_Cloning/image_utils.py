import cv2
import matplotlib as plt

def load_image(path):
    return cv2.imread(path)

def show_image(image, label="image"):
    cv2.imshow(label, image)
    cv2.waitKey(0)

def save_image(image, label="image"):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.imwrite(label, image)

def preprocess(image, image_is_rgb = False, crop_x1 = 69, crop_x2 = 135):
    color_space_code = cv2.COLOR_RGB2YUV if image_is_rgb is True else cv2.COLOR_BGR2YUV
    image = image[crop_x1:crop_x2, :]
    image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, color_space_code)

