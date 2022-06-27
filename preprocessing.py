import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)


def max_gray(img):
    B, G, R = cv2.split(img)
    idx1 = B > G
    result = np.where(idx1, B, G)
    idx2 = result > R
    result = np.where(idx2, result, R)

    return result


def min_gray(img):
    B, G, R = cv2.split(img)
    idx1 = B < G
    result = np.where(idx1, B, G)
    idx2 = result < R
    result = np.where(idx2, result, R)

    return result


def mean_gray(img):
    return np.mean(img, axis=-1)


def gamma_correction(img, gamma=1.6):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)

    return res


def preprocessing(img):
    gray = max_gray(img)
    gray = cv2.dilate(gray, np.ones((1, 4), np.uint8))
    gray = cv2.erode(gray, np.ones((4, 1), np.uint8))
    gray = gamma_correction(gray, 0.8)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=8)

    return gray


if __name__ == "__main__":
    imgs_dir = r'data/test'
    img_path = os.listdir(imgs_dir)
    for i in img_path:
        img_name = os.path.join(imgs_dir, i)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        processed_img = preprocessing(img)

        cv2.imwrite(os.path.join(r'data/test_processed', i), processed_img)
