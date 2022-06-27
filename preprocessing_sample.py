import cv2
import numpy as np
import preprocessing

if __name__ == "__main__":
    sample = cv2.imread(r'data/train1/xqlL_1572574159.jpg', cv2.IMREAD_COLOR)
    sample = preprocessing.max_gray(sample)
    # cv2.imwrite('figures/sample_max_gray.jpg', sample)
    # sample = cv2.dilate(sample, np.ones((1, 4), np.uint8))
    # cv2.imwrite('figures/sample_dilate.jpg', sample)
    # sample = cv2.erode(sample, np.ones((4, 1), np.uint8))
    sample = cv2.dilate(sample, np.ones((4, 1), np.uint8))
    sample = cv2.erode(sample, np.ones((1, 4), np.uint8))
    # cv2.imwrite('figures/sample_erode.jpg', sample)
    sample = preprocessing.gamma_correction(sample, 0.8)
    # cv2.imwrite('figures/sample_gamma_correction.jpg', sample)
    sample = cv2.convertScaleAbs(sample, alpha=1.2, beta=8)
    # cv2.imwrite('figures/sample_convert.jpg', sample)
    cv2.imwrite('figures/well.jpg', sample)


    # convert_grayscale = cv2.imread(r'data/train1/7FB7_1572574163.jpg', cv2.IMREAD_COLOR)
    # gray = preprocessing.max_gray(convert_grayscale)
    # cv2.imwrite('figures/convert_max_gray.jpg', gray)
    # gray = preprocessing.min_gray(convert_grayscale)
    # cv2.imwrite('figures/convert_min_gray.jpg', gray)
    # gray = preprocessing.mean_gray(convert_grayscale)
    # cv2.imwrite('figures/convert_mean_gray.jpg', gray)

