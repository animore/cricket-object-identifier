import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

def cricket_light_features(img):

    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_img = cv2.resize(gray_full, (64, 48))  # Reduced from (192,144) -> 14k to ~1.2k dims

    hog_feats = hog(
        hog_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # hog_img = cv2.resize(gray_full, (96, 72))
    #
    # hog_feats = hog(
    # hog_img,
    # orientations=6,
    # pixels_per_cell=(16, 16),
    # cells_per_block=(2, 2),
    # block_norm='L2-Hys',
    # feature_vector=True
    # )

    feats = list(hog_feats)
    return feats

