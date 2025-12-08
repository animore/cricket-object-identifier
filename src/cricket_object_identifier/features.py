import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

def cricket_light_features_522(img):
    # Input is 800x600
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------
    # 1. Resize image for lighter HOG (~1200 dims)
    # ----------------------------------------------
    # hog_img = cv2.resize(gray_full, (192, 144))

    # hog_feats = hog(
    #     hog_img,
    #     orientations=9,
    #     pixels_per_cell=(8, 8),
    #     cells_per_block=(2, 2),
    #     block_norm='L2-Hys',
    #     feature_vector=True
    # )

    hog_img = cv2.resize(gray_full, (96, 72))

    hog_feats = hog(
    hog_img,
    orientations=6,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    feature_vector=True
    )


    # ----------------------------------------------
    # 2. Color histograms (HSV + LAB)
    # ----------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    feats = list(hog_feats)

    for col_img in [hsv, lab]:
        for i in range(3):
            hist = cv2.calcHist([col_img], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist)

    # ----------------------------------------------
    # 3. LBP texture
    # ----------------------------------------------
    lbp_img = cv2.resize(gray_full, (256, 256))
    lbp = local_binary_pattern(lbp_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    feats.extend(lbp_hist)

    # ----------------------------------------------
    # 4. Hu Moments (shape)
    # ----------------------------------------------
    _, th = cv2.threshold(gray_full, 0, 255, cv2.THRESH_OTSU)
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments).flatten()
    feats.extend(np.log(np.abs(hu) + 1e-8))

    return feats

# def cricket_light_features_2000(img):
#     gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # -------------------------------------------------------
#     # 1. HOG tuned to produce ~1872 dims
#     # -------------------------------------------------------
#     hog_img = cv2.resize(gray_full, (112, 84))  # tuned size

#     hog_feats = hog(
#         hog_img,
#         orientations=4,                    # reduced dimensions
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         block_norm='L2-Hys',
#         feature_vector=True
#     )

#     feats = list(hog_feats)

#     # # -------------------------------------------------------
#     # # 2. Color histograms (HSV + LAB) → 96 dims
#     # # -------------------------------------------------------
#     # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     #
#     # for col_img in [hsv, lab]:
#     #     for i in range(3):
#     #         hist = cv2.calcHist([col_img], [i], None, [16], [0, 256])
#     #         hist = cv2.normalize(hist, hist).flatten()
#     #         feats.extend(hist)
#     #
#     # # -------------------------------------------------------
#     # # 3. LBP uniform → 59 dims
#     # # -------------------------------------------------------
#     # lbp_img = cv2.resize(gray_full, (256, 256))
#     # lbp = local_binary_pattern(lbp_img, P=8, R=1, method='uniform')
#     # lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
#     # feats.extend(lbp_hist)
#     #
#     # # -------------------------------------------------------
#     # # 4. Hu Moments → 7 dims
#     # # -------------------------------------------------------
#     # _, th = cv2.threshold(gray_full, 0, 255, cv2.THRESH_OTSU)
#     # moments = cv2.moments(th)
#     # hu = cv2.HuMoments(moments).flatten()
#     # feats.extend(np.log(np.abs(hu) + 1e-8))

#     return feats

def cricket_light_features_2000(img):
    """Extract ~1872 features: HOG + Color Histograms + LBP + Hu Moments."""
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------
    # 1. HOG → ~90 dims
    # -------------------------------------------------------
    hog_img = cv2.resize(gray_full, (112, 84))
    hog_feats = hog(
        hog_img,
        orientations=4,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    feats = list(hog_feats)

    # -------------------------------------------------------
    # 2. Color histograms (HSV + LAB + BGR) → ~288 dims (96 × 3)
    # -------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bgr = img

    for col_img in [hsv, lab, bgr]:
        for i in range(3):
            hist = cv2.calcHist([col_img], [i], None, [32], [0, 256])  # 32 bins per channel
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist)

    # -------------------------------------------------------
    # 3. LBP uniform → ~59 dims
    # -------------------------------------------------------
    lbp_img = cv2.resize(gray_full, (256, 256))
    lbp = local_binary_pattern(lbp_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    feats.extend(lbp_hist)

    # -------------------------------------------------------
    # 4. GLCM Texture Features → 4 dims
    # -------------------------------------------------------
    glcm = graycomatrix(gray_full, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        feats.append(graycoprops(glcm, prop)[0, 0])

    # -------------------------------------------------------
    # 5. Hu Moments (shape) → 7 dims
    # -------------------------------------------------------
    _, th = cv2.threshold(gray_full, 0, 255, cv2.THRESH_OTSU)
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments).flatten()
    feats.extend(np.log(np.abs(hu) + 1e-8))

    # -------------------------------------------------------
    # 6. Edge Statistics (Sobel) → ~20 dims
    # -------------------------------------------------------
    sobel_x = cv2.Sobel(gray_full, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray_full, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    feats.extend([
        np.mean(sobel), np.std(sobel), np.min(sobel), np.max(sobel),
        np.percentile(sobel, 25), np.percentile(sobel, 50), np.percentile(sobel, 75)
    ])

    # -------------------------------------------------------
    # 7. Local Statistics (mean/std in regions) → ~1300+ dims
    # -------------------------------------------------------
    # Divide image into grid and compute stats
    grid_h, grid_w = 16, 16  # 16×16 = 256 regions
    region_h, region_w = gray_full.shape[0] // grid_h, gray_full.shape[1] // grid_w
    
    for i in range(grid_h):
        for j in range(grid_w):
            r1, r2 = i * region_h, (i + 1) * region_h
            c1, c2 = j * region_w, (j + 1) * region_w
            region = gray_full[r1:r2, c1:c2].astype(np.float32) / 255.0
            
            feats.extend([
                np.mean(region), np.std(region), np.min(region), np.max(region),
                np.var(region)
            ])

    print(f"✅ cricket_light_features_2000 generated {len(feats)} features")
    return feats

def extract_features_135(cell_img):
    # cell_img is a cropped numpy array (H×W×3), BGR because OpenCV
    img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    feats = []

    # 1. 8-bin RGB histogram (c1-c24)
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [8], [0, 256])
        hist = hist.flatten()
        hist /= hist.sum() + 1e-6          # normalize
        feats.extend(hist)

    # 2. HOG 1 – 9 orientations, 2×2 blocks → 36 values (c25-c60)
    hog1 = hog(img, orientations=9, pixels_per_cell=(w//2, h//2),
               cells_per_block=(1,1), block_norm='L2-Hys', channel_axis=-1, feature_vector=True)
    feats.extend(hog1)

    # 3. HOG 2 on grayscale – another 36 values (c61-c96)
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    hog2 = hog(gray, orientations=9, pixels_per_cell=(w//2, h//2),
               cells_per_block=(1,1), block_norm='L2-Hys', feature_vector=True)
    feats.extend(hog2)

    # 4. 8-bin LAB histogram (c97-c120) – very common second color space
    lab = cv2.cvtColor(cell_img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [8], [0, 256])
        hist = hist.flatten()
        hist /= hist.sum() + 1e-6
        feats.extend(hist)

    # 5. Simple statistics (c121-c135) – 15 values, but some datasets use 11
    gray_flat = gray.astype(np.float32) / 255.0
    feats.extend([
        np.mean(gray_flat),                     # c121
        np.std(gray_flat),                      # c122
        np.var(gray_flat),                      # c123 (sometimes skipped)
        np.max(gray_flat),                      # c124
        np.min(gray_flat),                      # c125
        # entropy (sometimes included)
        # skewness, kurtosis, etc.
        # many implementations just pad with zeros or repeat mean/std
    ])

    # Pad or truncate to exactly 135
    if len(feats) > 135:
        feats = feats[:135]
    elif len(feats) < 135:
        feats.extend([0.] * (135 - len(feats)))

    return feats

def extract_features_46(cell):
    features = []

    # Convert to grayscale
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # 1. Color mean and std
    means = cell.mean(axis=(0,1))
    stds = cell.std(axis=(0,1))
    features.extend(means)
    features.extend(stds)

    # 2. Color histogram (8 bins × 3 channels = 24 features)
    for ch in range(3):
        hist = cv2.calcHist([cell], [ch], None, [8], [0,256]).flatten()
        features.extend(hist)

    # 3. LBP histogram (Local Binary Pattern → 10 bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    features.extend(lbp_hist)

    # 4. GLCM Texture features (Contrast, Homogeneity, Energy, Correlation)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        features.append(graycoprops(glcm, prop)[0,0])

    # 5. Edge features (Sobel magnitude statistics)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    features.append(np.mean(sobel))
    features.append(np.std(sobel))

    return features