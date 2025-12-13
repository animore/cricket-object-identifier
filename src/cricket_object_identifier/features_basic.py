import cv2
from skimage import io, color
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

    feats = list(hog_feats)
    return feats


def _hsv_histograms(img, bins=32):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist])
    hist = hist / (np.sum(hist) + 1e-8)
    return hist.tolist()


def _edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return [float(np.count_nonzero(edges)) / float(edges.size)]


def cricket_light_features_v2(img):
    # HOG from a slightly larger resize to capture bat/ball shapes better
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_img = cv2.resize(gray_full, (96, 72))
    hog_feats = hog(
        hog_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    color_feats = _hsv_histograms(img, bins=32)  # 96 dims
    edge_feat = _edge_density(img)  # 1 dim

    feats = list(hog_feats) + color_feats + edge_feat
    return feats

def cricket_light_features_v3(img):
    # HOG from a slightly larger resize to capture bat/ball shapes better
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray_full.mean()
    std = gray_full.std()
    hog_img = cv2.resize(gray_full, (96, 72))
    hog_feats = hog(
        hog_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # color_feats = _hsv_histograms(img, bins=32)  # 96 dims
    # edge_feat = _edge_density(img)  # 1 dim
    # feats = list(hog_feats) + color_feats + edge_feat
    feats = list(hog_feats)
    feats.append(mean)
    feats.append(std)
    return feats

def cricket_light_features_p(cell_img):
    gray = color.rgb2gray(cell_img)
    hog_feat = hog(gray, pixels_per_cell=(8,8), cells_per_block=(1,1), feature_vector=True)
    mean = gray.mean()
    std = gray.std()
    return list(np.concatenate([hog_feat, [mean, std]]))

def cricket_light_features_v4(img):
    """
    Works extremely well for:
    - Stumps (white, tall, strong edges)
    - Bats   (long, wood color, medium edges)
    - Balls  (red OR white OR pink, small, round, high saturation + circularity)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1. HOG on moderate resize (good for all objects) ─────────────────────
    hog_img = cv2.resize(gray, (64, 64))          # 64×64 works perfectly with 16×16 grid
    hog_feats = hog(
        hog_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # ── 2. HSV statistics (very powerful) ───────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [72], [0, 180]).flatten()   # 72 bins
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()   # saturation matters a lot
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    # Normalize
    h_hist /= (h_hist.sum() + 1e-8)
    s_hist /= (s_hist.sum() + 1e-8)
    v_hist /= (v_hist.sum() + 1e-8)

    # ── 3. Saturation & Brightness stats (balls are usually high saturation) ─
    mean_s = hsv[:, :, 1].mean() / 255.0
    mean_v = hsv[:, :, 2].mean() / 255.0
    std_s  = hsv[:, :, 1].std()  / 255.0

    # ── 4. Edge density ───────────────────────────────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # ── 5. Circularity / Compactness (the real ball killer feature) ───────────
    # Very cheap but incredibly effective for round objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity_score = 0.0
    if len(contours) > 0:
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 20:  # ignore tiny noise
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity_score = 4 * np.pi * area / (perimeter * perimeter)  # 1.0 = perfect circle

    # ── 6. Final feature vector ───────────────────────────────────────────────
    feats = (
        list(hog_feats) +
        h_hist.tolist() +
        s_hist.tolist() +
        v_hist.tolist() +
        [mean_s, mean_v, std_s, edge_density, circularity_score]
    )
    return feats

def get_feature_length(fn):
    # Roughly estimate feature vector length by running on a dummy image
    dummy = np.zeros((72, 96, 3), dtype=np.uint8)
    return len(fn(dummy))


# ---------------------------------------------------------
# NEW: Data Augmentation Function (Color Jitter)
# ---------------------------------------------------------
def apply_color_jitter(img):
    """
    Randomly adjusts Hue, Saturation, and Value (Brightness).
    This forces the model to ignore specific colors (like blue jerseys)
    and focus on HOG/Shape features.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1. Random Brightness (Value)
    value_noise = np.random.randint(-30, 30)
    v = cv2.add(v, value_noise)

    # 2. Random Saturation
    sat_noise = np.random.randint(-30, 30)
    s = cv2.add(s, sat_noise)

    # 3. Random Hue (Color Shift) - CRITICAL for fixing the "Blue Jersey" bug
    hue_noise = np.random.randint(-10, 10)  # Shift colors slightly
    h = cv2.add(h, hue_noise)

    final_hsv = cv2.merge((h, s, v))
    img_jittered = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_jittered

def apply_color_jitter_updated(img, apply_probability=0.7):
    """
    Apply strong color jitter with 70% probability during training only.
    This makes the model completely color-blind and forces it to use shape.
    """
    if np.random.rand() > apply_probability:
        return img  # sometimes return original (helps stability)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 1. Hue shift – very aggressive now (can turn red ball → green → blue)
    hue_shift = np.random.randint(-25, 26)        # ±25 is enough to destroy color cues
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    # 2. Saturation multiplier (can make ball almost gray)
    sat_mult = np.random.uniform(0.4, 2.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0, 255)

    # 3. Value (brightness) multiplier
    val_mult = np.random.uniform(0.6, 1.6)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_mult, 0, 255)

    # 4. Random gamma (very powerful)
    gamma = np.random.uniform(0.7, 1.5)
    hsv[..., 2] = np.power(hsv[..., 2]/255.0, gamma) * 255.0

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

