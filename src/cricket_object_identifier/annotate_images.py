import os
import cv2
import numpy as np
import pickle
import pandas as pd
from features_basic import cricket_light_features

# Inverse label map for displaying predictions
INVERSE_LABEL_MAP = {0: "Bat", 1: "Ball", 2: "Stump", 3: "Background"}

# Colors for different labels (BGR format for OpenCV)
LABEL_COLORS = {
    0: (0, 255, 0),  # Green for Bat
    1: (0, 0, 255),  # Red for Ball
    2: (255, 0, 0),  # Blue for Stump
    3: (255, 255, 255)  # White for Background (not used)
}


def load_model_and_features(model_path, feat_cols_path):
    """
    Load the trained model and feature column names from pickle files.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_cols_path, "rb") as f:
        feat_cols = pickle.load(f)
    return model, feat_cols


def process_image_for_prediction(image_path, grid_size=8):
    """
    Divide the image into grid cells and extract features for each cell.
    Similar to process_image_to_cells in utils_basic.py, but adapted for prediction.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    cell_info = []
    features = []

    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            cell_img = img[y1:y2, x1:x2]
            feats = cricket_light_features(cell_img)

            cell_number = r * grid_size + c
            cell_info.append((cell_number, r, c, (x1, y1, x2, y2)))
            features.append(feats)

    return img, cell_info, features


def predict_objects(model, feat_cols, features):
    """
    Create a DataFrame from features and predict labels using the model.
    """
    feature_df = pd.DataFrame(features, columns=feat_cols)
    predictions = model.predict(feature_df)
    return predictions


def annotate_image(img, cell_info, predictions, thickness=2, font_scale=0.5):
    """
    Draw rectangles and labels on the image for non-background predictions.
    """
    annotated_img = img.copy()
    for (cell_number, r, c, (x1, y1, x2, y2)), pred in zip(cell_info, predictions):
        if pred != 3:  # Skip background
            label_name = INVERSE_LABEL_MAP.get(pred, "Unknown")
            color = LABEL_COLORS.get(pred, (0, 255, 255))  # Default to yellow if unknown

            # Draw rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

            # Put text
            text_pos = (x1, y1 - 10 if y1 - 10 > 0 else y1 + 20)
            cv2.putText(annotated_img, label_name, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return annotated_img


def predict_and_tag_images(test_folder, output_folder, model_path, feat_cols_path):
    """
    Process all images in the test folder, predict objects, annotate, and save to output folder.

    Args:
    - test_folder (str): Path to the folder containing test images.
    - output_folder (str): Path to save annotated images.
    - model_path (str): Path to the pickled model file.
    - feat_cols_path (str): Path to the pickled feature columns file.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load model and features
    model, feat_cols = load_model_and_features(model_path, feat_cols_path)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Process each image in the test folder
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(test_folder, filename)
            try:
                # Process image and extract features
                img, cell_info, features = process_image_for_prediction(image_path)

                # Predict
                predictions = predict_objects(model, feat_cols, features)

                # Annotate
                annotated_img = annotate_image(img, cell_info, predictions)

                # Save annotated image
                output_path = os.path.join(output_folder, f"annotated_{filename}")
                cv2.imwrite(output_path, annotated_img)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage (replace with your paths)
if __name__ == "__main__":
    test_folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\predicition_images"
    output_folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\annotated_outputs"
    model_path = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\cricket_cell_model_stump.pkl"
    feat_cols_path = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\feature_columns_stump.pkl"
#
    predict_and_tag_images(test_folder, output_folder, model_path, feat_cols_path)