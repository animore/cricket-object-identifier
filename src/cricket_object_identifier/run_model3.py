from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from features import cricket_light_features_522

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import cv2
import numpy as np

# feature_csv = generate_feature_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\transformed_sample_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\features_sample_tagged_data_min.csv", label_map)

feature_csv = 'C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\features_sample_tagged_data_min.csv'

# 1) Load your features CSV
# csv_path = "features_train.csv"   # change to your file
csv_path = feature_csv   # change to your file
df = pd.read_csv(csv_path)

# 2) Define feature columns and target
# Assuming columns: image_filename, cell_number, cell_row, cell_col, label, c1...c522
meta_cols = ["image_filename", "cell_number", "cell_row", "cell_col", "label"]
feat_cols = [c for c in df.columns if c not in meta_cols]

import pandas as pd
from sklearn.utils import resample

# Assuming your data is in a DataFrame called 'df'
# Separate majority and minority classes
# df_majority = df[df.label == 3]
# df_minority_bat = df[df.label == 0]
# df_minority_ball = df[df.label == 1]
# df_minority_stump = df[df.label == 2]
#
# # Upsample minority classes (example for Bat)
# df_minority_bat_upsampled = resample(df_minority_bat,
#                                      replace=True,  # sample with replacement
#                                      n_samples=len(df_majority),  # to match majority class
#                                      random_state=123)
#
# # Upsample minority classes (example for Ball)
# df_minority_ball_upsampled = resample(df_minority_ball,
#                                      replace=True,  # sample with replacement
#                                      n_samples=len(df_majority),  # to match majority class
#                                      random_state=123)
#
# # Upsample minority classes (example for Stump)
# df_minority_stump_upsampled = resample(df_minority_stump,
#                                      replace=True,  # sample with replacement
#                                      n_samples=len(df_majority),  # to match majority class
#                                      random_state=123)
#
# # Repeat for ball and stump, then combine everything back together
# df = pd.concat([df_majority, df_minority_bat_upsampled,df_minority_ball_upsampled,df_minority_stump_upsampled])

X = df[feat_cols].values
y = df["label"].values      # labels like "bat", "ball", "stump", "background"/"none"



# 3) Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Build a pipeline: scaler + classifier
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )),
    ]
)

# 5) Train
model.fit(X_train, y_train)

# 6) Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict(X_train)
print('*************** classification_report(y_train, y_pred) ***************')
print(classification_report(y_train, y_pred))

# 7) Save trained model and feature column order
joblib.dump(model, "cricket_cell_model.joblib")
joblib.dump(feat_cols, "feature_columns.joblib")
print("Model and feature column names saved.")


def extract_cell_features_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w, _ = img.shape
    n_rows = 8
    n_cols = 8
    cell_h = h // n_rows
    cell_w = w // n_cols

    cell_features = []
    cell_number = 0

    for r in range(n_rows):
        for c in range(n_cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            cell_img = img[y1:y2, x1:x2]

            # YOUR FUNCTION - now confirmed working!
            feature_array = cricket_light_features_522(cell_img)  # returns list of 522 values

            # FIXED: Convert to dict using training column names
            features_dict = dict(zip(feat_cols, feature_array))

            row = {
                "image_filename": image_path,
                "cell_number": cell_number,
                "cell_row": r,
                "cell_col": c,
            }
            row.update(features_dict)
            cell_features.append(row)
            cell_number += 1

    return cell_features


import pandas as pd
import joblib

# 1) Load trained model and feature column order
model = joblib.load("cricket_cell_model.joblib")
feat_cols = joblib.load("feature_columns.joblib")

def predict_objects_in_image(image_path, prob_threshold=0.5):
    cell_features = extract_cell_features_from_image(image_path)
    df_cells = pd.DataFrame(cell_features)

    X_new = df_cells[feat_cols].values

    y_pred = model.predict(X_new)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        class_names = model.named_steps["clf"].classes_
    else:
        proba = None
        class_names = None

    df_cells["pred_label"] = y_pred

    if proba is not None:
        max_proba = proba.max(axis=1)
        df_cells["pred_confidence"] = max_proba
        df_cells.loc[df_cells["pred_confidence"] < prob_threshold, "pred_label"] = "background"

    return df_cells, class_names


def draw_predictions_on_image(image_path, preds_df, color_map=None, prob_threshold=0.5, out_path=None):
    """
    Draw rectangles on cells with predicted objects.
    """
    if color_map is None:
        color_map = {
            "bat": (255, 0, 0),  # Blue (BGR)
            "ball": (0, 0, 255),  # Red
            "stump": (0, 255, 0),  # Green
        }

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    n_rows = 8
    n_cols = 8
    cell_h = h // n_rows
    cell_w = w // n_cols

    for _, row in preds_df.iterrows():
        label = str(row["pred_label"])  # ✅ FIXED: Convert to string
        conf = row.get("pred_confidence", 1.0)

        if label == "background":
            continue
        if conf < prob_threshold:
            continue

        r = int(row["cell_row"])
        c = int(row["cell_col"])

        y1 = r * cell_h
        y2 = (r + 1) * cell_h
        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # ✅ FIXED: Ensure label is string for cv2.putText
        cv2.putText(
            img, label,
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    if out_path is None:
        out_path = "predicted_" + image_path.split("/")[-1]

    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to: {out_path}")




import os
from pathlib import Path

def predict_objects_in_folder(folder_path, prob_threshold=0.5,
                              valid_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Run cell-level predictions for all images in a folder.

    Args:
        folder_path (str or Path): directory containing images.
        prob_threshold (float): confidence threshold for assigning object labels.
        valid_exts (tuple): allowed file extensions.

    Returns:
        all_preds_df: concatenated DataFrame of predictions for all images.
        classes: class names from the model.
    """
    folder_path = Path(folder_path)
    image_paths = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    all_dfs = []
    classes = None

    for img_path in image_paths:
        preds_df, classes = predict_objects_in_image(str(img_path), prob_threshold=prob_threshold)
        all_dfs.append(preds_df)

    if all_dfs:
        all_preds_df = pd.concat(all_dfs, ignore_index=True)
    else:
        all_preds_df = pd.DataFrame()

    return all_preds_df, classes

def annotate_folder(folder_path, out_folder, prob_threshold=0.5):
    os.makedirs(out_folder, exist_ok=True)
    folder_path = Path(folder_path)

    for img_path in folder_path.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        preds_df, _ = predict_objects_in_image(str(img_path), prob_threshold=prob_threshold)
        out_path = str(Path(out_folder) / f"pred_{img_path.name}")
        draw_predictions_on_image(str(img_path), preds_df, prob_threshold=prob_threshold, out_path=out_path)

# Example:
# annotate_folder("test_images_folder", "annotated_outputs", prob_threshold=0.5)


# Example usage:
# if __name__ == "__main__":
#     folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\predicition_images_small"
#     preds_df, classes = predict_objects_in_folder(folder, prob_threshold=0.5)
#     print(preds_df.head())

# Now test your full pipeline:
# folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\predicition_images_small"  # your folder path
# preds_df, classes = predict_objects_in_folder(folder, prob_threshold=0.5)
# print(preds_df[["image_filename", "cell_number", "pred_label", "pred_confidence"]].head(10))
# annotate_folder(folder, "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\annotated_outputs", prob_threshold=0.5)

def save_predictions_as_compact_csv(all_preds_df, output_csv_path, class_to_num_map=None):
    """
    Convert per-cell predictions to compact CSV format:
    ImageFileName, TrainOrTest, c01, c02, ..., c64

    Args:
        all_preds_df: DataFrame from predict_objects_in_folder()
        output_csv_path: where to save the CSV
        class_to_num_map: dict like {"background":0, "bat":1, "ball":2, "stump":3}
    """
    if class_to_num_map is None:
        class_to_num_map = {
            "background": 3,
            "bat": 0,
            "ball": 1,
            "stump": 2
        }

    # Get unique images
    unique_images = all_preds_df["image_filename"].unique()

    compact_data = []

    for img_path in unique_images:
        img_df = all_preds_df[all_preds_df["image_filename"] == img_path].copy()

        # Sort by cell_number (01-64) and ensure we have all 64 cells
        img_df = img_df.sort_values("cell_number").reset_index(drop=True)

        # Map predictions to numbers (0,1,2,3)
        img_df["pred_num"] = img_df["pred_label"].map(class_to_num_map)

        # Create row: ImageFileName, TrainOrTest, c01, c02, ..., c64
        row = {
            "ImageFileName": Path(img_path).name,  # just filename
            "TrainOrTest": "Test",  # since these are predictions
        }

        # Add c01 to c64 columns
        for i, cell_num in enumerate(range(64)):
            col_name = f"c{i + 1:02d}"  # c01, c02, ..., c64
            if i < len(img_df):
                row[col_name] = img_df.iloc[i]["pred_num"]
            else:
                row[col_name] = 0  # default to background

        compact_data.append(row)

    # Save as CSV
    compact_df = pd.DataFrame(compact_data)
    compact_df.to_csv(output_csv_path, index=False)
    print(f"Saved compact predictions CSV: {output_csv_path}")
    print(f"Shape: {compact_df.shape}")
    print(compact_df.head())


# 1. Run predictions
folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\predicition_images_small"
preds_df, classes = predict_objects_in_folder(folder, prob_threshold=0.5)

# 2. Generate compact CSV
save_predictions_as_compact_csv(preds_df, "predictions_c01_c64.csv")

# 3. Annotate images (optional)
annotate_folder(folder, "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\model3\\annotated_outputs", prob_threshold=0.5)

print("✅ All done!")

