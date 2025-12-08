from utils2 import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================
print("=" * 80)
print("PHASE 1: TRAINING")
print("=" * 80)

label_map = {"Bat": 0, "Ball": 1, "Stump": 2}

# Transform CSV
transform_csv(
    "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\shubham_tagged_data.csv",
    "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\transformed_shubham_tagged_data.csv",
    label_map
)

# Generate features with augmentation
feature_csv = generate_feature_csv(
    "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\transformed_shubham_tagged_data.csv",
    "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\features_shubham_tagged_data.csv",
    label_map,
    "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\processed_imagesSH\\processed_images",
    augment_count=2
)

# Load features CSV
csv_path = feature_csv
df = pd.read_csv(csv_path)

# Define feature columns
meta_cols = ["image_filename", "cell_number", "cell_row", "cell_col", "label"]
feat_cols = [c for c in df.columns if c not in meta_cols]

print(f"\n✅ Feature count: {len(feat_cols)}")
print(f"   Sample features: {feat_cols[:5]}")

# Balance classes
from sklearn.utils import resample

df_majority = df[df.label == 3]
df_minority_bat = df[df.label == 0]
df_minority_ball = df[df.label == 1]
df_minority_stump = df[df.label == 2]

df_minority_bat_upsampled = resample(df_minority_bat, replace=True, n_samples=len(df_majority), random_state=123)
df_minority_ball_upsampled = resample(df_minority_ball, replace=True, n_samples=len(df_majority), random_state=123)
df_minority_stump_upsampled = resample(df_minority_stump, replace=True, n_samples=len(df_majority), random_state=123)

df = pd.concat([df_majority, df_minority_bat_upsampled, df_minority_ball_upsampled, df_minority_stump_upsampled])

X = df[feat_cols].values
y = df["label"].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build and train model
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)),
    ]
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)
print(classification_report(y_val, y_pred))

y_train_pred = model.predict(X_train)
print("\n" + "=" * 80)
print("TRAINING RESULTS")
print("=" * 80)
print(classification_report(y_train, y_train_pred))

print(f"\nTraining accuracy: {model.score(X_train, y_train):.3f}")
print(f"Validation accuracy: {model.score(X_val, y_val):.3f}")

# Save model and feature columns
with open("cricket_cell_model.pkl", "wb") as f:
    pickle.dump(model, f)
joblib.dump(feat_cols, "feature_columns.joblib")
print("\n✅ Model and feature columns saved!")

# ============================================================================
# PHASE 2: PREDICTION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: PREDICTION")
print("=" * 80)

# Load trained model and feature columns
with open("cricket_cell_model.pkl", "rb") as f:
    model = pickle.load(f)
feat_cols = joblib.load("feature_columns.joblib")

print(f"\n✅ Loaded model with {len(feat_cols)} features")

from features import cricket_light_features_2000

def extract_cell_features_from_image(image_path):
    """Extract features from image cells using cricket_light_features_2000."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w, _ = img.shape
    n_rows, n_cols = 8, 8
    cell_h, cell_w = h // n_rows, w // n_cols

    cell_features = []
    cell_number = 0

    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell_img = img[y1:y2, x1:x2]

            feature_array = cricket_light_features_2000(cell_img)

            row = {
                "image_filename": image_path,
                "cell_number": cell_number,
                "cell_row": r,
                "cell_col": c,
            }
            
            # Fill features, padding with 0 if short
            for i, feat_name in enumerate(feat_cols):
                row[feat_name] = feature_array[i] if i < len(feature_array) else 0.0
            
            cell_features.append(row)
            cell_number += 1

    return cell_features


def predict_objects_in_image(image_path, prob_threshold=0.3):
    """Predict objects in a single image."""
    cell_features = extract_cell_features_from_image(image_path)
    df_cells = pd.DataFrame(cell_features)

    X_new = df_cells[feat_cols].values
    y_pred = model.predict(X_new)
    
    proba = model.predict_proba(X_new)
    max_proba = proba.max(axis=1)
    df_cells["pred_confidence"] = max_proba

    class_map = {0: "bat", 1: "ball", 2: "stump", 3: "background"}
    df_cells["pred_label"] = df_cells[list(range(len(y_pred)))].apply(lambda row: class_map.get(y_pred[row.name], "background"), axis=1)

    df_cells.loc[df_cells["pred_confidence"] < prob_threshold, "pred_label"] = "background"

    print(f"\n📊 {Path(image_path).name}:")
    print(f"   Bat: {(df_cells['pred_label'] == 'bat').sum()}, Ball: {(df_cells['pred_label'] == 'ball').sum()}, Stump: {(df_cells['pred_label'] == 'stump').sum()}")
    print(f"   Avg confidence: {df_cells['pred_confidence'].mean():.3f}")

    return df_cells, model.named_steps["clf"].classes_


def draw_predictions_on_image(image_path, preds_df, prob_threshold=0.3, out_path=None):
    """Draw predictions on image."""
    color_map = {"bat": (255, 0, 0), "ball": (0, 0, 255), "stump": (0, 255, 0)}

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    cell_h, cell_w = h // 8, w // 8

    for _, row in preds_df.iterrows():
        if row["pred_label"] == "background" or row["pred_confidence"] < prob_threshold:
            continue

        r, c = int(row["cell_row"]), int(row["cell_col"])
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w

        color = color_map.get(row["pred_label"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, row["pred_label"], (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if out_path is None:
        out_path = f"pred_{Path(image_path).name}"

    cv2.imwrite(out_path, img)
    print(f"✅ Saved: {out_path}")


def predict_objects_in_folder(folder_path, prob_threshold=0.3):
    """Predict for all images in folder."""
    folder_path = Path(folder_path)
    image_paths = [p for p in folder_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

    all_dfs = []
    for img_path in image_paths:
        preds_df, _ = predict_objects_in_image(str(img_path), prob_threshold)
        all_dfs.append(preds_df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def annotate_folder(folder_path, out_folder, prob_threshold=0.3):
    """Annotate all images."""
    os.makedirs(out_folder, exist_ok=True)
    for f in Path(out_folder).glob("*.jpg"):
        f.unlink()

    for img_path in Path(folder_path).glob("*.[jp][pn]g"):
        preds_df, _ = predict_objects_in_image(str(img_path), prob_threshold)
        draw_predictions_on_image(str(img_path), preds_df, prob_threshold, f"{out_folder}/pred_{img_path.name}")


# Run predictions
folder = "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\shubham_images\\Processed_800x600RawSH\\Processed_800x600"
preds_df = predict_objects_in_folder(folder, prob_threshold=0.5)

annotate_folder(folder, "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\model3\\annotated_outputs", prob_threshold=0.5)

print("\n✅ Complete!")



