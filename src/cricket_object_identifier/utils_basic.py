import pandas as pd
import cv2
import numpy as np
from features_basic import *


import pandas as pd
import os

label_map = {"Bat": 1, "Ball": 2, "Stump": 3}

def transform_csv(input_path, output_path):
    df = pd.read_csv(input_path)  # assumes CSV has header

    rows_out = []

    for _, row in df.iterrows():
        values = row.dropna().tolist()

        if len(values) == 0:
            continue

        filename = values[0]
        obj_data = values[1:]

        added_any = False  # track whether any real objects were added

        # process triplets
        for i in range(0, len(obj_data), 3):
            if i + 2 >= len(obj_data):
                break

            try:
                cell_row = int(obj_data[i])
                cell_col = int(obj_data[i + 1])
            except ValueError:
                # not numeric → skip this triplet
                continue

            label = obj_data[i + 2]

            if label_map is not None:
                label = label_map.get(label, label)

            cell_number = cell_row * 8 + cell_col

            rows_out.append([filename, cell_number, cell_row, cell_col, label])
            added_any = True

        # If no triplets found → add a blank record for this image
        if not added_any:
            rows_out.append([filename, None, None, None, None])

    out_df = pd.DataFrame(
        rows_out,
        columns=["image_filename", "cell_number", "cell_row", "cell_col", "label"]
    )

    out_df.to_csv(output_path, index=False)
    print("Saved:", output_path)



# Example usage:


# transform_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\sample_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\transformed_sample_tagged_data.csv", label_map)


def process_image_to_cells(image_path, grid_size=8):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    rows = []

    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = r*cell_h, (r+1)*cell_h
            x1, x2 = c*cell_w, (c+1)*cell_w

            cell = img[y1:y2, x1:x2]
            # feats = []  # No features, just an empty list
            # feats = cricket_light_features(cell)

            cell_number = r * grid_size + c
            # rows.append((cell_number, r, c, feats))
            rows.append((cell_number, r, c))

    return rows

#Old method used to generate csv with features to visually check how many features are generated
def generate_feature_csv(input_csv, output_csv,image_dir):
    # df = pd.read_csv(input_csv, header=None,
                     # names=["image_filename", "cell_number", "cell_row", "cell_col", "label"])
    df = pd.read_csv(input_csv, header=0)

    final_rows = []

    for image_name in df["image_filename"].unique():
        # image_dir = r"C:\Users\ASUS\PycharmProjects\cricket-object-identifier\resources\images\sample_images\processed_images"

        full_path = os.path.join(image_dir, image_name)

        # full_path = os.path.join("path_to_images", image_name)
        cell_features = process_image_to_cells(full_path)

        # merge features with your existing label CSV
        subset = df[df["image_filename"] == image_name]

        background_id = 3  # no object
        for (cell_number, r, c, feats) in cell_features:
            label_row = subset[(subset["cell_row"] == r) & (subset["cell_col"] == c)]
            # label = label_row["label"].values[0] if len(label_row) else None
            if len(label_row):
                label = label_row["label"].values[0]  # 0,1,2
            else:
                label = background_id  # unlabeled => background

            # numeric mapping
            if label is not None:
                label = label_map.get(label, label)

            final_rows.append([image_name, cell_number, r, c, label] + feats)

    # Build column names
    feature_cols = [f"c{i}" for i in range(1, len(final_rows[0]) - 5 + 1)]
    columns = ["image_filename", "cell_number", "cell_row", "cell_col", "label"] + feature_cols

    out = pd.DataFrame(final_rows, columns=columns)
    out.to_csv(output_csv, index=False)
    return out


#Usage
# label_map = {"Bat": 0, "Ball": 1, "Stump": 2}
# generate_feature_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\transformed_sample_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\features_sample_tagged_data.csv", label_map)


def generate_metadata_csv(input_csv, output_csv, image_dir, grid_size=8):
    df = pd.read_csv(input_csv)

    final_rows = []
    background_id = 0  # no object

    for image_name in df["image_filename"].unique():
        full_path = os.path.join(image_dir, image_name)
        cell_info = process_image_to_cells(full_path, grid_size)

        subset = df[df["image_filename"] == image_name]

        for cell_number, r, c in cell_info:
            label_row = subset[
                (subset["cell_row"] == r) & (subset["cell_col"] == c)
            ]

            if len(label_row):
                label = label_row["label"].values[0]
            else:
                label = background_id

            label = label_map.get(label, label)

            final_rows.append([
                image_name, cell_number, r, c, label
            ])

    columns = ["image_filename", "cell_number", "cell_row", "cell_col", "label"]
    out = pd.DataFrame(final_rows, columns=columns)
    out.to_csv(output_csv, index=False)

    return out


def add_features_from_csv(csv_path, image_dir, grid_size=8):
    df = pd.read_csv(csv_path)

    feature_rows = []

    for image_name in df["image_filename"].unique():
        img = cv2.imread(os.path.join(image_dir, image_name))
        h, w = img.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size

        image_df = df[df["image_filename"] == image_name]

        for _, row in image_df.iterrows():
            r, c = int(row.cell_row), int(row.cell_col)

            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w

            cell = img[y1:y2, x1:x2]
            feats = cricket_light_features(cell)

            feature_rows.append(feats)

    feature_df = pd.DataFrame(feature_rows,
                              columns=[f"c{i}" for i in range(1, len(feature_rows[0]) + 1)])

    final_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    return final_df

def combine_train_test_csvs(
    train_csv_path,
    test_csv_path
):
    """
    Combine train and test prediction CSVs into a single file.

    Output:
        Written in the SAME directory as test_csv_path
        Filename: train_test_predictions.csv
    """

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Safety check: ensure same columns
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and Test CSV column mismatch")

    combined_df = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True
    )

    # Output CSV path = same dir as test CSV
    test_dir = os.path.dirname(os.path.abspath(test_csv_path))
    output_csv_path = os.path.join(
        test_dir, "train_test_predictions.csv"
    )

    combined_df.to_csv(output_csv_path, index=False)

    print(f"Combined CSV saved as: {output_csv_path}")
