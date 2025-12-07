import pandas as pd
import cv2
import numpy as np
from features import extract_features,cricket_light_features_2000,extract_features_135,cricket_light_features_522


import pandas as pd
import os


def transform_csv(input_path, output_path, label_map=None):
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
label_map = {"Bat": 0, "Ball": 1, "Stump": 2}

# transform_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\sample_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\transformed_sample_tagged_data.csv", label_map)







def process_image_to_cells(image_path, label_map, grid_size=8):
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
            feats = cricket_light_features_522(cell)

            cell_number = r * grid_size + c

            rows.append((cell_number, r, c, feats))

    return rows

def generate_feature_csv(input_csv, output_csv, label_map):
    # df = pd.read_csv(input_csv, header=None,
                     # names=["image_filename", "cell_number", "cell_row", "cell_col", "label"])
    df = pd.read_csv(input_csv, header=0)

    final_rows = []

    for image_name in df["image_filename"].unique():
        IMAGE_DIR = r"C:\Users\ASUS\PycharmProjects\cricket-object-identifier\resources\images\sample_images\processed_images"

        full_path = os.path.join(IMAGE_DIR, image_name)

        # full_path = os.path.join("path_to_images", image_name)
        cell_features = process_image_to_cells(full_path, label_map)

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
    return output_csv


#Usage
label_map = {"Bat": 0, "Ball": 1, "Stump": 2}
# generate_feature_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\transformed_sample_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\sample_images\\features_sample_tagged_data.csv", label_map)
