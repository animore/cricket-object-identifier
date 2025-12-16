# Cricket Object Identifier - Test Suite

## Problem Description
This project aims to build a model that detects whether regions in a cricket image contain a bat, a ball, stumps, or no object. The model should classify each region of an image into one of these categories.

## Dataset Requirements
- At least 300 images must be collected, with a balanced mix of:
	- Cricket bat images (various angles, partial/complete views)
	- Cricket ball images (on ground, in air, in hand, etc.)
	- Stumps (clear, occluded, etc.)
	- No-object images (grass, pitch, background scenes)
- All images must have a 4:3 aspect ratio and be resized to 800x600 pixels. Higher resolution images can be downsized, but images lower than 800x600 should not be used.
- A short README file describing image sources should be included in the dataset folder.

## Modeling Task
- Each 800x600 image is divided into an 8×8 grid (64 cells).
- For each grid cell, the model predicts:
	- 0 → no object
	- 1 → ball
	- 2 → bat
	- 3 → stump
	- If multiple objects are present in a cell, the classifier should detect any ONE of them.
- Only hand-crafted feature engineering techniques are allowed (no CNNs or similar automatic feature extraction methods).
- The trained model should be saved as `model_<teamname>.pkl`.

## Output Format
- The model should be run on both train and test datasets.
- Results should be saved in a CSV file with the following format:
	- `ImageFileName, TrainOrTest, c01, c02, ..., c64`
	- Each column c01 to c64 represents the predicted class for the corresponding grid cell (0, 1, 2, or 3).

### Example
For an image:
- Column c35 should be marked as 1 if it contains the ball.
- Columns c20, c55, and c63 should be marked as 2 if they contain the bat.
- Columns c28 and c36 should be marked as 3 if they contain the stump.
- All other columns should be marked as 0 (no object).


## How to Run the Program

1. **Train the Model:**
	- Execute `run_model_basic.py` to train the model. This script will generate a pickle file for the trained model and another pickle file for the feature columns.

2. **Annotate Images:**
	- Use the generated model and feature columns pickle files as input to `annotate_images.py`.
	- Run `annotate_images.py` to annotate images using the trained model.

---

