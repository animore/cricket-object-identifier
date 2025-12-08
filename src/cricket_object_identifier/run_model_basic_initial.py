from utils_basic import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle


transform_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\stump_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\transformed_stump_tagged_data.csv")

feature_csv = generate_feature_csv("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\transformed_stump_tagged_data.csv", "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\features_stump_tagged_data.csv","C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\stump_processed_images")

data = feature_csv
X = data.drop(columns=["image_filename", "cell_number", "cell_row", "cell_col", "label"])
y = data["label"]
feat_cols = X.columns.tolist()

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sampler = RandomOverSampler(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# Create and train the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,  # Limit depth to reduce overfitting
        class_weight='balanced',  # Backup weighting
        random_state=42
    ))
])


# model.fit(X_train, y_train)
model.fit(X_train_res, y_train_res)
# Evaluate the model
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n" + "=" * 80)
print("TRAINING RESULTS")
y_train_pred = model.predict(X_train)
print(classification_report(y_train, y_train_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("\n" + "=" * 80)

# Save trained model and feature column order using pickle
with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\cricket_cell_model_basic.pkl", "wb") as f:
    pickle.dump(model, f)
with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\feature_columns_basic.pkl", "wb") as f:
    pickle.dump(feat_cols, f)
print("Model and feature column names saved as pickle files.")
