from utils_basic import transform_csv, generate_feature_csv, generate_metadata_csv,add_features_from_csv
from sklearn.model_selection import train_test_split
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def build_models() -> dict:
    models = {}
    models["RandomForest"] = Pipeline([
        # ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )),
    ])
    # models["SVM_RBF"] = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', random_state=42)),
    # ])
    models["kNN"] = Pipeline([
        # ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=None)),
    ])
    # models["MLP"] = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", MLPClassifier(hidden_layer_sizes=(256,128), activation='relu', alpha=1e-4, batch_size='auto',
    #                            learning_rate='adaptive', max_iter=300, random_state=42, early_stopping=True)),
    # ])
    # models["XGBoost"] = XGBClassifier(
    #         n_estimators=400,
    #         max_depth=6,
    #         learning_rate=0.05,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         reg_lambda=1.0,
    #         objective='multi:softprob',
    #         num_class=len(set(y)),
    #         tree_method='hist',
    #         random_state=42,
    #     )
    return models


def evaluate_models(models: dict, X_train_res, y_train_res, X_test, y_test) -> dict:
    results = {}

    for name, mdl in models.items():
        print("\n" + "=" * 80)
        print(f"Training {name}")

        # Train
        mdl.fit(X_train_res, y_train_res)

        # Predict
        y_pred = mdl.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (only if model supports predict_proba OR decision_function)
        roc_auc = None
        try:
            if hasattr(mdl, "predict_proba"):
                y_scores = mdl.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_scores)
            elif hasattr(mdl, "decision_function"):
                y_scores = mdl.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_scores)
        except:
            roc_auc = None

        # Print results
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 Score : {f1:.3f}")
        if roc_auc is not None:
            print(f"ROC-AUC  : {roc_auc:.3f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Store everything
        results[name] = {
            "model": mdl,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "y_pred": y_pred
        }

    return results


def plot_comparisons(results: dict):
    names = list(results.keys())

    # Extract stored metrics
    accs    = [results[n]["accuracy"] for n in names]
    f1s     = [results[n]["f1"] for n in names]            # weighted F1
    recalls = [results[n]["recall"] for n in names]        # weighted recall
    precision = [results[n]["precision"] for n in names]

    # ---- Plot Accuracy ----
    plt.figure(figsize=(10,5))
    plt.bar(names, accs)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    print("Saved plot: model_accuracy_comparison.png")

    # ---- Plot Precision ----
    plt.figure(figsize=(10,5))
    plt.bar(names, precision)
    plt.ylabel('Precision')
    plt.title('Model Comparison - Precision')
    plt.ylim(0, 1)
    for i, v in enumerate(precision):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("model_precision_comparison.png")
    print("Saved plot: model_precision_comparison.png")

    # ---- Plot Weighted F1 ----
    plt.figure(figsize=(10,5))
    plt.bar(names, f1s)
    plt.ylabel('F1 Score (Weighted)')
    plt.title('Model Comparison - F1 Score (Weighted)')
    plt.ylim(0, 1)
    for i, v in enumerate(f1s):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("model_f1_comparison.png")
    print("Saved plot: model_f1_comparison.png")

    # ---- Plot Weighted Recall ----
    plt.figure(figsize=(10,5))
    plt.bar(names, recalls)
    plt.ylabel('Recall (Weighted)')
    plt.title('Model Comparison - Recall (Weighted)')
    plt.ylim(0, 1)
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("model_recall_comparison.png")
    print("Saved plot: model_recall_comparison.png")

    # ---- Confusion Matrices ----
    for name in names:
        cm = results[name]["confusion_matrix"]
        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')

        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center', color='black')

        plt.tight_layout()
        fname = f"confusion_matrix_{name}.png"
        plt.savefig(fname)
        print(f"Saved plot: {fname}")

def split_train_test_images(features_df, image_dir):
    # Get unique image names
    unique_images = features_df['image_filename'].unique()

    # Split images into train and test sets (80/20)
    train_imgs, test_imgs = train_test_split(
        unique_images, test_size=0.2, random_state=42
    )

    # Define output directories
    base_dir = os.path.abspath(os.path.join(image_dir, ".."))
    train_dir = os.path.join(base_dir, "train_images")
    test_dir = os.path.join(base_dir, "test_images")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy train images
    for img in train_imgs:
        src = os.path.join(image_dir, img)
        dst = os.path.join(train_dir, img)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found")

    # Copy test images
    for img in test_imgs:
        src = os.path.join(image_dir, img)
        dst = os.path.join(test_dir, img)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found")

    return train_imgs, test_imgs


# Uncomment and run for transforming file generated from image_tagging.ipynb tool used for image tagging
# transform_csv(
#     "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\tagged_data.csv",
#     "C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\transformed_tagged_data.csv",
# )

#Used for generating metadata file by adding data for background since image tagging utility only adds values for bat,ball,stump
# generate_metadata_csv(
#     input_csv="C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\transformed_tagged_data.csv",
#     output_csv="C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\features_final_combinedAll_tagged_data.csv",
#     image_dir="C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\all_images"
# )
#

#Add features by using HOG
feature_csv = add_features_from_csv(
    csv_path="C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\features_final_combinedAll_tagged_data.csv",
    image_dir="C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\all_images"
)


data = feature_csv
X = data.drop(columns=["image_filename", "cell_number", "cell_row", "cell_col", "label"])
y = data["label"]
feat_cols = X.columns.tolist()

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sampler = RandomOverSampler(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

split_train_test_images(data,"C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\all_images")

# Build models, train, evaluate, compare
models = build_models()
results = evaluate_models(models, X_train_res, y_train_res, X_test, y_test)
plot_comparisons(results)

# Choose best by accuracy and save
best_name = max(results.keys(), key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
print(f"Best model: {best_name} (acc={results[best_name]['accuracy']:.3f})")

# Choose best by MACRO F1 instead of accuracy
# best_name = max(results.keys(), key=lambda n: results[n]["macro_f1"])
# best_model = results[best_name]["model"]
# print(f"Best model: {best_name} (macro_f1={results[best_name]['macro_f1']:.3f})")

# Choose best by MACRO Recall instead of accuracy
# best_name = max(results.keys(), key=lambda n: results[n]["macro_recall"])
# best_model = results[best_name]["model"]
# print(f"Best model: {best_name} (macro_recall={results[best_name]['macro_recall']:.3f})")

#Combined final pickle
with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\cricket_cell_model_final_combined.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\final_combined\\feature_columns_final_combined.pkl", "wb") as f:
    pickle.dump(feat_cols, f)

#STUMP pickle
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\cricket_cell_model_stump.pkl", "wb") as f:
#     pickle.dump(best_model, f)
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\feature_columns_stump.pkl", "wb") as f:
#     pickle.dump(feat_cols, f)

#BAT pickle
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\bat\\cricket_cell_model_bat.pkl", "wb") as f:
#     pickle.dump(best_model, f)
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\bat\\feature_columns_bat.pkl", "wb") as f:
#     pickle.dump(feat_cols, f)

#BALL pickle
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\ball\\cricket_cell_model_ball.pkl", "wb") as f:
#     pickle.dump(best_model, f)
# with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\ball\\feature_columns_ball.pkl", "wb") as f:
#     pickle.dump(feat_cols, f)
print("Saved best model and feature columns (pickle)")



