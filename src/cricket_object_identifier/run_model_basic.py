from utils_basic import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingRegressor


try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
    print('HAS_XGB : ',HAS_XGB)
except Exception:
    HAS_XGB = False
    print('HAS_XGB : ', HAS_XGB)


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

def build_models() -> dict:
    models = {}
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )),
    ])
    models["SVM_RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)),
    ])
    models["kNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=None)),
    ])
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,128), activation='relu', alpha=1e-4, batch_size='auto',
                               learning_rate='adaptive', max_iter=300, random_state=42, early_stopping=True)),
    ])
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=len(set(y)),
            tree_method='hist',
            random_state=42,
        )
    return models


def evaluate_models(models: dict, X_train_res, y_train_res, X_test, y_test) -> dict:
    results = {}
    for name, mdl in models.items():
        print("\n" + "="*80)
        print(f"Training {name}")
        mdl.fit(X_train_res, y_train_res)
        y_pred = mdl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy ({name}): {acc:.3f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        results[name] = {
            "model": mdl,
            "accuracy": acc,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }
    return results


def plot_comparisons(results: dict):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(10,5))
    plt.bar(names, accs, color='steelblue')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison (Accuracy)')
    plt.ylim(0, 1)
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    print("Saved plot: model_accuracy_comparison.png")

    # Optional: plot confusion matrices
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


def build_simple_ensemble(results: dict):
    # Probability averaging ensemble for models supporting predict_proba
    prob_models = {n: r["model"] for n, r in results.items() if hasattr(r["model"], "predict_proba")}
    if not prob_models:
        print("No models with predict_proba available for ensembling.")
        return None
    print(f"Ensembling {list(prob_models.keys())}")
    return prob_models


# Build models, train, evaluate, compare
models = build_models()
results = evaluate_models(models, X_train_res, y_train_res, X_test, y_test)
plot_comparisons(results)

# Choose best by accuracy and save
best_name = max(results.keys(), key=lambda n: results[n]["accuracy"]) 
best_model = results[best_name]["model"]
print(f"Best model: {best_name} (acc={results[best_name]['accuracy']:.3f})")

with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\cricket_cell_model_stump.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("C:\\Users\\ASUS\\PycharmProjects\\cricket-object-identifier\\resources\\images\\cricket_objects\\stump\\feature_columns_stump.pkl", "wb") as f:
    pickle.dump(feat_cols, f)
print("Saved best model and feature columns (pickle)")

# Optional: simple ensemble usage example on test set
ensemble_models = build_simple_ensemble(results)
if ensemble_models is not None:
    # Average probabilities
    prob_list = []
    for name, mdl in ensemble_models.items():
        if isinstance(mdl, Pipeline):
            prob = mdl.predict_proba(X_test)
        else:
            prob = mdl.predict_proba(X_test)
        prob_list.append(prob)
    avg_prob = sum(prob_list) / len(prob_list)
    y_pred_ens = avg_prob.argmax(axis=1)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    print(f"Ensemble accuracy (avg-prob): {acc_ens:.3f}")

