
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_TEST_FILE = PROJECT_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

TARGET = "income"
POSITIVE_CLASS = ">50K"
NEGATIVE_CLASS = "<=50K"

# --------------------------------------------------
# 2. APPLICATION HEADER
# --------------------------------------------------

st.title("📊 Adult Income Classification Dashboard")

st.markdown(
    """
    ### Machine Learning Classification Project

    This application evaluates five trained classification
    models using the UCI Adult Income dataset.

    Upload your test CSV, select a classifier and explore
    its performance using standard evaluation metrics.
    """
)

st.divider()

# --------------------------------------------------
# 3. LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(
            model_path
        )

    return loaded_models


try:
    trained_models = load_models()

    st.success(
        f"Successfully loaded {len(trained_models)} trained models."
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()

# --------------------------------------------------
# 4. DATASET UPLOAD
# --------------------------------------------------

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

try:

    if uploaded_file is not None:

        test_df = pd.read_csv(uploaded_file)

        st.info("Using uploaded test dataset.")

    elif DEFAULT_TEST_FILE.exists():

        test_df = pd.read_csv(DEFAULT_TEST_FILE)

        st.info("Using the default test dataset.")

    else:

        st.warning("Please upload a test CSV file.")
        st.stop()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()

# --------------------------------------------------
# 5. DATA VALIDATION
# --------------------------------------------------

st.subheader("2. Dataset Overview")

if TARGET not in test_df.columns:

    st.error(
        "The uploaded dataset must contain the income target column."
    )
    st.stop()

# Remove whitespace and normalize target labels
test_df[TARGET] = (
    test_df[TARGET]
    .astype(str)
    .str.strip()
    .str.rstrip(".")
)

expected_features = set(
    trained_models["Logistic Regression"].feature_names_in_
)

actual_features = set(test_df.columns) - {TARGET}

missing_features = expected_features - actual_features
extra_features = actual_features - expected_features

if missing_features or extra_features:

    st.error(
        f"Feature mismatch. Missing: {sorted(missing_features)}. "
        f"Unexpected: {sorted(extra_features)}"
    )
    st.stop()

# Reorder features to match the training dataset
X_test = test_df[list(
    trained_models["Logistic Regression"].feature_names_in_
)]

y_test = test_df[TARGET]

valid_classes = {NEGATIVE_CLASS, POSITIVE_CLASS}

if set(y_test.unique()) != valid_classes:

    st.error(
        "Target column must contain both <=50K and >50K classes."
    )
    st.stop()

# Display dataset information
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(test_df):,}")
col2.metric("Input Features", len(X_test.columns))
col3.metric("Target Classes", y_test.nunique())

with st.expander("Preview uploaded dataset"):

    st.dataframe(
        test_df.head(10),
        use_container_width=True
    )

# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

st.subheader("3. Model Selection")

selected_model_name = st.selectbox(
    "Choose a classification model",
    list(MODEL_FILES.keys())
)

# Evaluate all five saved models
all_results = {}
all_predictions = {}

for model_name, model in trained_models.items():

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        (y_test == POSITIVE_CLASS).astype(int),
        probabilities
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=POSITIVE_CLASS,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=POSITIVE_CLASS,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=POSITIVE_CLASS,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        predictions
    )

    all_results[model_name] = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

    all_predictions[model_name] = predictions

# --------------------------------------------------
# 7. DISPLAY SELECTED MODEL METRICS
# --------------------------------------------------

selected_metrics = all_results[selected_model_name]

st.subheader(
    f"4. Performance Metrics – {selected_model_name}"
)

metric_columns = st.columns(6)

for column, (metric_name, metric_value) in zip(
    metric_columns,
    selected_metrics.items()
):

    column.metric(
        metric_name,
        f"{metric_value:.4f}"
    )

# --------------------------------------------------
# 8. DISPLAY ALL MODEL COMPARISONS
# --------------------------------------------------

st.subheader("5. Five-Model Performance Comparison")

comparison_df = pd.DataFrame.from_dict(
    all_results,
    orient="index"
)

comparison_df.index.name = "Model"

st.dataframe(
    comparison_df.style.format("{:.4f}").highlight_max(
        axis=0,
        color="lightgreen"
    ),
    use_container_width=True
)

# --------------------------------------------------
# 9. CONFUSION MATRIX
# --------------------------------------------------

st.subheader("6. Confusion Matrix")

selected_predictions = all_predictions[
    selected_model_name
]

matrix = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[NEGATIVE_CLASS, POSITIVE_CLASS]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual <=50K", "Actual >50K"],
    columns=["Predicted <=50K", "Predicted >50K"]
)

st.dataframe(
    matrix_df,
    use_container_width=True
)

# --------------------------------------------------
# 10. CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("7. Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    use_container_width=True
)

# --------------------------------------------------
# 11. APPLICATION FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "BITS Pilani WILP | M.Tech Artificial Intelligence "
    "and Machine Learning | Machine Learning Assignment 2"
)

st.caption(
    "Models are loaded from saved Joblib files. "
    "Uploaded test data is evaluated without retraining."
)

