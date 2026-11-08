# Adult Income Classification – Machine Learning Model Comparison

## 1. Problem Statement

The objective of this project is to build and compare five supervised machine learning classification algorithms to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models are trained and evaluated using the same dataset and preprocessing approach.

## 2. Dataset Description

**Dataset:** Adult Income (Census Income) Dataset
**Source:** UCI Machine Learning Repository
**Dataset link:** https://archive.ics.uci.edu/dataset/2/adult

The dataset contains demographic and employment-related information used to predict whether an individual's annual income is above or below $50K.

- Total records: 48,842
- Training records: 32,561
- Testing records: 16,281
- Input features: 14
- Target variable: Income
- Target classes: <=50K and >50K

The dataset contains numerical and categorical features. Missing values are present in certain categorical columns.

## 3. Data Preprocessing

The following preprocessing steps were applied:

- Separated input features and target variable.
- Numerical missing values were handled using median imputation.
- Categorical missing values were handled using most-frequent imputation.
- Numerical features were standardized using StandardScaler.
- Categorical features were encoded using OneHotEncoder.
- A ColumnTransformer was used to combine numerical and categorical preprocessing.
- Preprocessing was fitted only on training data and then applied to testing data to avoid data leakage.

The preprocessing and classification model were combined into pipelines.

## 4. Models Implemented

### Logistic Regression
Used as a linear classification baseline to model the probability of income belonging to the >50K class.

### Decision Tree
Used a tree-based classifier with a maximum depth of 15 to capture non-linear relationships.

### K-Nearest Neighbors
Used KNN with 5 neighbors to classify observations based on nearby training samples.

### Gaussian Naive Bayes
Used Gaussian Naive Bayes to estimate class probabilities based on the input features.

### Support Vector Machine
Used an RBF-kernel SVM with probability calibration for probability-based evaluation.

## 5. Model Performance Comparison

All models were evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8507 | 0.9038 | 0.7282 | 0.5871 | 0.6501 | 0.5618 |
| Decision Tree | 0.8476 | 0.8612 | 0.6937 | 0.6355 | 0.6633 | 0.5660 |
| KNN | 0.8333 | 0.8564 | 0.6639 | 0.5962 | 0.6282 | 0.5224 |
| Gaussian Naive Bayes | 0.5511 | 0.7611 | 0.3378 | 0.9376 | 0.4967 | 0.3300 |
| SVM | 0.8577 | 0.8994 | 0.7622 | 0.5783 | 0.6576 | 0.5786 |

## 6. Observations

### Logistic Regression
Logistic Regression achieved the highest AUC score of 0.9038, indicating strong ranking and discrimination capability. It also achieved good accuracy and precision.

### Decision Tree
Decision Tree achieved the highest F1-score of 0.6633. It provides a reasonable balance between precision and recall and can capture non-linear patterns.

### K-Nearest Neighbors
KNN achieved moderate performance across the evaluation metrics. Its accuracy was 0.8333, while its F1-score was 0.6282.

### Gaussian Naive Bayes
Gaussian Naive Bayes achieved the highest recall of 0.9376 but had low precision of 0.3378. This indicates that it identifies most positive income cases but also produces many false positives.

### Support Vector Machine
SVM achieved the highest accuracy of 0.8577 and the highest MCC of 0.5786. It also achieved the highest precision of 0.7622 among the five models.

## 7. Overall Best Model

Based on the overall comparison, Support Vector Machine (SVM) is considered the best-performing model for this experiment because it achieved the highest accuracy (0.8577) and MCC (0.5786).

However, the best model can depend on the evaluation objective. Logistic Regression achieved the highest AUC, Decision Tree achieved the highest F1-score, and Gaussian Naive Bayes achieved the highest recall.

Therefore, model selection should consider the specific requirements of the classification problem rather than relying on accuracy alone.

## 8. GitHub Repository

Repository: https://github.com/gkumar052/ML-Assignment-2-Adult-Income.git

## 9. Live Streamlit Application

Live application: https://adult-income-ml-classifier.streamlit.app/

The Streamlit application allows users to:

- Upload a test CSV file.
- Select a classification model.
- View evaluation metrics.
- View the confusion matrix and classification report.
- Compare the performance of all five trained models.

The application loads the saved models and evaluates uploaded test data without retraining the models.

## 10. Project Files

- `ML_Assignment_2_Experiment.ipynb` – Model development, training and evaluation.
- `app.py` – Streamlit application.
- `requirements.txt` – Python dependencies.
- `test_data.csv` – Test dataset for application evaluation.
- `models/` – Five saved trained machine learning pipelines.
- `results/` – Model comparison, confusion matrices and classification reports.
- `Screenshot_2026-11-08_04-58-33.png` – BITS Virtual Lab execution screenshot.

## 11. Conclusion

This project implemented and compared five supervised machine learning classification algorithms on the Adult Income dataset. The experiment demonstrates how different algorithms perform across multiple evaluation metrics.

SVM achieved the best accuracy and MCC, while Logistic Regression achieved the best AUC. The results highlight the importance of evaluating classification models using multiple performance measures.
