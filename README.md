# Loan Approval Prediction

A machine learning project that predicts whether a loan application is
**Approved** or **Rejected** using applicant and financial information.
The project uses **Logistic Regression** and includes exploratory data
analysis, preprocessing, model evaluation, and a Streamlit interface.

## Project Overview

The goal is to build a binary classification model that learns patterns
from historical loan applications and predicts a status for a new
applicant.

### Dataset Features

The dataset includes the following input features:

-   `no_of_dependents`
-   `education`
-   `self_employed`
-   `income_annum`
-   `loan_amount`
-   `loan_term`
-   `cibil_score`
-   `residential_assets_value`
-   `commercial_assets_value`
-   `luxury_assets_value`
-   `bank_asset_value`

**Target:** `loan_status` (`Approved` or `Rejected`)

The `loan_id` identifier is removed before modeling. Column names are
stripped of leading/trailing spaces.

## Workflow

1.  **Data inspection and cleaning**
    -   Inspect dataset structure and summary statistics.
    -   Check missing values and duplicate rows.
    -   Remove `loan_id`.
2.  **Exploratory Data Analysis**
    -   Review loan-status distribution.
    -   Compare loan status with education and self-employment.
    -   Use a correlation heatmap to inspect relationships among
        numerical features.
3.  **Train/test split**
    -   80% training data and 20% test data.
    -   `random_state=42`.
4.  **Preprocessing**
    -   `StandardScaler` for numerical features.
    -   `OneHotEncoder(handle_unknown="ignore")` for categorical
        features.
    -   `LabelEncoder` for the target labels.
5.  **Modeling**
    -   Logistic Regression with L2 regularization.
    -   `C=1.0`, `solver="lbfgs"`, `max_iter=1000`.
    -   Grid search with stratified 5-fold cross-validation selected
        `C=1` from the tested values.
6.  **Evaluation**
    -   Accuracy, precision, recall, F1 score, and ROC AUC.
    -   Confusion matrix and classification report.

## Reported Results

The following results are from the current notebook run:

  Metric                               Train     Test
  --------------------------------- -------- --------
  Accuracy                            92.04%   90.52%
  Recall (class 1)                    89.50%   86.48%
  Precision (class 1)                 89.50%   87.86%
  F1 score (class 1)                  89.50%   87.16%
  ROC AUC as reported in notebook     91.54%   89.69%

Grid search reported: - **Best parameter:** `C = 1` - **Best
cross-validation F1:** `0.89445`

The classification report for the test set showed: - Class `0`:
precision `0.92`, recall `0.93`, F1 `0.92` (support: 536) - Class `1`:
precision `0.88`, recall `0.86`, F1 `0.87` (support: 318) - Overall
accuracy: `0.91` (854 test examples)

> Note: `LabelEncoder` assigns class IDs in alphabetical order. Check
> `encoder.classes_` to confirm which original label corresponds to
> class `0` and class `1`. Also, for a standard ROC AUC evaluation, use
> predicted probabilities (`predict_proba`) rather than hard class
> predictions.

## Streamlit App

The Streamlit interface allows a user to enter applicant details and
receive: - A predicted loan status. - The model's class probabilities.

The supplied app trains a preprocessing + Logistic Regression pipeline
from a CSV at startup. This is a clean, reproducible app workflow, but
its predictions can differ slightly from the notebook's saved test
results because it trains from the full CSV rather than loading the
notebook's already-fitted model.

### Run locally

1.  Download or clone this repository.
2.  Place `loan_approval_dataset.csv` in the same folder as `app.py`.
3.  Install dependencies:

``` bash
pip install -r requirements.txt
```

4.  Start the app:

``` bash
streamlit run app.py
```

## Repository Structure

``` text
loan-approval-prediction/
├── Loan_Approval.ipynb
├── app.py
├── requirements.txt
├── loan_approval_dataset.csv   # add locally; check dataset license/terms before redistribution
└── README.md
```

## Tools & Libraries

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Streamlit

## Limitations

-   This is an educational ML project, not a real lending decision
    system.
-   Predictions depend on the dataset's coverage and quality.
-   Model performance should be checked with cross-validation and a
    held-out test set.
-   A high correlation between features may indicate multicollinearity;
    it does not automatically mean a feature should be removed.
-   The Streamlit app's input ranges are interface defaults and should
    be validated against the dataset's actual ranges before deployment.

## Future Improvements

-   Use a single `Pipeline`/`ColumnTransformer` in both training and
    inference to keep preprocessing consistent.
-   Calculate ROC AUC using `predict_proba`.
-   Tune the decision threshold based on the project goal and the costs
    of false approvals/rejections.
-   Compare Logistic Regression with baseline and tree-based models.
-   Add automated tests and deploy the Streamlit app.

## Author

Khaled Elsafy

[GitHub profile](https://github.com/Khaledelsafy123)
