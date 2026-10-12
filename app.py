import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "loan_approval_dataset.csv"

NUM_COLS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CAT_COLS = ["education", "self_employed"]
TARGET = "loan_status"


@st.cache_resource
def load_and_train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Could not find loan_approval_dataset.csv. Put the CSV file in the same folder as app.py."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Remove the identifier because it is not a meaningful prediction feature.
    df = df.drop(columns=["loan_id"], errors="ignore")

    required = NUM_COLS + CAT_COLS + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in the CSV: {missing}")

    df = df.dropna(subset=required).copy()
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(str).str.strip()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model, sorted(y.unique().tolist()), df


try:
    model, target_classes, training_df = load_and_train_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant's information to predict the loan status.")
st.caption("This app trains a Logistic Regression model.")

with st.sidebar:
    st.header("Applicant Information")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self-employed", ["No", "Yes"])
    dependents = st.slider("Number of dependents", 0, 5, 2)
    income = st.number_input("Annual income", min_value=200_000, max_value=9_900_000, value=5_000_000, step=100_000)
    loan_amount = st.number_input("Loan amount", min_value=300_000, max_value=39_500_000, value=15_000_000, step=100_000)
    loan_term = st.slider("Loan term (years)", 2, 20, 10)
    cibil_score = st.slider("CIBIL score", 300, 900, 600)
    residential_assets = st.number_input("Residential assets value", min_value=-100_000, max_value=29_100_000, value=5_000_000, step=100_000)
    commercial_assets = st.number_input("Commercial assets value", min_value=0, max_value=19_400_000, value=3_000_000, step=100_000)
    luxury_assets = st.number_input("Luxury assets value", min_value=300_000, max_value=39_600_000, value=15_000_000, step=100_000)
    bank_assets = st.number_input("Bank asset value", min_value=0, max_value=14_700_000, value=5_000_000, step=100_000)

applicant = pd.DataFrame([{
    "no_of_dependents": dependents,
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets,
    "commercial_assets_value": commercial_assets,
    "luxury_assets_value": luxury_assets,
    "bank_asset_value": bank_assets,
    "education": education,
    "self_employed": self_employed,
}])

st.subheader("Applicant Summary")
st.dataframe(applicant, use_container_width=True, hide_index=True)

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    prediction = model.predict(applicant)[0]
    probabilities = model.predict_proba(applicant)[0]
    probability_by_class = dict(zip(model.named_steps["classifier"].classes_, probabilities))

    if prediction.strip().lower() == "approved":
        st.success(f"Prediction: {prediction}")
    else:
        st.warning(f"Prediction: {prediction}")

    st.subheader("Prediction Probabilities")
    probability_df = pd.DataFrame({
        "Loan Status": list(probability_by_class.keys()),
        "Probability": [float(v) for v in probability_by_class.values()]
    })
    probability_df["Probability (%)"] = (probability_df["Probability"] * 100).round(2)
    st.dataframe(probability_df[["Loan Status", "Probability (%)"]], use_container_width=True, hide_index=True)
    st.bar_chart(probability_df.set_index("Loan Status")["Probability"])

with st.expander("About this model"):
    st.write(f"Training rows used: {len(training_df):,}")
    st.write("Algorithm: Logistic Regression with L2 regularization (C=1.0).")
    st.write("Numeric features are standardized; categorical features are one-hot encoded.")
    st.write("The model uses the dataset's original loan_status labels. Predictions are estimates, not financial advice.")
