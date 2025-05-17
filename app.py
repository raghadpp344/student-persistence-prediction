import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV, cached for performance
@st.cache_data
def load_data():
    return pd.read_csv("train_preprocessed.csv")

# Build and train the model, cached to avoid retraining on each run
@st.cache_resource
def build_and_train_model(df):
    X = df.drop(columns=['Y'])
    y = df['Y']

    # Split dataset into training (90%) and testing (10%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing pipeline: scale numerical, one-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    # Compute class weights to handle class imbalance
    weights_arr = compute_class_weight(class_weight='balanced',
                                       classes=np.unique(y_train), y=y_train)
    weights = {cls: w for cls, w in zip(np.unique(y_train), weights_arr)}

    # Build full pipeline with preprocessing and Random Forest classifier
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=11,
            class_weight=weights
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set and generate metrics
    y_pred_test = model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "classification_report": report,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

    return model, metrics, cat_cols, num_cols

# --- Streamlit UI setup ---

st.set_page_config(page_title="🎓 Student Continuation Prediction", page_icon="🧠")
st.title("🎓 Student Continuation Prediction Model")

df = load_data()
model, metrics, cat_cols, num_cols = build_and_train_model(df)

st.subheader("Enter Student Data:")

user_input = {}

# Create input widgets for numeric columns
for col in num_cols:
    min_, max_ = float(df[col].min()), float(df[col].max())
    mean_ = float(df[col].mean())

    # For 'age' or 'day' columns, use integer input
    if "age" in col.lower() or "day" in col.lower():
        user_input[col] = st.number_input(
            col, min_value=int(min_), max_value=int(max_), value=int(mean_), step=1)
    else:
        user_input[col] = st.slider(col, min_, max_, mean_)

# Create select boxes for categorical columns
for col in cat_cols:
    user_input[col] = st.selectbox(col, list(df[col].unique()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ The student **will continue**")
    else:
        st.error("❌ The student **might not continue**")

# Show detailed model analysis inside an expander
with st.expander("Model Analysis)()"):
    st.write(f"Accuracy: {metrics['accuracy']:.2%}")
    
    precision = metrics["classification_report"]['1']['precision']
    recall = metrics["classification_report"]['1']['recall']
    f1 = metrics["classification_report"]['1']['f1-score']

    st.write(f"**Precision :** {precision:.2%}")
    st.write(f"**Recall :** {recall:.2%}")
    st.write(f"**F1-Score :** {f1:.2%}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Class Distribution")
    dist = df['Y'].value_counts(normalize=True).rename({1:'Not Continue', 0:'Continue'}) * 100
    st.bar_chart(dist)

# Show sample data preview
with st.expander("📊 Sample Data Preview"):
    st.dataframe(df.head())