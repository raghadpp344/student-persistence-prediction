# 🎓 Student Continuation Prediction

An interactive **Streamlit** application that predicts whether a student will **continue** or **drop out** based on demographic and academic data. The model is trained using a **cleaned dataset** and implemented with a **Random Forest classifier**.

---

## 📌 Objective

To build a machine learning model that accurately predicts **student persistence**, using historical data from the **[Measuring Student Persistence and Completion Rate](https://www.kaggle.com/competitions/measuring-student-persistence-and-completion-rate)** competition on Kaggle.

---

## 📁 Dataset

**Source:** Kaggle Competition

**Files used:**
- `train.csv`: Original raw dataset  
- `train_preprocessed.csv`: Cleaned dataset after preprocessing (handling missing values, encoding, etc.)

---

## 🧼 Data Preprocessing

The following steps were applied to prepare the data:

- Removed nulls and handled missing values  
- Identified numerical and categorical columns  
- Scaled numerical features using `StandardScaler`  
- Encoded categorical features with `OneHotEncoder`  
- Handled class imbalance using `compute_class_weight`

---

## 🧠 Model

**Algorithm:** `RandomForestClassifier`

**Parameters:**
- `n_estimators = 150`  
- `max_depth = 11`  
- `class_weight = balanced`

**Train/Test Split:**  
- 90% training  
- 10% testing

**Evaluation Metrics:**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix

---

## 💻 Streamlit App Features

- Interactive input interface for user to enter student data  
- Real-time prediction with clear output:
  - ✅ Student **will continue**
  - ❌ Student **might not continue**
- Model evaluation dashboard:
  - Accuracy, Precision, Recall, F1-score  
  - Confusion matrix heatmap  
  - Class distribution bar chart
- Dataset preview (first few rows of the cleaned data)

---

## ▶️ How to Run

1. Make sure `train_preprocessed.csv` is in the project root directory.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. run the app:
```bash
python -m streamlit run app.py
```

## 📦 Dependencies
pandas

numpy

scikit-learn

streamlit

matplotlib

seaborn

## 📂 Project Structure
.
├── train.csv                  # Raw dataset
├── train_preprocessed.csv     # Cleaned dataset
├── app.py                     # Main Streamlit app
├── project4.ipynb             # Development notebook
├── README.md                  # Project documentation
└── requirements.txt           # List of dependencies

