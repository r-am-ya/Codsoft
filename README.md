# Codsoft
# 💳 Credit Card Fraud Detection
This project is part of the **CodSoft Machine Learning Internship**.
🚀 Project Objective is
Detect fraudulent credit card transactions using machine learning algorithms like Logistic Regression and Random Forest.
📂 Dataset
Used is publicly available dataset from [Kaggle - Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
⚙️ Technologies :
- Python, Pandas, NumPy
- Scikit-learn, imbalanced-learn
- Matplotlib, Seaborn
📊 Workflow :
- Exploratory Data Analysis (EDA)
- Categorical Encoding & Feature Engineering
- SMOTE + Undersampling for class imbalance
- Trained Logistic Regression & Random Forest
- Visualized Feature Importances & Precision-Recall Curve
✅ Results (Random Forest):
- **Accuracy:** ~100%
- **AUC Score:** 0.98
- **F1-score (fraud):** 0.62
- Saved model as `fraud_detection_model.pkl`
 🧪 Sample Prediction:
Tested the model with unseen transactions using `joblib.load()`.
📸 Screenshots:
![Feature Importance](![feature_importance](https://github.com/user-attachments/assets/9e427064-aeb1-47a5-b8d6-44bb0ef666c4))  
![Precision-Recall](![precision_recall](https://github.com/user-attachments/assets/dc641f71-044f-45cf-8cc8-d22b5224de35))

